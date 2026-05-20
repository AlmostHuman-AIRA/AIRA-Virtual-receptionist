import asyncio
import base64
import json
import logging
import time
import io
import os
import wave
import re
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from silero_vad import load_silero_vad, get_speech_timestamps
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.tts_processor import KokoroTTSProcessor
from services.processor_service import process_text_for_client
from services.wake_word_service import get_wake_word_service

# --- Other imports remain the same ---
from services.face_recognition_service import verify_person_face, warmup_deepface
from services.person_detection_service import (
    get_person_detection_service,
    warmup_mediapipe,
)

# Thread pool for running blocking DeepFace and MediaPipe calls without blocking the async event loop.
_face_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="deepface")

# Trigger model warmup immediately when the server file is loaded
_face_executor.submit(warmup_deepface)
_face_executor.submit(warmup_mediapipe)

logging.basicConfig(
    # ... rest of your code ...
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
router = APIRouter()

vad_model = load_silero_vad()
WAKE_WORD = os.getenv("OPENWAKEWORD_WAKEWORD", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.getenv("OPENWAKEWORD_THRESHOLD", "0.35"))
ww_service = get_wake_word_service(WAKE_WORD)
ww_service.threshold = WAKE_WORD_THRESHOLD

SAMPLE_RATE = 16000
OWW_CHUNK_SAMPLES = 1280
OWW_CHUNK_BYTES = OWW_CHUNK_SAMPLES * 2
SILERO_WINDOW_SAMPLES = 8000
MAX_SILENCE_MS = 1200
FOLLOWUP_TIMEOUT_SECONDS = float(os.getenv("FOLLOWUP_TIMEOUT", "12.0"))
MAX_MISMATCH_STRIKES = 3

_NAME_STOPWORDS = frozenset(
    {
        "i",
        "i'm",
        "im",
        "it",
        "it's",
        "its",
        "i've",
        "ive",
        "hello",
        "hi",
        "hey",
        "sorry",
        "so",
        "not",
        "no",
        "yes",
        "me",
        "my",
        "is",
        "am",
        "are",
        "was",
        "the",
        "a",
        "an",
        "here",
        "there",
        "this",
        "that",
        "just",
        "now",
        "please",
        "name",
        "call",
        "called",
        "actually",
        "really",
        "well",
        "oh",
        "ah",
        "um",
        "uh",
        "ok",
        "okay",
        "yeah",
        "yep",
        "and",
        "from",
        "for",
        "with",
        "but",
        "or",
        "to",
        "at",
        "in",
        "on",
        "of",
        "could",
        "would",
        "should",
        "can",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "need",
        "want",
        "like",
        "please",
        "help",
        "hr",
        "interview",
        "delivery",
        "meeting",
        "schedule",
        "visit",
        "here",
    }
)

# Keywords that indicate the speaker is a visitor/delivery person, not an employee
_VISITOR_KEYWORDS = (
    "delivery",
    "courier",
    "package",
    "parcel",
    "swiggy",
    "zomato",
    "amazon",
    "flipkart",
    "dunzo",
    "porter",
    "blinkit",
    "zepto",
    "visitor",
    "guest",
    "appointment",
    "here to meet",
    "here to see",
    "meeting with",
    "visiting",
    "i have a delivery",
    "drop off",
    "interview",
    "interviewing",
    "intern",
    "internship",
    "joining",
    "new joiner",
    "onboarding",
    "new employee",
    "starting today",
    "interview",
    "interviewing",
    "intern",
    "internship",
    "delivery",
    "courier",
    "package",
    "visitor",
    "guest",
    "joining",
    "onboarding",
    "appointment",
    "visiting",
)


def _detect_person_type(text: str) -> str:
    """
    Returns 'visitor' if the transcript contains delivery/visitor keywords,
    otherwise returns 'employee'.
    """
    lower = text.lower()
    for keyword in _VISITOR_KEYWORDS:
        if keyword in lower:
            return "visitor"
    return "employee"


def _extract_spoken_name(text: str) -> str | None:
    """
    Try to extract a candidate name from phrases like:
      - "I'm John Doe"
      - "I am John Doe"
      - "This is John Doe"
      - "My name is John Doe"
      - "Lucy here" / "It's Lucy here"
    """
    if not text:
        return None

    # Capture exactly 1 or 2 words immediately following the intro phrase.
    # Capture 1-2 words: each word is letters/hyphens/apostrophes only.
    # No dots allowed at end of token — prevents "here." being swallowed.
    name_pattern = r"([A-Za-z'\-]+(?:\s+(?!and\b|or\b|but\b|for\b|from\b|that\b|who\b|which\b)[A-Za-z'\-]+)?)"

    patterns = [
        rf"\b(?:i am|i'm)\s+{name_pattern}",
        rf"\bmy name is\s+{name_pattern}",
        rf"\bthis is\s+{name_pattern}",
        rf"\bit'?s\s+{name_pattern}",
        rf"\b([a-zA-Z'-]+)\s+here\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?:;")
            # Strip trailing "here" if captured
            name = re.sub(r"\s+here$", "", name, flags=re.IGNORECASE).strip()

            # --- UPDATED LOGIC ---
            # Reject if the whole phrase is a stopword, OR if the first word is a stopword
            if name.lower() not in _NAME_STOPWORDS and len(name) >= 2:
                first_word = name.split()[0].lower()
                if first_word not in _NAME_STOPWORDS:
                    return name
            # ---------------------

    # Fallback for short direct intros like "John" or "John Doe".
    cleaned = re.sub(r"[^a-zA-Z\s.'-]", " ", text).strip()
    if cleaned:
        words = [w for w in cleaned.split() if w]
        # If the whole utterance is very short (1-3 words)
        if 1 <= len(words) <= 3:
            name_words = [w for w in words if w.lower() not in _NAME_STOPWORDS]
            if 1 <= len(name_words) <= 2 and all(w[0].isupper() for w in name_words):
                return " ".join(name_words)
    return None


def _candidate_names_from_transcript(text: str) -> list[str]:
    if not text:
        return []

    # Strip meeting-target names FIRST, before any extraction runs.
    # Prevents "I'm here to meet Priya" or "schedule a meeting with Lucy"
    # from extracting the HOST name as the speaker's name.
    safe_text = re.sub(
        r"\b(?:meet|see|looking for|appointment with|meeting with|schedule\s+\w+\s+with|here for|here to meet|visiting|talk to|speak to|speak with)\s+([A-Z][a-z.\'-]+)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    candidates: list[str] = []

    # 1. Try explicit intro phrases — on safe_text, not original text
    primary = _extract_spoken_name(safe_text)
    if primary:
        candidates.append(primary)
        parts = primary.split()
        if len(parts) > 1:
            candidates.append(parts[0])

    # 2. Fallback: capitalized words — also on safe_text
    capitalized_words = re.findall(r"\b[A-Z][a-z.\'-]+\b", safe_text)
    for cw in capitalized_words:
        if len(cw) >= 3 and cw.lower() not in _NAME_STOPWORDS:
            candidates.append(cw)

    # Preserve order, remove duplicates
    seen = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate.strip())

    return unique


def _resolve_employee_name(candidate_name: str) -> str | None:
    # Reject obvious non-names before touching the DB to prevent fuzzy false matches
    # e.g. "I'm" fuzzy-matching to "Jim"
    if not candidate_name or len(candidate_name) < 3:
        return None
    if candidate_name.lower() in _NAME_STOPWORDS:
        return None
    try:
        from receptionist.database import get_employee_by_name

        employee = get_employee_by_name(candidate_name)
        if employee:
            return employee.name
    except Exception:
        return None
    return None


def create_wav_from_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return wav_io.getvalue()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    from services.query_router import clear_session_state

    clear_session_state(client_id)
    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()
    text_queue: asyncio.Queue[str] = asyncio.Queue()
    session_state = {
        "mode": "PASSIVE",
        "awaiting_face": False,
        "is_verified": False,
        "visitor_reference_image_b64": None,
        "pending_identity_name": None,
        "person_type": "employee",
        "mismatch_strikes": 0,
        "face_verify_in_progress": False,
        "conversation_complete": False,
        "visitor_captured": False,  # ← ADD
        "brain_is_thinking": False,  # ← ADD
        "presence_count": 0,
        "last_presence_trigger": 0.0,
    }

    try:
        await websocket.send_text(
            json.dumps(
                {"status": "connected", "client_id": client_id, "state": "passive"}
            )
        )

        async def send_keepalive():
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)
                except WebSocketDisconnect:
                    break
                except RuntimeError:
                    break
                except Exception:
                    await asyncio.sleep(2)

        async def listener():
            oww_carry = bytearray()
            audio_buffer = bytearray()
            speech_seen = False
            followup_entered_at = (
                time.time()
            )  # safe default — prevents stale timeout on first FOLLOWUP
            previous_mode = session_state["mode"]
            bytes_received_count = 0

            while True:
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    logger.info(f"Client {client_id} disconnected normally.")
                    break
                except RuntimeError as e:
                    if 'Cannot call "receive"' in str(e):
                        logger.info(f"WebSocket {client_id} closed cleanly.")
                        break
                    logger.error(f"WebSocket RuntimeError: {e}")
                    break

                current_mode = session_state["mode"]

                if current_mode == "PASSIVE" and previous_mode != "PASSIVE":
                    oww_carry.clear()
                    audio_buffer.clear()
                    speech_seen = False
                    followup_entered_at = (
                        time.time()
                    )  # reset so next session can't inherit a stale clock
                    session_state["is_verified"] = False
                    session_state["verified_name"] = None  # <--- ADD THIS
                    session_state["visitor_reference_image_b64"] = None
                    session_state["visitor_captured"] = False  # ← ADD THIS
                    session_state["pending_identity_name"] = None
                    session_state["person_type"] = "employee"
                    session_state["mismatch_strikes"] = 0
                    session_state["conversation_complete"] = False
                    session_state["presence_count"] = 0

                    try:
                        ww_service.model.reset()
                    except Exception:
                        pass
                elif current_mode == "FOLLOWUP" and previous_mode != "FOLLOWUP":
                    audio_buffer.clear()
                    speech_seen = False
                    bytes_received_count = 0
                    logger.info(
                        f"[{client_id}] Entered FOLLOWUP/LISTENING mode. Waiting for audio..."
                    )
                    # Don't start the timeout clock while waiting for capture_reference.
                    # The camera round-trip can take seconds on slow hardware.
                    # Clock starts once capture completes and pending_text is queued.
                    if not session_state.get("awaiting_face"):
                        followup_entered_at = time.time()

                previous_mode = current_mode

                raw_bytes = message.get("bytes") or message.get("data")
                raw_text = message.get("text")

                if raw_text:
                    try:
                        msg = json.loads(raw_text)

                        if msg.get("type") == "presence_frame":
                            if session_state["mode"] != "PASSIVE":
                                continue

                            # Cooldown: skip if we triggered recently (prevent rapid re-activation)
                            PRESENCE_FRAME_COOLDOWN = float(
                                os.getenv("PRESENCE_FRAME_COOLDOWN", "9.0")
                            )
                            if (
                                time.time() - session_state["last_presence_trigger"]
                                < PRESENCE_FRAME_COOLDOWN
                            ):
                                continue

                            image_b64 = msg.get("image_b64", "")
                            if not image_b64:
                                continue

                            # Run MediaPipe detection in thread pool (non-blocking)
                            detection_service = get_person_detection_service()
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                _face_executor,
                                lambda: detection_service.detect_person(image_b64),
                            )

                            if result["detected"]:
                                session_state["presence_count"] += 1
                                logger.info(
                                    f"[{client_id}] Person detected (count={session_state['presence_count']}, "
                                    f"confidence={result['confidence']:.2f}, face_ratio={result['face_ratio']:.3f})"
                                )

                                PRESENCE_CONFIRM_FRAMES = int(
                                    os.getenv("PRESENCE_CONFIRM_FRAMES", "2")
                                )
                                if (
                                    session_state["presence_count"]
                                    >= PRESENCE_CONFIRM_FRAMES
                                ):
                                    logger.info(
                                        f"[{client_id}] ✅ Person confirmed! Activating AIRA."
                                    )
                                    session_state["presence_count"] = 0
                                    session_state["last_presence_trigger"] = time.time()
                                    session_state["mode"] = "PROCESSING"

                                    # Notify frontend
                                    await websocket.send_text(
                                        json.dumps({"type": "person_detected"})
                                    )

                                    # Reuse existing wake-up flow
                                    await text_queue.put("WAKE_WORD_TRIGGERED")
                                    oww_carry.clear()
                                    audio_buffer.clear()
                                    speech_seen = False
                            else:
                                # No face → reset consecutive count
                                session_state["presence_count"] = 0
                            continue

                        # ── Face verification request from frontend ──────────────
                        # Triggered when the employee says their name and LLM identifies them.
                        # Frontend sends: { type: "verify_face", audio_name: "John", image_b64: "..." }
                        if msg.get("type") == "verify_face":
                            if session_state["mode"] == "PASSIVE":
                                logger.info(...)
                                continue
                            if session_state["mode"] in (
                                "PROCESSING",
                                "SPEAKING",
                            ):  # ← ADD THIS
                                session_state["face_verify_in_progress"] = False
                                continue
                            if session_state.get("face_verify_in_progress"):
                                continue
                            if session_state.get("conversation_complete"):
                                continue
                            session_state["face_verify_in_progress"] = True

                            audio_name = msg.get("audio_name", "")
                            image_b64 = msg.get("image_b64", "")
                            # Always trust server-side session_state over frontend-sent person_type
                            # (frontend may be hardcoded to "employee")
                            person_type = session_state.get("person_type", "employee")
                            session_action = msg.get("session_action") or (
                                "capture_reference"
                                if person_type == "visitor"
                                else "compare_reference"
                            )
                            session_state["awaiting_face"] = False
                            logger.info(
                                f"[{client_id}] Face verification requested for: '{audio_name}' "
                                f"(person_type={person_type}, action={session_action})"
                            )

                            if (
                                person_type == "visitor"
                                and session_action == "capture_reference"
                            ):
                                # ── VISITOR: Use DB-backed photo deduplication ──────
                                # On first visit: saves the photo to DB
                                # On return visit: compares live frame against stored photo
                                loop = asyncio.get_event_loop()
                                result = await loop.run_in_executor(
                                    _face_executor,
                                    lambda: verify_person_face(
                                        person_type="visitor",
                                        audio_name=audio_name,
                                        image_b64=image_b64,
                                    ),
                                )

                                session_state["is_verified"] = True
                                session_state["visitor_captured"] = True
                                session_state["pending_identity_name"] = None

                                try:
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "face_verification_result",
                                                "verified": result.get(
                                                    "verified", True
                                                ),
                                                "distance": result.get(
                                                    "distance", -1.0
                                                ),
                                                "audio_name": audio_name,
                                                "has_photo": result.get(
                                                    "has_photo", True
                                                ),
                                                "message": result.get("message", ""),
                                                "person_type": person_type,
                                                "session_action": session_action,
                                                "reference_captured": True,
                                                "is_new": result.get("is_new", True),
                                                "visitor_id": result.get("visitor_id"),
                                            }
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"[{client_id}] Could not send visitor result. Error: {e}"
                                    )
                                    break

                                # AFTER:
                                pending_text = session_state.get("pending_text")
                                if pending_text:
                                    session_state["mode"] = (
                                        "PROCESSING"  # ← block listener while LLM runs
                                    )
                                    await websocket.send_text(
                                        json.dumps({"state": "processing"})
                                    )
                                    await text_queue.put(pending_text)
                                    session_state["pending_text"] = None
                                    logger.info(
                                        f"[{client_id}] Visitor photo processed — queuing LLM response."
                                    )
                                else:
                                    session_state["mode"] = "PASSIVE"
                                    await websocket.send_text(
                                        json.dumps({"state": "passive"})
                                    )

                                session_state["face_verify_in_progress"] = False
                                # Visitor capture/compare is done — stop accepting face frames
                                session_state["conversation_complete"] = True
                                continue

                            was_already_verified = session_state.get(
                                "is_verified", False
                            )
                            # Run DeepFace in a thread pool (it's blocking/CPU-intensive)
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                _face_executor,
                                lambda: verify_person_face(
                                    person_type=person_type,
                                    audio_name=audio_name,
                                    image_b64=image_b64,
                                ),
                            )
                            session_state["face_verify_in_progress"] = False
                            if result.get("verified"):
                                session_state["is_verified"] = True
                                session_state["verified_name"] = (
                                    audio_name  # <--- ADD THIS
                                )
                                session_state["pending_identity_name"] = None
                                # Face verified — restart the FOLLOWUP timeout clock now
                                # (it was parked at +3600 while we waited for the camera frame)
                                followup_entered_at = time.time()
                            elif person_type == "employee":
                                session_state["is_verified"] = False
                                session_state["pending_identity_name"] = audio_name
                                # Restart the clock even on mismatch so the user gets a
                                # chance to hear the rejection message and try again
                                followup_entered_at = time.time()

                            logger.info(
                                f"[{client_id}] Face verify result for '{audio_name}': "
                                f"verified={result['verified']}, distance={result['distance']}"
                            )

                            # Send result back to frontend (for the UI badge)
                            # --- CALCULATE STRIKES & FRONTEND STATE BEFORE SENDING ---
                            frontend_verified = result["verified"]
                            speak_message = False

                            if result["has_photo"] and result["message"]:
                                if result["verified"]:
                                    # ── Successful match ──────────────────────
                                    session_state["mismatch_strikes"] = 0
                                    if not speech_seen:
                                        followup_entered_at = time.time()
                                        audio_buffer.clear()
                                    if not was_already_verified:
                                        logger.info(
                                            f"[{client_id}] Initial face match — queueing confirmation."
                                        )
                                        speak_message = True

                                else:
                                    # ── Real mismatch OR No face in frame (person ducked out) ──
                                    reason = (
                                        "No face detected"
                                        if not result.get("face_detected", True)
                                        else "Mismatch"
                                    )

                                    if not was_already_verified:
                                        # INITIAL VERIFICATION: Reject immediately and speak!
                                        logger.info(
                                            f"[{client_id}] Initial {reason} — rejecting immediately."
                                        )
                                        speak_message = True
                                        session_state["mismatch_strikes"] = 0
                                        session_state["is_verified"] = False
                                        frontend_verified = False
                                    else:
                                        # CONTINUOUS VERIFICATION: Use strike/debounce logic
                                        session_state["mismatch_strikes"] += 1
                                        strikes = session_state["mismatch_strikes"]

                                        if strikes >= MAX_MISMATCH_STRIKES:
                                            logger.info(
                                                f"[{client_id}] {reason} Strike {strikes} — Revoking verification."
                                            )
                                            speak_message = True
                                            session_state["mismatch_strikes"] = 0
                                            session_state["is_verified"] = False
                                            frontend_verified = False
                                        else:
                                            logger.info(
                                                f"[{client_id}] {reason} Strike {strikes} — Debouncing. Hiding from frontend."
                                            )
                                            frontend_verified = (
                                                True  # Keep UI green during debounce
                                            )
                                            if not speech_seen:
                                                followup_entered_at = time.time()
                                                audio_buffer.clear()
                            # --- NOW SEND TO FRONTEND ---
                            try:
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "face_verification_result",
                                            "verified": frontend_verified,  # <-- Uses the masked value
                                            "distance": result["distance"],
                                            "audio_name": audio_name,
                                            "has_photo": result["has_photo"],
                                            "message": result.get("message", ""),
                                            "person_type": person_type,
                                            "session_action": session_action,
                                        }
                                    )
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[{client_id}] Could not send face result. Error: {e}"
                                )
                                break

                            # --- HANDLE AI SPEECH & UI STATE ---
                            if speak_message:
                                # Add SYSTEM: prefix to bypass the LLM
                                await text_queue.put(f"SYSTEM:{result['message']}")
                                session_state["mode"] = "FOLLOWUP"
                                await websocket.send_text(
                                    json.dumps({"state": "listening"})
                                )
                            elif was_already_verified:
                                # If it was a silent debounce, OR a successful continuous match, keep listening
                                session_state["mode"] = "FOLLOWUP"
                                await websocket.send_text(
                                    json.dumps({"state": "listening"})
                                )

                                # --------------------------
                        # ── Stop-speaking control message ────────────────────────
                        if msg.get("action") == "stop_speaking":
                            session_state["mode"] = "PASSIVE"
                            session_state["awaiting_face"] = False
                            session_state["visitor_reference_image_b64"] = None
                            session_state["pending_identity_name"] = None
                            await websocket.send_text(json.dumps({"state": "passive"}))
                    except Exception:
                        pass
                    continue

                if not raw_bytes:
                    continue

                bytes_received_count += 1
                if bytes_received_count == 1:
                    logger.info(
                        f"[{client_id}] Successfully receiving audio stream from frontend (Mode: {current_mode})"
                    )

                if session_state["mode"] in ("PROCESSING", "SPEAKING"):
                    continue
                if session_state.get("face_verify_in_progress") or session_state.get(
                    "awaiting_face"
                ):
                    audio_buffer.clear()
                    speech_seen = False
                    continue

                if session_state["mode"] == "PASSIVE":
                    oww_carry.extend(raw_bytes)
                    while len(oww_carry) >= OWW_CHUNK_BYTES:
                        chunk = bytes(oww_carry[:OWW_CHUNK_BYTES])
                        oww_carry = oww_carry[OWW_CHUNK_BYTES:]
                        triggered, score = ww_service.process_chunk(chunk)
                        if triggered:
                            logger.info(
                                f"[{client_id}] Wake word triggered! Score: {score}"
                            )
                            session_state["mode"] = "PROCESSING"
                            await text_queue.put("WAKE_WORD_TRIGGERED")
                            oww_carry.clear()
                            audio_buffer.clear()
                            speech_seen = False
                            break

                elif session_state["mode"] in ("ACTIVE", "FOLLOWUP"):
                    # Time out back to PASSIVE if they don't say anything
                    if (
                        current_mode == "FOLLOWUP"
                        and not speech_seen
                        and not session_state.get("brain_is_thinking")  # ← ADD
                        and (
                            time.time() - followup_entered_at > FOLLOWUP_TIMEOUT_SECONDS
                        )
                    ):
                        logger.info(
                            f"[{client_id}] Followup timeout reached (no speech detected). Returning to PASSIVE."
                        )
                        session_state["conversation_complete"] = (
                            True  # Stop face frames immediately
                        )
                        session_state["mode"] = "PASSIVE"
                        session_state["awaiting_face"] = False
                        await websocket.send_text(json.dumps({"state": "passive"}))
                        continue

                    audio_buffer.extend(raw_bytes)
                    audio_np = (
                        np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    if len(audio_np) >= SILERO_WINDOW_SAMPLES:
                        speech = get_speech_timestamps(
                            audio_np,
                            vad_model,
                            sampling_rate=SAMPLE_RATE,
                            min_speech_duration_ms=200,
                            min_silence_duration_ms=250,
                            return_seconds=False,
                        )

                        if speech:
                            if not speech_seen:
                                logger.info(f"[{client_id}] VAD detected speech start.")
                                speech_seen = True

                            last_speech_end = speech[-1]["end"]
                            total_samples = len(audio_np)
                            current_silence_ms = (
                                (total_samples - last_speech_end) / SAMPLE_RATE
                            ) * 1000

                            if current_silence_ms >= MAX_SILENCE_MS:
                                logger.info(
                                    f"[{client_id}] VAD detected speech end (User paused for {current_silence_ms:.0f}ms). Processing..."
                                )
                                session_state["mode"] = "PROCESSING"
                                await websocket.send_text(
                                    json.dumps({"state": "processing"})
                                )
                                wav_bytes = create_wav_from_pcm(bytes(audio_buffer))
                                text = await whisper_processor.transcribe_audio(
                                    wav_bytes
                                )

                                logger.info(
                                    f"[{client_id}] Whisper transcribed: '{text}'"
                                )

                                if not text or text in ("NOISE_DETECTED", "NO_SPEECH"):
                                    logger.info(
                                        f"[{client_id}] Ignored noise/silence. Returning to FOLLOWUP."
                                    )
                                    session_state["mode"] = "FOLLOWUP"

                                    # --- ADD THESE 3 LINES ---
                                    audio_buffer.clear()
                                    speech_seen = False
                                    followup_entered_at = time.time()
                                    # -------------------------

                                    await websocket.send_text(
                                        json.dumps({"state": "listening"})
                                    )
                                    continue

                                if text and text not in ("NOISE_DETECTED", "NO_SPEECH"):
                                    text_lower = text.lower()
                                    if any(
                                        w in text_lower
                                        for w in [
                                            "thank you",
                                            "thanks",
                                            "bye",
                                            "goodbye",
                                            "that's all",
                                        ]
                                    ):
                                        session_state["conversation_complete"] = True
                                        await text_queue.put(text)
                                        continue

                                    employee_name = None

                                    if not session_state.get(
                                        "is_verified"
                                    ) and not session_state.get("visitor_captured"):
                                        known_name = session_state.get("verified_name")
                                        if known_name:
                                            employee_name = known_name
                                            logger.info(
                                                f"[{client_id}] Face tracking was lost. Re-verifying known user: '{employee_name}'"
                                            )
                                        else:
                                            detected_person_type = _detect_person_type(
                                                text
                                            )
                                            session_state["person_type"] = (
                                                detected_person_type
                                            )

                                            candidates = (
                                                _candidate_names_from_transcript(text)
                                            )
                                            loop = asyncio.get_event_loop()

                                            if detected_person_type == "visitor":
                                                # For visitors, only use the extracted name if it came
                                                # from an explicit self-introduction phrase
                                                # ("I am X", "my name is X", etc.).
                                                # The capitalized-word fallback can pick up the host
                                                # name ("schedule a meeting with Lucy" → "Lucy"),
                                                # so we re-run the strict extractor on safe_text only.
                                                safe_text_for_name = re.sub(
                                                    r"\b(?:meet|see|looking for|appointment with|meeting with|schedule\s+\w+\s+with|here for|here to meet|visiting|talk to|speak to|speak with)\s+([A-Z][a-z.\'-]+)\b",
                                                    "",
                                                    text,
                                                    flags=re.IGNORECASE,
                                                )
                                                strict_name = _extract_spoken_name(
                                                    safe_text_for_name
                                                )
                                                if strict_name:
                                                    employee_name = strict_name
                                                    logger.info(
                                                        f"[{client_id}] Visitor detected. "
                                                        f"Speaker name extracted: '{employee_name}' "
                                                        f"(skipping DB lookup)"
                                                    )
                                                else:
                                                    # No explicit self-intro found — don't guess a name.
                                                    # The LLM will ask for it naturally.
                                                    employee_name = None
                                                    logger.info(
                                                        f"[{client_id}] Visitor detected but no explicit "
                                                        f"self-introduction found — skipping name extraction."
                                                    )
                                            else:
                                                for candidate in candidates:
                                                    employee_name = (
                                                        await loop.run_in_executor(
                                                            _face_executor,
                                                            _resolve_employee_name,
                                                            candidate,
                                                        )
                                                    )
                                                    if employee_name:
                                                        logger.info(
                                                            f"[{client_id}] Employee identified from candidate '{candidate}' as '{employee_name}'"
                                                        )
                                                        break

                                    # If we found a name, trigger face capture/verification
                                    # If we found a name, trigger face capture/verification
                                    if employee_name:
                                        person_type = session_state["person_type"]
                                        logger.info(
                                            f"[{client_id}] identity_detected emitted: "
                                            f"'{employee_name}' (person_type={person_type})"
                                        )
                                        session_state["is_verified"] = False
                                        session_state["awaiting_face"] = True
                                        session_state["pending_identity_name"] = (
                                            employee_name
                                        )
                                        session_state["pending_text"] = text

                                        # Switch to FOLLOWUP so face frames are not blocked
                                        # by the PROCESSING guard, and clear the audio buffer
                                        # so the same utterance is not re-processed in a loop.
                                        # NOTE: followup_entered_at is NOT set here.
                                        # awaiting_face=True means the FOLLOWUP transition guard
                                        # will defer the clock until capture_reference completes.
                                        session_state["mode"] = "FOLLOWUP"
                                        audio_buffer.clear()
                                        speech_seen = False
                                        # Park the timeout clock far in the future so the
                                        # FOLLOWUP timeout check cannot fire while we are
                                        # waiting for the camera frame from the frontend.
                                        # The clock is restarted in two places:
                                        #   • capture_reference handler (visitor path)
                                        #   • after face_verify_in_progress clears (employee path)
                                        followup_entered_at = time.time() + 3600
                                        await websocket.send_text(
                                            json.dumps({"state": "listening"})
                                        )

                                        try:
                                            await websocket.send_text(
                                                json.dumps(
                                                    {
                                                        "type": "employee_identified",
                                                        "name": employee_name,
                                                        "person_type": person_type,
                                                        "session_action": (
                                                            "capture_reference"
                                                            if person_type == "visitor"
                                                            else "compare_reference"
                                                        ),
                                                    }
                                                )
                                            )
                                        except WebSocketDisconnect:
                                            logger.warning(
                                                f"[{client_id}] Client disconnected before face request could be sent."
                                            )
                                            break
                                        except Exception as e:
                                            logger.error(
                                                f"[{client_id}] Failed to send face request: {e}"
                                            )
                                    elif session_state.get("pending_identity_name"):
                                        # Add SYSTEM: prefix to bypass the LLM
                                        await text_queue.put(
                                            "SYSTEM:Please complete face verification before continuing."
                                        )
                                    else:
                                        # Already verified OR no name found — chat normally
                                        await text_queue.put(text)

        async def brain():
            while True:
                text = await text_queue.get()

                # --- 1. DETECT TERMINAL STATE ---
                is_terminal = False
                if (
                    text
                    and text
                    not in (
                        "WAKE_WORD_TRIGGERED",
                        "SYSTEM:Please complete face verification before continuing.",
                    )
                    and not text.startswith("SYSTEM:")
                ):
                    text_lower = text.lower()
                    if any(
                        w in text_lower
                        for w in ["thank you", "thanks", "bye", "goodbye", "that's all"]
                    ):
                        is_terminal = True
                        session_state["conversation_complete"] = True
                # --------------------------------

                manager.client_state[client_id] = "THINKING"
                session_state["brain_is_thinking"] = True

                # --- BYPASS LLM FOR SYSTEM MESSAGES ---
                if text and text.startswith("SYSTEM:"):
                    reply_text = text.split("SYSTEM:", 1)[1]
                else:
                    # --- INJECT USER IDENTITY INTO LLM PROMPT ---
                    verified_name = session_state.get("verified_name")
                    if verified_name:
                        # Secretly tell the LLM who is speaking
                        prompt_text = f"[System Note: The user speaking is verified as {verified_name}] {text}"
                    else:
                        prompt_text = text

                    reply_text = await process_text_for_client(client_id, prompt_text)
                    session_state["brain_is_thinking"] = False

                logger.info(f"[{client_id}] AI Response generated: '{reply_text}'")

                # --- THE TRY BLOCK STARTS HERE ---
                try:
                    if not reply_text or not reply_text.strip():
                        session_state["mode"] = "PASSIVE"
                        await websocket.send_text(json.dumps({"state": "passive"}))
                        continue

                    audio, word_timings = (
                        await tts_processor.synthesize_remaining_speech_with_timing(
                            reply_text
                        )
                    )
                    if audio is not None and len(audio) > 0:
                        session_state["mode"] = "SPEAKING"
                        manager.client_state[client_id] = "SPEAKING"
                        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                        wav_bytes = create_wav_from_pcm(audio_bytes, sample_rate=24000)
                        b64 = base64.b64encode(wav_bytes).decode("utf-8")
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "audio": b64,
                                    "word_timings": word_timings,
                                    "sample_rate": 24000,
                                    "method": "native_kokoro_timing",
                                    "state": "speaking",
                                }
                            )
                        )

                        # Wait for audio to physically finish playing on the frontend
                        await asyncio.sleep((len(audio) / 24000.0) + 0.5)

                    # --- 2. ROUTE TO CORRECT MODE ---
                    if is_terminal:
                        logger.info(
                            f"[{client_id}] Terminal phrase detected. Returning to PASSIVE mode."
                        )
                        session_state["mode"] = "PASSIVE"
                        session_state["awaiting_face"] = False
                        await websocket.send_text(json.dumps({"state": "passive"}))
                    else:
                        # Give the user 12 seconds to reply after Jarvis speaks
                        session_state["mode"] = "FOLLOWUP"
                        await websocket.send_text(json.dumps({"state": "listening"}))
                    # --------------------------------

                except (RuntimeError, WebSocketDisconnect):
                    # This catches the error if the user closes the tab
                    logger.info(
                        f"[{client_id}] Client disconnected before AI could finish responding."
                    )
                    break
                finally:
                    session_state["brain_is_thinking"] = False
                    # This ensures the queue is ALWAYS marked as done, even if it errors or continues
                    text_queue.task_done()

        listener_task = asyncio.create_task(listener())
        brain_task = asyncio.create_task(brain())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Only listener or brain dying should end the session
        # Keepalive dying alone must NOT kill everything
        await asyncio.wait(
            [listener_task, brain_task], return_when=asyncio.FIRST_COMPLETED
        )
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
    finally:
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
