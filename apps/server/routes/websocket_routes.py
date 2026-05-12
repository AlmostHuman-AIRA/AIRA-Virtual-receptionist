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
from services.face_recognition_service import verify_employee_face
from services.person_detection_service import (
    get_person_detection_service,
    warmup_mediapipe,
)

# Thread pool for blocking calls: DeepFace + MediaPipe (2 workers)
_face_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="deepface")

# Warm up MediaPipe face detection model in background at startup
_face_executor.submit(warmup_mediapipe)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s[%(levelname)s] %(name)s: %(message)s"
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


# Keywords that indicate the speaker is naming their HOST, not themselves.
# If the transcript contains these words, we must NOT trigger employee face-verify.
_HOST_INTENT_PHRASES = re.compile(
    r"\b(here to see|here to meet|i(?:'m| am) here for|looking for|meeting with|appointment with|see the|meet the)",
    re.IGNORECASE,
)

# Job-title / role words — a name composed entirely of these is NOT a person name.
_ROLE_TITLE_WORDS = {
    "manager",
    "director",
    "ceo",
    "cto",
    "coo",
    "cfo",
    "president",
    "hr",
    "dhs",
    "admin",
    "head",
    "lead",
    "chief",
    "officer",
    "vp",
    "vice",
    "senior",
    "junior",
    "associate",
    "assistant",
    "supervisor",
    "receptionist",
    "secretary",
    "coordinator",
    "executive",
    "intern",
    "engineer",
    "developer",
    "analyst",
    "accountant",
    "consultant",
}


def _is_self_introduction(text: str) -> bool:
    """
    Return True only when the speaker is introducing THEMSELVES —
    not when they are naming who they want to visit.
    """
    if not text:
        return False
    # If the sentence contains host-intent language, the person is naming a host.
    if _HOST_INTENT_PHRASES.search(text):
        return False
    # Must contain a self-intro trigger phrase.
    return bool(
        re.search(
            r"\b(i am|i'?m|my name is|this is|i'?m called)\b",
            text,
            re.IGNORECASE,
        )
    )


def _extract_spoken_name(text: str) -> str | None:
    """
    Try to extract a candidate name from phrases like:
      - "I'm John Doe"
      - "I am John Doe"
      - "This is John Doe"
      - "My name is John Doe"
    """
    if not text:
        return None
    patterns = [
        r"\b(?:i am|i'm)\s+([a-zA-Z][a-zA-Z\s.'-]{1,60})",
        r"\bmy name is\s+([a-zA-Z][a-zA-Z\s.'-]{1,60})",
        r"\bthis is\s+([a-zA-Z][a-zA-Z\s.'-]{1,60})",
        r"\b([a-zA-Z][a-zA-Z\s.'-]{1,60})\s+here\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?:;")

    # Fallback for short direct intros like "John" or "John Doe".
    cleaned = re.sub(r"[^a-zA-Z\s.'-]", " ", text).strip()
    if cleaned:
        words = [w for w in cleaned.split() if w]
        if 1 <= len(words) <= 2:
            return " ".join(words)
    return None


def _is_role_only_name(name: str) -> bool:
    """Return True if every word in the name is a job-title keyword (not a real name)."""
    words = {w.lower().strip() for w in name.split() if w.strip()}
    return bool(words) and words.issubset(_ROLE_TITLE_WORDS)


def _candidate_names_from_transcript(text: str) -> list[str]:
    """
    Build a short list of candidate names from transcript text.
    This improves robustness when STT adds extra words around the name.
    """
    if not text:
        return []

    candidates: list[str] = []
    primary = _extract_spoken_name(text)
    if primary:
        candidates.append(primary)

    # Try common sub-parts from the first intro phrase chunk.
    cleaned = re.sub(r"[^a-zA-Z\s.'-]", " ", text).strip()
    words = [w for w in cleaned.split() if w]
    if words:
        # Full 2-word option and first-name fallback.
        if len(words) >= 2:
            candidates.append(f"{words[0]} {words[1]}")
        candidates.append(words[0])

    # Preserve order while removing duplicates.
    seen = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate.strip())
    return unique


def _resolve_employee_name(candidate_name: str) -> str | None:
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
    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()
    text_queue: asyncio.Queue[str] = asyncio.Queue()
    session_state = {
        "mode": "PASSIVE",
        "awaiting_face": False,
        "presence_count": 0,  # consecutive frames with face detected
        "last_presence_trigger": 0.0,  # timestamp of last camera-triggered activation
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
                except Exception:
                    break

        async def listener():
            oww_carry = bytearray()
            audio_buffer = bytearray()
            speech_seen = False
            followup_entered_at = 0.0
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
                    session_state["presence_count"] = 0  # reset on every PASSIVE entry
                    # NOTE: last_presence_trigger is intentionally NOT reset here —
                    # the 8-second cooldown should persist across mode transitions.
                    try:
                        ww_service.model.reset()
                    except Exception:
                        pass
                elif current_mode == "FOLLOWUP" and previous_mode != "FOLLOWUP":
                    audio_buffer.clear()
                    speech_seen = False
                    followup_entered_at = time.time()
                    bytes_received_count = 0
                    logger.info(
                        f"[{client_id}] Entered FOLLOWUP/LISTENING mode. Waiting for audio..."
                    )

                previous_mode = current_mode

                raw_bytes = message.get("bytes") or message.get("data")
                raw_text = message.get("text")

                if raw_text:
                    try:
                        msg = json.loads(raw_text)

                        # ── Camera presence detection ────────────────────────────
                        # Frontend sends a JPEG frame every 1.5 s while AIRA is passive.
                        # MediaPipe checks if a face is looking at the camera; 2 consecutive
                        # positives trigger the greeting (same activation path as wake word).
                        if msg.get("type") == "presence_frame":
                            # Only process in PASSIVE mode
                            if session_state["mode"] != "PASSIVE":
                                continue

                            # 30-second cooldown after the last activation.
                            # This prevents AIRA from re-greeting immediately after
                            # a session ends while the visitor is still near the kiosk.
                            PRESENCE_COOLDOWN = float(
                                os.getenv("PRESENCE_COOLDOWN_SECONDS", "30.0")
                            )
                            if (
                                time.time() - session_state["last_presence_trigger"]
                                < PRESENCE_COOLDOWN
                            ):
                                continue

                            image_b64 = msg.get("image_b64", "")
                            if not image_b64:
                                continue

                            # Strip data-URL prefix: "data:image/jpeg;base64,XXX" → "XXX"
                            if "," in image_b64:
                                image_b64 = image_b64.split(",", 1)[1]

                            try:
                                image_bytes = base64.b64decode(image_b64)
                            except Exception:
                                continue

                            # Run MediaPipe in thread pool (CPU-intensive, blocking)
                            detection_svc = get_person_detection_service()
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                _face_executor,
                                lambda: detection_svc.detect_person(image_bytes),
                            )

                            if result["detected"]:
                                session_state["presence_count"] += 1
                                logger.info(
                                    f"[{client_id}] Face at kiosk "
                                    f"(count={session_state['presence_count']}, "
                                    f"conf={result['confidence']:.2f}, "
                                    f"ratio={result['face_ratio']:.3f})"
                                )
                                if session_state["presence_count"] >= 2:
                                    logger.info(
                                        f"[{client_id}] ✅ Person confirmed! Activating AIRA."
                                    )
                                    session_state["presence_count"] = 0
                                    session_state["last_presence_trigger"] = time.time()
                                    session_state["mode"] = "PROCESSING"
                                    await websocket.send_text(
                                        json.dumps({"type": "person_detected"})
                                    )
                                    # Reuse the exact same wake-word activation path
                                    await text_queue.put("WAKE_WORD_TRIGGERED")
                                    oww_carry.clear()
                                    audio_buffer.clear()
                                    speech_seen = False
                            else:
                                # No face (or too small) — reset consecutive counter
                                session_state["presence_count"] = 0

                            continue

                        # ── Face verification request from frontend ──────────────
                        # Triggered when the employee says their name and LLM identifies them.
                        # Frontend sends: { type: "verify_face", audio_name: "John", image_b64: "..." }
                        if msg.get("type") == "verify_face":
                            audio_name = msg.get("audio_name", "")
                            image_b64 = msg.get("image_b64", "")
                            session_state["awaiting_face"] = False
                            logger.info(
                                f"[{client_id}] Face verification requested for: '{audio_name}'"
                            )

                            # Run DeepFace in a thread pool (it's blocking/CPU-intensive)
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                _face_executor,
                                verify_employee_face,
                                audio_name,
                                image_b64,
                            )

                            logger.info(
                                f"[{client_id}] Face verify result for '{audio_name}': "
                                f"verified={result['verified']}, distance={result['distance']}"
                            )

                            # Send result back to frontend (for the UI badge)
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "face_verification_result",
                                        "verified": result["verified"],
                                        "distance": result["distance"],
                                        "audio_name": audio_name,
                                        "has_photo": result["has_photo"],
                                        "message": result.get("message", ""),
                                    }
                                )
                            )

                            # Speak verification result when a comparison was made.
                            # - mismatch => verbal challenge
                            # - match => verbal confirmation ("you can proceed")
                            if result["has_photo"] and result["message"]:
                                if result["verified"]:
                                    logger.info(
                                        f"[{client_id}] Face match — queueing verification confirmation for '{audio_name}'"
                                    )
                                else:
                                    logger.info(
                                        f"[{client_id}] Face mismatch — queueing verbal challenge for '{audio_name}'"
                                    )
                                await text_queue.put(result["message"])

                            continue

                        # ── Stop-speaking control message ────────────────────────
                        if msg.get("action") == "stop_speaking":
                            session_state["mode"] = "PASSIVE"
                            session_state["awaiting_face"] = False
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
                        and (
                            time.time() - followup_entered_at > FOLLOWUP_TIMEOUT_SECONDS
                        )
                    ):
                        logger.info(
                            f"[{client_id}] Followup timeout reached (no speech detected). Returning to PASSIVE."
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

                                if text and text not in ("NOISE_DETECTED", "NO_SPEECH"):
                                    # Only attempt employee face-verification when the speaker
                                    # is introducing THEMSELVES as an employee.
                                    # If they are naming who they want to VISIT (e.g. "I'm here
                                    # to see the HR manager"), skip verification entirely and
                                    # let the query router handle it normally.
                                    employee_name = None
                                    if _is_self_introduction(text):
                                        candidates = _candidate_names_from_transcript(
                                            text
                                        )
                                        loop = asyncio.get_event_loop()
                                        for candidate in candidates:
                                            # Skip pure role/title strings — not real names
                                            if _is_role_only_name(candidate):
                                                logger.debug(
                                                    f"[{client_id}] Skipping role-only candidate: '{candidate}'"
                                                )
                                                continue
                                            resolved = await loop.run_in_executor(
                                                _face_executor,
                                                _resolve_employee_name,
                                                candidate,
                                            )
                                            if resolved:
                                                employee_name = resolved
                                                logger.info(
                                                    f"[{client_id}] Employee identified from candidate '{candidate}' as '{employee_name}'"
                                                )
                                                break
                                    else:
                                        logger.debug(
                                            f"[{client_id}] Skipping employee lookup — speaker is naming a host, not themselves: '{text}'"
                                        )

                                    if employee_name:
                                        logger.info(
                                            f"[{client_id}] employee_identified emitted: '{employee_name}'"
                                        )
                                        session_state["awaiting_face"] = True
                                        await websocket.send_text(
                                            json.dumps(
                                                {
                                                    "type": "employee_identified",
                                                    "name": employee_name,
                                                }
                                            )
                                        )
                                    else:
                                        await text_queue.put(text)
                                else:
                                    session_state["mode"] = "PASSIVE"
                                    session_state["awaiting_face"] = False
                                    await websocket.send_text(
                                        json.dumps({"state": "passive"})
                                    )
                                audio_buffer.clear()
                                speech_seen = False

        async def brain():
            while True:
                text = await text_queue.get()
                manager.client_state[client_id] = "THINKING"
                reply_text = await process_text_for_client(client_id, text)

                logger.info(f"[{client_id}] AI Response generated: '{reply_text}'")

                if not reply_text or not reply_text.strip():
                    session_state["mode"] = "PASSIVE"
                    await websocket.send_text(json.dumps({"state": "passive"}))
                    text_queue.task_done()
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

                # ALWAYS give the user 12 seconds to reply after Jarvis speaks
                session_state["mode"] = "FOLLOWUP"
                await websocket.send_text(json.dumps({"state": "listening"}))

                text_queue.task_done()

        tasks = [
            asyncio.create_task(listener()),
            asyncio.create_task(brain()),
            asyncio.create_task(send_keepalive()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
