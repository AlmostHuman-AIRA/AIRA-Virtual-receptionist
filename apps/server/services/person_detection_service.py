"""
person_detection_service.py
----------------------------
Lightweight person-presence detection using MediaPipe Face Detection.

Used to replace the wake-word trigger: when a person stands in front of
the camera for a sustained period (~4.5 s / 3 consecutive detections),
AIRA activates automatically.

WHY MediaPipe over YOLOv8?
  1. Detects *faces facing the camera*, not just any person in the background.
  2. 3-5× faster on CPU (~10-30 ms vs ~50-100 ms per frame).
  3. Short-range model (model_selection=0) is optimised for < 2 m — perfect for
     a reception kiosk.
  4. Lighter dependency than `ultralytics`.
"""

import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Tunables (via .env) ──────────────────────────────────────────────────────
PRESENCE_DETECT_CONFIDENCE = float(os.getenv("PRESENCE_DETECT_CONFIDENCE", "0.5"))
# Minimum face-area / frame-area ratio.  A value of 0.02 ≈ a face that
# occupies ~14 % of the frame width.  Filters out small background faces.
PRESENCE_MIN_FACE_RATIO = float(os.getenv("PRESENCE_MIN_FACE_RATIO", "0.02"))


class PersonDetectionService:
    """Singleton wrapper around MediaPipe Face Detection."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (< 2 m), ideal for kiosk
            min_detection_confidence=PRESENCE_DETECT_CONFIDENCE,
        )
        logger.info(
            "PersonDetectionService initialised  "
            "(confidence=%.2f, min_face_ratio=%.3f)",
            PRESENCE_DETECT_CONFIDENCE,
            PRESENCE_MIN_FACE_RATIO,
        )

    # ── public API ────────────────────────────────────────────────────────────
    def detect_person(self, image_b64: str) -> dict:
        """
        Decode a base64 JPEG and run MediaPipe face detection.

        Returns
        -------
        {
            "detected":   bool   – True when a sufficiently-large face is found
            "confidence": float  – highest detection confidence (0.0 if none)
            "face_ratio": float  – largest face area / frame area (0.0 if none)
        }
        """
        try:
            # 1. Decode base64 → numpy BGR → RGB
            raw_b64 = image_b64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(raw_b64)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if bgr is None:
                logger.warning("Failed to decode image from base64.")
                return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # 2. Run MediaPipe
            results = self._detector.process(rgb)

            if not results.detections:
                return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

            # 3. Find the largest / most-confident detection
            frame_h, frame_w = rgb.shape[:2]
            frame_area = frame_h * frame_w

            best_confidence = 0.0
            best_face_ratio = 0.0

            for detection in results.detections:
                score = detection.score[0]
                bbox = detection.location_data.relative_bounding_box
                face_area = (bbox.width * frame_w) * (bbox.height * frame_h)
                face_ratio = face_area / frame_area if frame_area > 0 else 0.0

                if score > best_confidence:
                    best_confidence = score
                if face_ratio > best_face_ratio:
                    best_face_ratio = face_ratio

            detected = (
                best_confidence >= PRESENCE_DETECT_CONFIDENCE
                and best_face_ratio >= PRESENCE_MIN_FACE_RATIO
            )

            return {
                "detected": detected,
                "confidence": round(best_confidence, 3),
                "face_ratio": round(best_face_ratio, 4),
            }

        except Exception as e:
            logger.error("Person detection failed: %s", e, exc_info=True)
            return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._detector.close()
        except Exception:
            pass


# ── Singleton ────────────────────────────────────────────────────────────────
_service_instance: Optional[PersonDetectionService] = None


def get_person_detection_service() -> PersonDetectionService:
    """Lazy singleton — model loaded once and kept in memory."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PersonDetectionService()
    return _service_instance


def warmup_mediapipe() -> None:
    """
    Run a dummy detection at startup so the first real frame doesn't
    incur a cold-start penalty (~200-500 ms model load).
    """
    logger.info("Starting MediaPipe Face Detection warmup…")
    try:
        service = get_person_detection_service()
        # Create a small dummy image (100×100 black)
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        dummy_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        result = service.detect_person(dummy_b64)
        logger.info(
            "✅ MediaPipe warmup complete (dummy result: detected=%s).",
            result["detected"],
        )
    except Exception as e:
        logger.warning("MediaPipe warmup failed: %s", e)
