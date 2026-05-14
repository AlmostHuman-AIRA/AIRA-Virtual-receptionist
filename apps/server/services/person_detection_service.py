"""
person_detection_service.py
----------------------------
Lightweight person-presence detection using MediaPipe Face Detection (Tasks API).

Designed for kiosk / reception-desk scenarios:
  - Uses the short-range model (< 2 m) which is ideal for someone standing at a desk.
  - Filters by face-to-frame area ratio so distant passers-by don't trigger activation.
  - Singleton pattern: the model is loaded once and reused across all WebSocket sessions.
  - Thread-safe: MediaPipe's Python bindings are safe for sequential calls from a thread pool.

Environment variables (all optional):
  PRESENCE_DETECT_CONFIDENCE  – min detection confidence  (default 0.5)
  PRESENCE_MIN_FACE_RATIO     – min face_area / frame_area (default 0.02)
"""

import base64
import logging
import os
import io
from pathlib import Path
from typing import Dict
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# ── Configuration via env vars ────────────────────────────────────────────────
_CONFIDENCE = float(os.getenv("PRESENCE_DETECT_CONFIDENCE", "0.5"))
_MIN_FACE_RATIO = float(os.getenv("PRESENCE_MIN_FACE_RATIO", "0.02"))

# Path to the BlazeFace short-range TFLite model
_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "blaze_face_short_range.tflite"
)


class PersonDetectionService:
    def __init__(self):
        """Initialize MediaPipe FaceDetector using the Tasks API."""
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"MediaPipe model not found at {_MODEL_PATH}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            )

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=_CONFIDENCE,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        logger.info("PersonDetectionService initialised successfully (Tasks API).")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_person(self, image_b64: str) -> Dict:
        """
        Detect whether a face is present and large enough in the frame.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded JPEG image (may or may not have the data URI prefix).

        Returns
        -------
        dict  {detected: bool, confidence: float, face_ratio: float}
        """
        try:
            # 1. Decode base64 → numpy RGB array
            rgb_image = self._decode_image(image_b64)
            if rgb_image is None:
                return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

            # 2. Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # 3. Run face detection
            result = self._detector.detect(mp_image)

            if not result.detections:
                return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

            # 4. Pick the highest-confidence detection
            best = max(result.detections, key=lambda d: d.categories[0].score)
            confidence = best.categories[0].score

            # 5. Calculate face-to-frame area ratio
            bbox = best.bounding_box
            frame_h, frame_w = rgb_image.shape[:2]
            if frame_w > 0 and frame_h > 0:
                face_ratio = (bbox.width * bbox.height) / (frame_w * frame_h)
            else:
                face_ratio = 0.0

            # 6. Check if the face is large enough (close to the camera)
            detected = face_ratio >= _MIN_FACE_RATIO

            return {
                "detected": detected,
                "confidence": float(confidence),
                "face_ratio": float(face_ratio),
            }

        except Exception as e:
            logger.error("PersonDetectionService.detect_person error: %s", e)
            return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _decode_image(image_b64: str):
        """Decode a base64 JPEG/PNG string into an RGB numpy array."""
        try:
            # Strip the data URI prefix if present
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(image_b64)

            # Use PIL for reliable decoding
            from PIL import Image

            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return np.array(pil_image)

        except Exception as e:
            logger.error("Image decode failed: %s", e)
            return None


# ── Module-level singleton accessor ───────────────────────────────────────────

_service_instance: PersonDetectionService | None = None


def get_person_detection_service() -> PersonDetectionService:
    """Return (or create) the singleton PersonDetectionService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PersonDetectionService()
    return _service_instance


def warmup_mediapipe() -> None:
    """
    Pre-load the MediaPipe model so the first real detection is fast.
    Called at server startup from websocket_routes.py (same pattern as warmup_deepface).
    """
    logger.info("Warming up MediaPipe Face Detection model (Tasks API)...")
    try:
        svc = get_person_detection_service()
        # Create a tiny dummy image and run one detection to fully load the model
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy_b64 = base64.b64encode(_encode_dummy_jpeg(dummy)).decode("utf-8")
        svc.detect_person(dummy_b64)
        logger.info("MediaPipe Face Detection model warmed up successfully.")
    except Exception as e:
        logger.error("MediaPipe warmup failed: %s", e)


def _encode_dummy_jpeg(rgb_array: np.ndarray) -> bytes:
    """Encode a numpy RGB array as JPEG bytes for warmup."""
    from PIL import Image

    img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
