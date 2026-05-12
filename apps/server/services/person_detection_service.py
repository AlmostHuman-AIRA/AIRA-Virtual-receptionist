"""
person_detection_service.py
──────────────────────────────────────────────────────────────────────────────
MediaPipe Face Detection (Tasks API) for camera-based presence detection.

MediaPipe 0.10+ removed the old `mp.solutions` API.
This implementation uses the new `mediapipe.tasks.vision.FaceDetector` API.

The short-range BlazeFace .tflite model is downloaded automatically on first
startup and cached at: apps/server/face_detector.tflite  (~1 MB, one-time).

Design:
  - Singleton pattern — model loaded once, reused across all WebSocket sessions.
  - Short-range BlazeFace model optimised for faces within ~2 m (kiosk range).
  - MIN_FACE_RATIO filter ignores tiny faces of people far from the screen.
  - warmup_mediapipe() pre-loads the model at startup so first detection is fast.
"""

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Configurable via .env ──────────────────────────────────────────────────────
PRESENCE_DETECT_CONFIDENCE: float = float(
    os.getenv("PRESENCE_DETECT_CONFIDENCE", "0.5")
)
PRESENCE_MIN_FACE_RATIO: float = float(os.getenv("PRESENCE_MIN_FACE_RATIO", "0.02"))

# Short-range BlazeFace model (Google-hosted, ~1 MB)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
# Cache the model next to this service file so it survives virtual-env rebuilds
_MODEL_PATH = Path(__file__).parent / "face_detector.tflite"
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_model_downloaded() -> str:
    """Download the BlazeFace tflite model if not already cached. Returns path."""
    if not _MODEL_PATH.exists():
        logger.info(
            "Downloading BlazeFace short-range model (~1 MB) to %s …", _MODEL_PATH
        )
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            logger.info("BlazeFace model downloaded successfully.")
        except Exception as exc:
            logger.error("Failed to download BlazeFace model: %s", exc)
            raise
    return str(_MODEL_PATH)


class PersonDetectionService:
    """
    Singleton MediaPipe Face Detection (Tasks API) wrapper for presence detection.

    Uses mediapipe.tasks.vision.FaceDetector (MediaPipe 0.10+).
    """

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model_downloaded()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=PRESENCE_DETECT_CONFIDENCE,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        self._mp = mp  # keep reference to avoid repeated imports
        logger.info(
            "PersonDetectionService initialised "
            "(MediaPipe Tasks FaceDetector, conf=%.2f, min_ratio=%.3f)",
            PRESENCE_DETECT_CONFIDENCE,
            PRESENCE_MIN_FACE_RATIO,
        )

    def detect_person(self, image_bytes: bytes) -> dict:
        """
        Check whether a face looking at the camera is present in the frame.

        Args:
            image_bytes: Raw JPEG/PNG bytes (already base64-decoded).

        Returns:
            {
                "detected":   bool,   # True only if face found AND large enough
                "confidence": float,  # Best detection confidence (0.0–1.0)
                "face_ratio": float,  # Face bounding-box area / frame area
            }
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning(
                    "PersonDetectionService: could not decode image bytes "
                    "(not a valid JPEG/PNG?)"
                )
                return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # Wrap numpy array in a MediaPipe Image (Tasks API format)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=img_rgb
            )
            results = self._detector.detect(mp_image)

            if not results.detections:
                return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}

            best_conf: float = 0.0
            best_ratio: float = 0.0

            for detection in results.detections:
                # Tasks API: confidence is in detection.categories[0].score
                score = (
                    float(detection.categories[0].score)
                    if detection.categories
                    else 0.0
                )
                # Tasks API: bounding_box is in pixel coords
                bbox = detection.bounding_box
                face_ratio = (bbox.width / w) * (bbox.height / h)

                if score > best_conf:
                    best_conf = score
                    best_ratio = face_ratio

            # Ignore tiny faces — person is too far from the kiosk
            if best_ratio < PRESENCE_MIN_FACE_RATIO:
                logger.debug(
                    "Face detected but too small for presence trigger "
                    "(ratio=%.4f < threshold=%.4f)",
                    best_ratio,
                    PRESENCE_MIN_FACE_RATIO,
                )
                return {
                    "detected": False,
                    "confidence": best_conf,
                    "face_ratio": best_ratio,
                }

            return {
                "detected": True,
                "confidence": best_conf,
                "face_ratio": best_ratio,
            }

        except Exception as exc:
            logger.error("PersonDetectionService.detect_person error: %s", exc)
            return {"detected": False, "confidence": 0.0, "face_ratio": 0.0}


# ── Singleton accessor ─────────────────────────────────────────────────────────
_service_instance: PersonDetectionService | None = None


def get_person_detection_service() -> PersonDetectionService:
    """Lazy singleton — model is loaded exactly once per process."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PersonDetectionService()
    return _service_instance


def warmup_mediapipe() -> None:
    """
    Pre-load the MediaPipe Tasks model at server startup.
    Call this from a ThreadPoolExecutor (it blocks for ~300–600 ms on first run
    including the model download if needed).
    Pattern mirrors warmup_deepface() in face_recognition_service.py.
    """
    logger.info("Warming up MediaPipe face detection model...")
    svc = get_person_detection_service()
    # Run inference on a blank frame to force all JIT compilation / caching
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    _, dummy_bytes = cv2.imencode(".jpg", dummy)
    svc.detect_person(dummy_bytes.tobytes())
    logger.info("MediaPipe face detection warmup complete.")
