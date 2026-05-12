import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load model (It will auto-download yolov8n.pt on first run)
model = YOLO("yolov8n.pt")


# In services/vision_service.py
def is_person_in_frame(image_bytes: bytes) -> bool:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            # THIS IS LIKELY YOUR PROBLEM
            logger.error(
                "Vision detection error: Failed to decode image bytes. Are they valid JPEG/PNG?"
            )
            return False

        img_small = cv2.resize(img, (320, 240))

        # Run inference
        results = model(img_small, verbose=False)

        # ADD THIS DEBUG PRINT
        for r in results:
            if len(r.boxes) > 0:
                logger.debug(f"Detected {len(r.boxes)} objects.")

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                    return True
        return False
    except Exception as e:
        logger.error(f"Vision detection error: {e}")
        return False
