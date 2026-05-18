# cleanup_old_visitors.py
"""
Deletes visitor photos from disk and removes DB rows (plus related logs
and meetings) for visitors who haven't been seen in RETENTION_DAYS days.

Run automatically at startup via main.py, or manually:
    python -m receptionist.cleanup_old_visitors
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from .database import SessionLocal
from .models import Meeting, ReceptionLog, Visitor

logger = logging.getLogger(__name__)

RETENTION_DAYS = 90  # change this to whatever your policy requires


def purge_old_visitors() -> int:
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    db = SessionLocal()
    deleted_count = 0

    try:
        old_visitors = db.query(Visitor).filter(Visitor.last_seen < cutoff).all()

        for visitor in old_visitors:
            # 1. Delete the photo file from disk if it exists
            if visitor.id_photo_path:
                photo = Path(visitor.id_photo_path)
                if photo.exists():
                    try:
                        photo.unlink()
                        logger.info("Deleted photo: %s", photo)
                    except OSError as e:
                        logger.warning("Could not delete photo %s: %s", photo, e)

            # 2. Delete related meeting rows first (foreign key constraint)
            db.query(Meeting).filter(Meeting.visitor_id == visitor.id).delete(
                synchronize_session=False
            )

            # 3. Delete related reception log rows
            db.query(ReceptionLog).filter(ReceptionLog.visitor_id == visitor.id).delete(
                synchronize_session=False
            )

            # 4. Delete the visitor row itself
            db.delete(visitor)
            deleted_count += 1

        db.commit()
        logger.info(
            "Cleanup complete: removed %d visitor(s) last seen before %s.",
            deleted_count,
            cutoff.strftime("%Y-%m-%d"),
        )
        return deleted_count

    except Exception as e:
        db.rollback()
        logger.error("Visitor cleanup failed: %s", e)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    purge_old_visitors()
