import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from models.whisper_processor import WhisperProcessor
from models.groq_processor import GroqProcessor
from models.tts_processor import KokoroTTSProcessor
import sqlite3
from pathlib import Path

from receptionist.database import engine, _db_path
from receptionist.models import Base
from receptionist.seed_data import seed_database
from services.face_recognition_service import cleanup_old_captures

logger = logging.getLogger(__name__)


def _migrate_visitors_columns() -> None:
    """Idempotent: adds first_seen / last_seen to visitors if missing.
    Safe to run on every startup — skips columns that already exist."""
    from datetime import datetime

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    needed = [("first_seen", "DATETIME", now), ("last_seen", "DATETIME", now)]
    try:
        conn = sqlite3.connect(str(_db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(visitors)")
        existing = {row[1] for row in cur.fetchall()}
        for col, col_type, default in needed:
            if col in existing:
                continue
            cur.execute(f"ALTER TABLE visitors ADD COLUMN {col} {col_type}")
            cur.execute(
                f"UPDATE visitors SET {col} = ? WHERE {col} IS NULL", (default,)
            )
            conn.commit()
            logger.info("Migration: added column '%s' to visitors table.", col)
        conn.close()
    except Exception as exc:
        logger.error("Column migration failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing models on startup...")
    try:
        # Initialize processors to load models
        whisper_processor = WhisperProcessor.get_instance()
        llm_processor = GroqProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()

        # Initialize receptionist database (SQLite)
        loop = asyncio.get_running_loop()

        def _init_db():
            Base.metadata.create_all(bind=engine)
            _migrate_visitors_columns()  # adds first_seen/last_seen if missing
            seed_database()

        await loop.run_in_executor(None, _init_db)
        deleted_captures = await loop.run_in_executor(None, cleanup_old_captures)
        logger.info(
            "Diagnostic capture cleanup completed. Deleted=%s", deleted_captures
        )
        # Cleanup old visitors (and their photos) from disk and DB
        from receptionist.cleanup_old_visitors import purge_old_visitors

        deleted_visitors = await loop.run_in_executor(None, purge_old_visitors)
        logger.info("Visitor retention cleanup completed. Deleted=%s", deleted_visitors)
        # ────────────────────────────────────────────────────────────
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    # Close any remaining connections
    from managers.connection_manager import manager

    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")
