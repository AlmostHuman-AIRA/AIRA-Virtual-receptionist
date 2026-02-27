import logging
from fastapi import APIRouter

from managers.connection_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Get server statistics (voice-only)"""
    return manager.get_stats()
