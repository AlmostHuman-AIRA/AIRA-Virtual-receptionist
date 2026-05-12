"""
slack_reply_store.py
--------------------
Thread-safe in-memory store for Slack replies.

KEY DESIGN — two parallel stores:
  1. _store_by_host   : keyed by employee DB name (lowercased)
  2. _store_pending   : keyed by slack channel_id (fallback)

slack_watcher checks _store_by_host first, then _store_pending.
This means even if Sannidhi's Slack display name ("SANNIDHIVK") doesn't
match "Suresh" in the DB, her reply still gets picked up via channel fallback.
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_store_by_host: dict[str, str] = {}  # db_employee_name_lower -> reply_text
_store_pending: dict[str, str] = {}  # slack_channel_id       -> reply_text
_store_lock = threading.Lock()


def save_reply(sender_name: str, reply_text: str, channel_id: str = "") -> None:
    """Save under sender name AND channel_id (fallback)."""
    key = sender_name.strip().lower()
    text = reply_text.strip()
    with _store_lock:
        _store_by_host[key] = text
        if channel_id:
            _store_pending[channel_id] = text
    logger.info(
        "SLACK_STORE | saved | sender_key='%s' channel='%s' text='%s'",
        key,
        channel_id,
        text,
    )


def save_reply_for_host(host_name: str, reply_text: str) -> None:
    """Explicitly save keyed by DB employee name."""
    key = host_name.strip().lower()
    with _store_lock:
        _store_by_host[key] = reply_text.strip()
    logger.info("SLACK_STORE | saved_by_host | host_key='%s'", key)


def pop_reply(employee_name: str) -> Optional[str]:
    """Check _store_by_host. Returns None on miss — use pop_any_channel_reply as fallback."""
    key = employee_name.strip().lower()
    with _store_lock:
        result = _store_by_host.pop(key, None)
    if result:
        logger.info("SLACK_STORE | pop_by_host HIT | key='%s'", key)
    else:
        logger.debug(
            "SLACK_STORE | pop_by_host MISS | key='%s' | store_keys=%s",
            key,
            list(_store_by_host.keys()),
        )
    return result


def pop_any_channel_reply(channel_id: str) -> Optional[str]:
    """
    Fallback: return ANY reply in this channel regardless of sender name.
    Used when Slack display name != DB employee name.
    """
    with _store_lock:
        result = _store_pending.pop(channel_id, None)
    if result:
        logger.info("SLACK_STORE | pop_channel HIT | channel='%s'", channel_id)
    else:
        logger.debug(
            "SLACK_STORE | pop_channel MISS | channel='%s' | channels=%s",
            channel_id,
            list(_store_pending.keys()),
        )
    return result


def peek_reply(employee_name: str) -> Optional[str]:
    key = employee_name.strip().lower()
    with _store_lock:
        return _store_by_host.get(key)


def dump_store() -> dict:
    """Debug snapshot of both stores."""
    with _store_lock:
        return {
            "by_host": dict(_store_by_host),
            "by_channel": dict(_store_pending),
        }
