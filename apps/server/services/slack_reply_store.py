"""
slack_reply_store.py
--------------------
Thread-safe in-memory store for Slack replies coming in via the Events API.
Keyed by employee_name (lowercased) so query_router can poll for replies
without any direct coupling to the webhook layer.
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_store: dict[str, str] = {}  # employee_name_lower -> reply_text
_store_lock = threading.Lock()


def save_reply(employee_name: str, reply_text: str) -> None:
    """
    Called by the Slack webhook handler when an employee replies
    in the notification thread.

    Args:
        employee_name:  The name of the employee who replied (any case).
        reply_text:     The plain-text content of their Slack message.
    """
    key = employee_name.strip().lower()
    with _store_lock:
        _store[key] = reply_text.strip()
    logger.info("Stored Slack reply from '%s': %s", employee_name, reply_text)


def pop_reply(employee_name: str) -> Optional[str]:
    """
    Retrieve and DELETE the stored reply for an employee.
    Returns None if no reply has arrived yet.

    Args:
        employee_name:  The name to look up (any case).
    """
    key = employee_name.strip().lower()
    with _store_lock:
        return _store.pop(key, None)


def peek_reply(employee_name: str) -> Optional[str]:
    """
    Read the stored reply WITHOUT removing it.
    Useful if you want to check existence before acting.
    """
    key = employee_name.strip().lower()
    with _store_lock:
        return _store.get(key)
