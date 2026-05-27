import logging
import threading
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import asyncio

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Switch from Incoming Webhook → Web API ────────────────────────────────────
# SLACK_WEBHOOK_URL is no longer used.
# You need two new env vars:
#   SLACK_BOT_TOKEN   — your bot's OAuth token (xoxb-...)
#   SLACK_CHANNEL_ID  — the channel/DM ID to post into
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")  # e.g. C08xxxxxxx or D08xxxxxxx

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="slack_notifier")
_last_notified: dict = {}
_notify_lock = threading.Lock()


def _post_message(text: str) -> dict | None:
    """
    Synchronous helper — calls chat.postMessage and returns the full API response.
    Returns None on failure.
    """
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        logger.error("SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set in environment.")
        return None
    try:
        response = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": SLACK_CHANNEL_ID, "text": text},
            timeout=10,
        )
        data = response.json()
        if not data.get("ok"):
            logger.error("Slack chat.postMessage failed: %s", data.get("error"))
            return None
        return data
    except Exception as e:
        logger.error("Slack postMessage exception: %s", e)
        return None


def _get_employee_slack_info(employee_name: str) -> tuple[str | None, str | None]:
    """
    Look up slack_user_id and slack_dm_channel from the employees DB by name.
    Returns (slack_user_id, slack_dm_channel) or (None, None).
    """
    import sqlite3
    from pathlib import Path

    DB_PATH = Path(
        r"C:/Users/Administrator/Desktop/CPU-compatible-AI-/apps/server/receptionist/office.db"
    )
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT slack_user_id, slack_dm_channel FROM employees WHERE name = ?",
            (employee_name,),
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0], row[1]
    except Exception as e:
        logger.error("DB lookup failed for %s: %s", employee_name, e)
    return None, None


def _register_and_log(session_id: str, data: dict, label: str) -> None:
    """Helper: register a thread for reply polling and log success."""
    from services.slack_reply_poller import register_thread

    register_thread(
        session_id=session_id,
        channel_id=data["channel"],
        thread_ts=data["ts"],
    )
    logger.info(
        "✅ %s | channel=%s thread_ts=%s",
        label,
        data["channel"],
        data["ts"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARRIVAL NOTIFICATION
# Fallback chain: DM (App inbox) → @mention in channel → plain channel post
# ─────────────────────────────────────────────────────────────────────────────


def _send_arrival_thread(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
    host_slack_user_id: str | None = None,
):
    logger.info(
        "Slack arrival thread started for %s -> %s", visitor_name, employee_name
    )

    # ── Resolve user ID from DB if not passed in ──────────────────────────
    if not host_slack_user_id:
        host_slack_user_id, _ = _get_employee_slack_info(employee_name)
        if host_slack_user_id:
            logger.info(
                "Resolved slack_user_id for %s: %s", employee_name, host_slack_user_id
            )
        else:
            logger.warning(
                "No slack_user_id for %s — will go straight to channel", employee_name
            )

    message = (
        f"🛎️ *Visitor Arrival for {employee_name}*\n"
        f"• *Visitor Name:* {visitor_name}\n"
        f"• *Category:* {visitor_type}\n"
        f"• *Purpose:* {purpose}\n\n"
        f"_Please head to the front desk._"
    )

    data = None

    if host_slack_user_id:
        # ── Primary: DM (lands in App inbox, private to host) ─────────────
        data = _post_dm(host_slack_user_id, message)
        if data:
            _register_and_log(session_id, data, f"Arrival DM → {employee_name}")
            return

        logger.warning("DM failed for %s — trying @mention fallback", employee_name)

        # ── Fallback 1: @mention in #reception_desk ───────────────────────
        data = _post_channel_mention(host_slack_user_id, message)
        if data:
            _register_and_log(session_id, data, f"Arrival @mention → {employee_name}")
            return

        logger.warning(
            "@mention failed for %s — trying plain channel post", employee_name
        )

    # ── Fallback 2: plain channel post (no user ID, or all else failed) ───
    channel_message = message + f"\n_Host: {employee_name}_"
    data = _post_message(channel_message)
    if data:
        _register_and_log(session_id, data, f"Arrival channel post → {employee_name}")
    else:
        logger.error("❌ All notification methods failed for %s", employee_name)


def send_slack_arrival(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
    host_slack_user_id: str | None = None,
):
    """Submit arrival notification; deduplicate per session."""
    with _notify_lock:
        if _last_notified.get(session_id) == visitor_name:
            logger.warning(
                "Blocking duplicate notification to '%s' in session %s",
                employee_name,
                session_id,
            )
            return
        _last_notified[session_id] = visitor_name

    logger.info("Queuing Slack arrival notification for %s...", visitor_name)
    _executor.submit(
        _send_arrival_thread,
        employee_name,
        visitor_name,
        visitor_type,
        purpose,
        session_id,
        host_slack_user_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MEETING NOTIFICATION
# Fallback chain: DM (App inbox) → @mention in channel → plain channel post
# ─────────────────────────────────────────────────────────────────────────────


def send_slack_meeting_scheduled(
    host_name: str,
    host_email: str,
    visitor_name: str,
    date_str: str,
    time_str: str,
    purpose: str,
    session_id: str,
    host_slack_user_id: str | None = None,
):
    if not host_slack_user_id and host_email:
        host_slack_user_id = get_slack_user_id_by_email(host_email)
        if not host_slack_user_id:
            logger.warning(
                "Could not resolve Slack user for %s — will fall back to channel",
                host_email,
            )

    # AFTER — include host_name in the key
    with _notify_lock:
        key = f"meeting_{session_id}_{host_name}"  # ← per-host key
        new_value = f"{visitor_name}_{date_str}_{time_str}"
        is_reschedule = key in _last_notified and _last_notified.get(key) != new_value

        if _last_notified.get(key) == new_value:
            logger.warning(
                "Blocking duplicate meeting notification for %s (%s)",
                visitor_name,
                host_name,
            )
            return
        _last_notified[key] = new_value

    def _send():
        # ── Build message ─────────────────────────────────────────────────
        if is_reschedule:
            message = (
                f"🔄 *Meeting Rescheduled*\n"
                f"• *Visitor:* {visitor_name}\n"
                f"• *New Date:* {date_str}  |  *New Time:* {time_str}\n"
                f"• *Purpose:* {purpose}\n\n"
                f"_Visitor has requested a new time. Please reply to confirm._"
            )
            log_label = "Reschedule"
        else:
            message = (
                f"📅 *New Meeting Request via AIRA*\n"
                f"• *Visitor:* {visitor_name}\n"
                f"• *Date:* {date_str}  |  *Time:* {time_str}\n"
                f"• *Purpose:* {purpose}\n\n"
                f"_Please reply to confirm or suggest a different time._"
            )
            log_label = "New meeting"

        if host_slack_user_id:
            # ── Primary: DM ───────────────────────────────────────────────
            data = _post_dm(host_slack_user_id, message)
            if data:
                _register_and_log(session_id, data, f"{log_label} DM → {host_name}")
                return

            logger.warning(
                "DM failed for %s (user_id=%s) — trying @mention fallback",
                host_name,
                host_slack_user_id,
            )

            # ── Fallback 1: @mention in #reception_desk ───────────────────
            data = _post_channel_mention(host_slack_user_id, message)
            if data:
                _register_and_log(
                    session_id, data, f"{log_label} @mention → {host_name}"
                )
                return

            logger.warning(
                "@mention failed for %s — trying plain channel post", host_name
            )
        else:
            logger.warning(
                "host_slack_user_id missing for %s (%s) — skipping DM and @mention",
                host_name,
                host_email,
            )

        # ── Fallback 2: plain channel post ────────────────────────────────
        channel_message = message + f"\n_Host: {host_name} ({host_email})_"
        data = _post_message(channel_message)
        if data:
            _register_and_log(
                session_id, data, f"{log_label} channel post → {host_name}"
            )
        else:
            logger.error("❌ All notification methods failed for %s", host_name)

    _executor.submit(_send)


# ─────────────────────────────────────────────────────────────────────────────
# SLACK API HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def get_slack_user_id_by_email(email: str) -> str | None:
    """Look up a Slack user ID by their email address."""
    try:
        response = httpx.get(
            "https://slack.com/api/users.lookupByEmail",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            params={"email": email},
            timeout=10,
        )
        data = response.json()
        if data.get("ok"):
            return data["user"]["id"]
        logger.error("users.lookupByEmail failed: %s", data.get("error"))
        return None
    except Exception as e:
        logger.error("users.lookupByEmail exception: %s", e)
        return None


def _get_dm_channel(user_id: str) -> str | None:
    """Open a DM channel with a user and return the channel ID."""
    try:
        response = httpx.post(
            "https://slack.com/api/conversations.open",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"users": user_id},
            timeout=10,
        )
        data = response.json()
        if data.get("ok"):
            return data["channel"]["id"]
        logger.error("conversations.open failed: %s", data.get("error"))
        return None
    except Exception as e:
        logger.error("conversations.open exception: %s", e)
        return None


def _post_dm(user_id: str, text: str) -> dict | None:
    """Send a DM to a specific user (lands in App inbox)."""
    dm_channel = _get_dm_channel(user_id)
    if not dm_channel:
        return None
    try:
        response = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": dm_channel, "text": text},
            timeout=10,
        )
        data = response.json()
        if not data.get("ok"):
            logger.error("DM postMessage failed: %s", data.get("error"))
            return None
        return data
    except Exception as e:
        logger.error("DM postMessage exception: %s", e)
        return None


def _post_channel_mention(user_id: str, text: str) -> dict | None:
    """
    Fallback: post to the shared reception channel with an @mention.
    Used when _post_dm fails.
    """
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        logger.error("SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set.")
        return None
    try:
        response = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={
                "channel": SLACK_CHANNEL_ID,
                "text": f"<@{user_id}>\n{text}",
            },
            timeout=10,
        )
        data = response.json()
        if not data.get("ok"):
            logger.error("channel mention postMessage failed: %s", data.get("error"))
            return None
        return data
    except Exception as e:
        logger.error("channel mention postMessage exception: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CLEANUP
# ─────────────────────────────────────────────────────────────────────────────


def clear_session(session_id: str):
    """Clean up notification history for a session."""
    from services.slack_reply_poller import unregister_thread

    unregister_thread(session_id)
    with _notify_lock:
        # AFTER
        keys_to_remove = [
            k
            for k in _last_notified
            if k == session_id
            or k.startswith(f"meeting_{session_id}")  # ← startswith catches all hosts
        ]
        for k in keys_to_remove:
            _last_notified.pop(k, None)
    logger.info("Cleared Slack cache for session %s", session_id)
