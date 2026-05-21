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
#                       (right-click the channel in Slack → Copy link → last segment is the ID)
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


# ── notify_slack.py (fixed) ──────────────────────────────────────────────────


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


def _send_arrival_thread(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
    host_slack_user_id: str | None = None,
):
    logger.info(f"Slack thread started for {visitor_name} -> {employee_name}")

    # ── FIX: look up from DB if not passed in ────────────────────────────────
    if not host_slack_user_id:
        host_slack_user_id, cached_dm_channel = _get_employee_slack_info(employee_name)
        if host_slack_user_id:
            logger.info(
                f"Resolved slack_user_id for {employee_name} from DB: {host_slack_user_id}"
            )
        else:
            logger.warning(
                f"No slack_user_id in DB for {employee_name} — will fall back to channel"
            )

    message = (
        f"🛎️ *Visitor Arrival for {employee_name}*\n"
        f"• *Visitor Name:* {visitor_name}\n"
        f"• *Category:* {visitor_type}\n"
        f"• *Purpose:* {purpose}\n\n"
        f"_Please head to the front desk._"
    )

    data = None

    # ── Try DM first ─────────────────────────────────────────────────────────
    if host_slack_user_id:
        data = _post_dm(host_slack_user_id, message)
        if data:
            logger.info(f"✅ Arrival DM sent to {employee_name}")
        else:
            logger.warning(f"DM failed for {employee_name} — falling back to channel")

    # ── ALSO post to channel (so reception desk sees it too) ─────────────────
    channel_message = message + f"\n_Host: {employee_name}_"
    channel_data = _post_message(channel_message)

    # ── Register thread — prefer DM thread, fall back to channel thread ──────
    final_data = data or channel_data
    if final_data:
        from services.slack_reply_poller import register_thread

        register_thread(
            session_id=session_id,
            channel_id=final_data["channel"],
            thread_ts=final_data["ts"],
        )
        logger.info(
            f"✅ Registered thread | channel={final_data['channel']} ts={final_data['ts']}"
        )
    else:
        logger.error(f"❌ Both DM and channel post failed for {employee_name}")


def send_slack_arrival(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
    host_slack_user_id: str | None = None,  # ← add this
):
    """Submit arrival notification; deduplicate per session."""
    with _notify_lock:
        if _last_notified.get(session_id) == visitor_name:
            logger.warning(
                f"Blocking duplicate notification to '{employee_name}' in session {session_id}"
            )
            return
        _last_notified[session_id] = visitor_name

    logger.info(f"Queuing Slack notification for {visitor_name}...")
    _executor.submit(
        _send_arrival_thread,
        employee_name,
        visitor_name,
        visitor_type,
        purpose,
        session_id,
        host_slack_user_id,  # ← pass through
    )


def send_slack_meeting_scheduled(
    host_name,
    host_email,
    visitor_name,
    date_str,
    time_str,
    purpose,
    session_id,
    host_slack_user_id: str | None = None,  # <-- add this
):
    if not host_slack_user_id and host_email:
        host_slack_user_id = get_slack_user_id_by_email(host_email)
        if not host_slack_user_id:
            logger.warning(
                "Could not resolve Slack user for %s — reschedule will fall back to channel",
                host_email,
            )

    with _notify_lock:
        key = f"meeting_{session_id}"
        new_value = f"{visitor_name}_{date_str}_{time_str}"
        is_reschedule = key in _last_notified and _last_notified.get(key) != new_value

        if _last_notified.get(key) == new_value:
            logger.warning(
                f"Blocking duplicate meeting notification for {visitor_name}"
            )
            return
        _last_notified[key] = new_value

    def _send():
        # ── Build message text ────────────────────────────────────────────────
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

        # ── Always prefer DM; fall back to channel only if user ID is missing ─
        if host_slack_user_id:
            data = _post_dm(host_slack_user_id, message)
            if data:
                from services.slack_reply_poller import register_thread

                register_thread(
                    session_id=session_id,
                    channel_id=data["channel"],
                    thread_ts=data["ts"],
                )
                logger.info(
                    "✅ %s DM sent to %s | thread_ts=%s",
                    log_label,
                    host_name,
                    data["ts"],
                )
                return
            # DM failed — log and fall through to channel
            logger.warning(
                "_post_dm failed for %s (user_id=%s) — falling back to channel",
                host_name,
                host_slack_user_id,
            )
        else:
            logger.warning(
                "host_slack_user_id missing for %s (%s) — falling back to channel",
                host_name,
                host_email,
            )

        # ── Channel fallback (adds host info so they can be identified) ───────
        channel_message = message + f"\n_Host: {host_name} ({host_email})_"
        data = _post_message(channel_message)
        if data:
            from services.slack_reply_poller import register_thread

            register_thread(
                session_id=session_id,
                channel_id=data["channel"],
                thread_ts=data["ts"],
            )
            logger.info(
                "✅ %s posted to channel for %s | thread_ts=%s",
                log_label,
                host_name,
                data["ts"],
            )

    _executor.submit(_send)


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
    """Send a DM to a specific user."""
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


def clear_session(session_id: str):
    """Clean up notification history for a session."""
    from services.slack_reply_poller import unregister_thread

    unregister_thread(session_id)
    with _notify_lock:
        keys_to_remove = [
            k for k in _last_notified if k == session_id or k == f"meeting_{session_id}"
        ]
        for k in keys_to_remove:
            _last_notified.pop(k, None)
    logger.info(f"Cleared Slack cache for session {session_id}")
