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


def _send_arrival_thread(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
):
    logger.info(f"Slack thread started for {visitor_name} -> {employee_name}")

    message = (
        f"🛎️ *Visitor Arrival for {employee_name}*\n"
        f"• *Visitor Name:* {visitor_name}\n"
        f"• *Category:* {visitor_type}\n"
        f"• *Purpose:* {purpose}\n\n"
        f"_Please head to the front desk._"
    )

    data = _post_message(message)
    if data:
        # Register the thread so slack_watcher can poll for replies
        from services.slack_reply_poller import register_thread

        register_thread(
            session_id=session_id,
            channel_id=data["channel"],  # actual channel ID confirmed by Slack
            thread_ts=data["ts"],  # this message's ts = thread anchor
        )
        logger.info(
            f"✅ Slack notification sent for {employee_name} | "
            f"channel={data['channel']} thread_ts={data['ts']}"
        )


def send_slack_arrival(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
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
        if is_reschedule:
            if host_slack_user_id:
                # ✅ DM path
                message = (
                    f"🔄 *Meeting Rescheduled*\n"
                    f"• *Visitor:* {visitor_name}\n"
                    f"• *New Date:* {date_str}  |  *New Time:* {time_str}\n"
                    f"• *Purpose:* {purpose}\n\n"
                    f"_Visitor has agreed to the new time._"
                )
                data = _post_dm(host_slack_user_id, message)
                if data:
                    from services.slack_reply_poller import register_thread

                    register_thread(
                        session_id=session_id,
                        channel_id=data["channel"],
                        thread_ts=data["ts"],
                    )
                    logger.info(
                        f"✅ Reschedule DM sent to {host_name} | thread_ts={data['ts']}"
                    )
            else:
                # Fallback: DM failed, post to channel with reschedule label
                logger.warning(
                    f"host_slack_user_id missing — falling back to channel for reschedule"
                )
                message = (
                    f"🔄 *Meeting Rescheduled via AIRA*\n"
                    f"• *Host:* {host_name} ({host_email})\n"
                    f"• *Visitor:* {visitor_name}\n"
                    f"• *New Date:* {date_str}  |  *New Time:* {time_str}\n"
                    f"• *Purpose:* {purpose}\n\n"
                    f"_Visitor agreed to the updated time._"
                )
                data = _post_message(message)
                if data:
                    from services.slack_reply_poller import register_thread

                    register_thread(
                        session_id=session_id,
                        channel_id=data["channel"],
                        thread_ts=data["ts"],
                    )

        else:
            # ✅ First notification → post to channel as before
            message = (
                f"📅 *New Meeting Scheduled via AIRA*\n"
                f"• *Host:* {host_name} ({host_email})\n"
                f"• *Visitor:* {visitor_name}\n"
                f"• *Date:* {date_str}  |  *Time:* {time_str}\n"
                f"• *Purpose:* {purpose}\n\n"
                f"_A Google Calendar invite with email has been sent to the host._"
            )
            data = _post_message(message)
            if data:
                from services.slack_reply_poller import register_thread

                register_thread(
                    session_id=session_id,
                    channel_id=data["channel"],
                    thread_ts=data["ts"],
                )
                logger.info(
                    f"✅ Meeting notification sent for {visitor_name} with {host_name} | thread_ts={data['ts']}"
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
