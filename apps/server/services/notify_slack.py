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
    host_name, host_email, visitor_name, date_str, time_str, purpose, session_id
):
    with _notify_lock:
        key = f"meeting_{session_id}"
        # Include time in the value so rescheduled times are NOT blocked
        new_value = f"{visitor_name}_{date_str}_{time_str}"
        if _last_notified.get(key) == new_value:
            logger.warning(
                f"Blocking duplicate meeting notification for {visitor_name}"
            )
            return
        _last_notified[key] = new_value  # store time-aware key

    def _send():
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
                f"✅ Meeting notification sent for {visitor_name} with {host_name} | "
                f"thread_ts={data['ts']}"
            )

    _executor.submit(_send)


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
