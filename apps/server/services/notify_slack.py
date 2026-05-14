import logging
import requests
import threading
import os
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="slack_notifier")

_last_notified: dict = {}
_notify_lock = threading.Lock()


def _send_slack_notification_thread(
    employee_name: str, visitor_name: str, visitor_type: str, purpose: str
):
    logger.info(f"Slack thread started for {visitor_name} -> {employee_name}")

    if not SLACK_WEBHOOK_URL:
        logger.error("CRITICAL: SLACK_WEBHOOK_URL is NOT SET in environment variables.")
        return

    message = (
        f"🛎️ *Visitor Arrival for {employee_name}*\n"
        f"• *Visitor Name:* {visitor_name}\n"
        f"• *Category:* {visitor_type}\n"
        f"• *Purpose:* {purpose}\n\n"
        f"_Please head to the front desk._"
    )

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
        if response.status_code == 200:
            logger.info(
                f"✅ Successfully posted Slack notification for {employee_name}."
            )
        else:
            logger.error(
                f"❌ Failed to post to Slack. Status: {response.status_code}, Body: {response.text}"
            )
    except Exception as e:
        logger.error(f"❌ Slack Webhook exception: {e}")


def send_slack_arrival(
    employee_name: str,
    visitor_name: str,
    visitor_type: str,
    purpose: str,
    session_id: str,
):
    """Submit arrival notification to the thread pool; deduplicate per session."""
    if not SLACK_WEBHOOK_URL:
        print("DEBUG: SLACK_WEBHOOK_URL is missing!")

    with _notify_lock:
        if _last_notified.get(session_id) == visitor_name:
            logger.warning(
                f"Blocking duplicate notification for {visitor_name} in session {session_id}"
            )
            return
        _last_notified[session_id] = visitor_name

    logger.info(f"Queuing Slack notification for {visitor_name}...")
    _executor.submit(
        _send_slack_notification_thread,
        employee_name,
        visitor_name,
        visitor_type,
        purpose,
    )


def send_slack_meeting_scheduled(
    host_name: str,
    host_email: str,
    visitor_name: str,
    date_str: str,
    time_str: str,
    purpose: str,
    session_id: str,
):
    """Send a formatted meeting-scheduled notification (📅 style, matches second screenshot)."""
    if not SLACK_WEBHOOK_URL:
        logger.error("CRITICAL: SLACK_WEBHOOK_URL is NOT SET in environment variables.")
        return

    with _notify_lock:
        key = f"meeting_{session_id}"
        if _last_notified.get(key) == visitor_name:
            logger.warning(
                f"Blocking duplicate meeting notification for {visitor_name}"
            )
            return
        _last_notified[key] = visitor_name

    def _send():
        message = (
            f"📅 *New Meeting Scheduled via AIRA*\n"
            f"• *Host:* {host_name} ({host_email})\n"
            f"• *Visitor:* {visitor_name}\n"
            f"• *Date:* {date_str}  |  *Time:* {time_str}\n"
            f"• *Purpose:* {purpose}\n\n"
            f"_A Google Calendar invite with email has been sent to the host._"
        )
        try:
            response = requests.post(
                SLACK_WEBHOOK_URL, json={"text": message}, timeout=10
            )
            if response.status_code == 200:
                logger.info(
                    f"✅ Meeting notification sent for {visitor_name} with {host_name}."
                )
            else:
                logger.error(
                    f"❌ Slack meeting notification failed. Status: {response.status_code}"
                )
        except Exception as e:
            logger.error(f"❌ Slack meeting notification exception: {e}")

    _executor.submit(_send)


def clear_session(session_id: str):
    """Clean up the notification history for a session."""
    with _notify_lock:
        keys_to_remove = [k for k in _last_notified.keys() if k == session_id]
        for k in keys_to_remove:
            _last_notified.pop(k, None)
        logger.info(f"Cleared Slack notification cache for session {session_id}")
