"""
notification_service.py
────────────────────────────────────────────────────────────────────────────
Fires Slack notification when a meeting is scheduled via AIRA.

Email is handled automatically by Google Calendar via sendUpdates='all'
in calendar_service.send_calendar_invite — no separate SMTP needed.

Environment variable (set in apps/server/.env):
  SLACK_WEBHOOK_URL – Incoming Webhook URL
"""

import asyncio
import logging
import os
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def _post_slack(text: str) -> None:
    """Fire-and-forget Slack webhook POST (runs in a daemon thread)."""
    if not SLACK_WEBHOOK_URL:
        logger.info("Slack skipped — SLACK_WEBHOOK_URL not configured.")
        return
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        if resp.status_code != 200:
            logger.warning("Slack webhook returned %s: %s", resp.status_code, resp.text)
        else:
            logger.info("Slack meeting notification sent.")
    except Exception as exc:
        logger.warning("Slack post failed: %s", exc)


async def send_meeting_notification(
    employee_name: str,
    employee_email: Optional[str],
    organizer_name: str,
    meeting_date: str,
    meeting_time: str,
    purpose: str,
) -> bool:
    """
    Posts a Slack notification when a meeting is scheduled.
    Email invite is already sent by Google Calendar (sendUpdates='all').
    Runs the Slack POST in a daemon thread so it never blocks the event loop.
    """
    try:
        message = (
            f"📅 *New Meeting Scheduled via AIRA*\n"
            f"• *Host:* {employee_name}"
            + (f" ({employee_email})" if employee_email else "")
            + f"\n"
            f"• *Visitor:* {organizer_name}\n"
            f"• *Date:* {meeting_date}  |  *Time:* {meeting_time}\n"
            f"• *Purpose:* {purpose}\n\n"
            f"_A Google Calendar invite with email has been sent to the host._"
        )
        threading.Thread(target=_post_slack, args=(message,), daemon=True).start()
        await asyncio.sleep(0)
        return True
    except Exception as exc:
        logger.error("send_meeting_notification failed: %s", exc)
        return False
