"""
slack_webhook_receiver.py
--------------------------
FastAPI router that receives Slack Events API callbacks.

HOW IT WORKS:
  1. In your Slack App settings, enable "Event Subscriptions".
  2. Set Request URL to: https://<your-domain>/slack/events
  3. Subscribe to bot event: `message.channels` (or `message.groups` for private)
  4. When HR replies in the notification channel/thread, Slack POSTs here.
  5. We extract the text + the user's display name → save to slack_reply_store.

IMPORTANT — Slack thread context:
  When notify_slack.py posts a message, Slack returns a `ts` (timestamp) for
  that message. To get ONLY replies in that specific thread (not the whole channel),
  you should store that `ts` alongside the employee_name and compare it to the
  `thread_ts` field in incoming events. For simplicity, this implementation
  matches on the employee mention in the text, which works fine for small teams.
  See the TODO below if you want strict thread-level matching.
"""

import hashlib
import hmac
import logging
import os
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from services.slack_reply_store import save_reply

logger = logging.getLogger(__name__)

router = APIRouter()

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID", "")  # Your bot's Slack user ID


# ─────────────────────────────────────────────────────────────────────────────
# SIGNATURE VERIFICATION  (security — never skip this in production)
# ─────────────────────────────────────────────────────────────────────────────


def _verify_slack_signature(request_body: bytes, headers: dict) -> bool:
    """Validates that the request genuinely came from Slack."""
    if not SLACK_SIGNING_SECRET:
        logger.warning("SLACK_SIGNING_SECRET not set — skipping verification (unsafe!)")
        return True

    timestamp = headers.get("x-slack-request-timestamp", "")
    slack_signature = headers.get("x-slack-signature", "")

    # Reject stale requests (replay attack protection)
    if abs(time.time() - int(timestamp)) > 300:
        return False

    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    my_signature = (
        "v0="
        + hmac.new(
            SLACK_SIGNING_SECRET.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    return hmac.compare_digest(my_signature, slack_signature)


# ─────────────────────────────────────────────────────────────────────────────
# SLACK USER → DISPLAY NAME LOOKUP
# ─────────────────────────────────────────────────────────────────────────────


async def _get_slack_display_name(user_id: str, bot_token: str) -> str:
    """
    Calls Slack users.info to resolve a user_id like 'U0123ABC' to a real name.
    Caches nothing here for simplicity — add an lru_cache if needed.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://slack.com/api/users.info",
                params={"user": user_id},
                headers={"Authorization": f"Bearer {bot_token}"},
                timeout=5,
            )
            data = resp.json()
            if data.get("ok"):
                profile = data["user"]["profile"]
                return profile.get("display_name") or profile.get("real_name", user_id)
    except Exception as e:
        logger.error("Could not resolve Slack user %s: %s", user_id, e)
    return user_id


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVENT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/slack/events")
async def slack_events(request: Request):
    body_bytes = await request.body()
    headers = dict(request.headers)

    # 1. Verify signature
    if not _verify_slack_signature(body_bytes, headers):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")

    payload = await request.json()

    # 2. Slack URL verification handshake (one-time, when you first add the URL)
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    # 3. Handle actual events
    event = payload.get("event", {})
    event_type = event.get("type")

    if event_type == "message":
        await _handle_message_event(event)

    # Slack expects a 200 quickly — always return fast
    return Response(status_code=200)


async def _handle_message_event(event: dict):
    """
    Processes an incoming Slack message event and stores the reply
    so query_router.py can pick it up.
    """
    # Ignore messages from the bot itself (prevents echo loops)
    user_id = event.get("user", "")
    if user_id == BOT_USER_ID:
        return

    # Ignore edited/deleted sub-types
    if event.get("subtype"):
        return

    text: str = event.get("text", "").strip()
    if not text:
        return

    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    sender_name = await _get_slack_display_name(user_id, bot_token)

    logger.info("Slack message from %s (%s): %s", sender_name, user_id, text)

    # ── Store the reply keyed by the SENDER's name ───────────────────────────
    # When your notification says "Visitor for HR Manager John", and John replies,
    # we store the reply under John's display name. query_router looks it up
    # using state["sched_employee_name"] or state["meeting_with_resolved"].
    #
    # TODO (strict thread matching): Also store the `ts` from notify_slack.py
    # and only save replies where event["thread_ts"] == that stored ts.
    # This eliminates false matches in busy channels.
    # ─────────────────────────────────────────────────────────────────────────
    save_reply(sender_name, text)
