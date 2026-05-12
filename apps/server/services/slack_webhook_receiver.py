"""
slack_webhook_receiver.py
--------------------------
Receives Slack Events API callbacks.

Saves every reply under:
  1. sender display name (lowercased)  → matched if sender IS the DB employee
  2. channel_id                        → fallback for ANY reply in the channel

Also exposes GET /slack/debug for live diagnosis.
"""

import hashlib
import hmac
import logging
import os
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from services.slack_reply_store import save_reply, dump_store

logger = logging.getLogger(__name__)
router = APIRouter()

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")

# channel_id → session_id mapping so we can update state["notification_channel_id"]
# populated on first reply received in a channel
_channel_session_map: dict[str, str] = {}


def _verify_slack_signature(request_body: bytes, headers: dict) -> bool:
    if not SLACK_SIGNING_SECRET:
        logger.warning("SLACK_SIGNING_SECRET not set — skipping verification (unsafe!)")
        return True
    timestamp = headers.get("x-slack-request-timestamp", "")
    slack_sig = headers.get("x-slack-signature", "")
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
    except ValueError:
        return False
    base = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    expected = (
        "v0="
        + hmac.new(
            SLACK_SIGNING_SECRET.encode(), base.encode(), hashlib.sha256
        ).hexdigest()
    )
    return hmac.compare_digest(expected, slack_sig)


async def _get_display_name(user_id: str) -> str:
    if not SLACK_BOT_TOKEN:
        return user_id
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://slack.com/api/users.info",
                params={"user": user_id},
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                timeout=5,
            )
            data = resp.json()
            if data.get("ok"):
                p = data["user"]["profile"]
                name = p.get("display_name") or p.get("real_name", user_id)
                logger.info("SLACK_RECV | resolved %s → '%s'", user_id, name)
                return name
    except Exception as e:
        logger.error("SLACK_RECV | name lookup failed for %s: %s", user_id, e)
    return user_id


@router.post("/slack/events")
async def slack_events(request: Request):
    body_bytes = await request.body()

    if not _verify_slack_signature(body_bytes, dict(request.headers)):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")

    payload = await request.json()

    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    if event.get("type") == "message":
        await _handle_message(event)

    return Response(status_code=200)


async def _handle_message(event: dict):
    user_id = event.get("user", "")
    channel_id = event.get("channel", "").strip()
    text = event.get("text", "").strip()

    # Ignore bot's own messages and edits/deletions
    if user_id == BOT_USER_ID or event.get("subtype") or not text:
        return

    sender_name = await _get_display_name(user_id)

    logger.info(
        "SLACK_RECV | from='%s' channel='%s' text='%s'",
        sender_name,
        channel_id,
        text,
    )

    # ── Save under BOTH keys ──────────────────────────────────────────────────
    # Key 1: sender display name  (works when sender == notified employee)
    # Key 2: channel_id           (works regardless of who replies)
    save_reply(sender_name, text, channel_id=channel_id)


# ── Debug endpoint ────────────────────────────────────────────────────────────


@router.get("/slack/debug")
async def slack_debug():
    """
    Open http://localhost:8000/slack/debug in your browser.

    Shows what's currently in the reply store.
    Use this to diagnose key mismatches:
      - 'by_host' keys = sender display names (lowercased)
      - 'by_channel' keys = Slack channel IDs

    If 'by_host' has 'sannidhivk' but your DB has 'Suresh',
    the channel fallback in slack_watcher will still relay the reply.
    """
    store = dump_store()
    logger.info("SLACK_DEBUG | %s", store)
    return {
        "store": store,
        "diagnosis": {
            "by_host_keys_explanation": "Slack display names of people who replied (lowercased)",
            "by_channel_keys_explanation": "Slack channel IDs where replies arrived",
            "what_to_check": (
                "If by_host keys don't match your DB employee names, "
                "the channel fallback still works — as long as "
                "state['notification_channel_id'] is set in the session. "
                "If by_channel is empty, your Events API subscription may not "
                "be receiving messages. Check your Slack App → Event Subscriptions."
            ),
        },
    }
