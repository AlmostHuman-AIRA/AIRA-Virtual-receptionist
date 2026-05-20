"""
slack_reply_poller.py
---------------------
Polls Slack's conversations.replies API to fetch host replies.
No webhook, no tunnel, no Events API needed.

How it works:
  1. When a Slack notification is sent, notify_slack saves the
     (thread_ts, channel_id) for that session into _thread_registry.
  2. slack_watcher calls poll_for_reply(session_id) every 3 seconds.
  3. poll_for_reply fetches the thread replies via Slack API and
     returns any new reply from a non-bot user.
"""

import logging
import os
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")

# session_id -> {"channel": ..., "thread_ts": ..., "last_seen_ts": ...}
_thread_registry: dict[str, dict] = {}


def register_thread(session_id: str, channel_id: str, thread_ts: str) -> None:
    """
    Call this right after posting the Slack notification.
    Stores the channel + thread_ts so we know what to poll.
    """
    _thread_registry[session_id] = {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "last_seen_ts": thread_ts,  # skip the original bot message
    }
    logger.info(
        "POLLER | registered | session=%s channel=%s thread_ts=%s",
        session_id,
        channel_id,
        thread_ts,
    )


def unregister_thread(session_id: str) -> None:
    _thread_registry.pop(session_id, None)


async def poll_for_reply(session_id: str) -> Optional[tuple[str, str]]:
    if not SLACK_BOT_TOKEN:
        logger.error("POLLER | SLACK_BOT_TOKEN not set")
        return None

    entry = _thread_registry.get(session_id)
    if not entry:
        logger.debug("POLLER | no thread registered for session=%s", session_id)
        return None

    channel = entry["channel"]
    thread_ts = entry["thread_ts"]
    last_seen_ts = entry["last_seen_ts"]

    # ✅ DM channels (D...) don't support threads — use conversations.history instead
    is_dm = channel.startswith("D")

    try:
        async with httpx.AsyncClient() as client:
            if is_dm:
                resp = await client.get(
                    "https://slack.com/api/conversations.history",
                    params={
                        "channel": channel,
                        "oldest": last_seen_ts,  # only fetch messages after last seen
                        "limit": 10,
                    },
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    timeout=5,
                )
            else:
                resp = await client.get(
                    "https://slack.com/api/conversations.replies",
                    params={
                        "channel": channel,
                        "ts": thread_ts,
                        "oldest": last_seen_ts,
                        "limit": 10,
                    },
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    timeout=5,
                )
            data = resp.json()

        if not data.get("ok"):
            logger.warning(
                "POLLER | Slack API error: %s (channel=%s)", data.get("error"), channel
            )
            return None

        messages = data.get("messages", [])
        for msg in messages:
            msg_ts = msg.get("ts", "")
            if msg_ts <= last_seen_ts:
                continue
            if msg.get("subtype") == "bot_message" or msg.get("bot_id"):
                continue
            if not msg.get("text", "").strip():
                continue

            entry["last_seen_ts"] = msg_ts
            sender = await _get_display_name(msg.get("user", ""))
            reply_text = msg["text"].strip()
            logger.info(
                "POLLER | reply found | session=%s sender='%s' text='%s'",
                session_id,
                sender,
                reply_text,
            )
            return (sender, reply_text)

    except Exception as e:
        logger.error("POLLER | poll failed for session=%s: %s", session_id, e)

    return None


async def _get_display_name(user_id: str) -> str:
    if not user_id or not SLACK_BOT_TOKEN:
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
                return p.get("display_name") or p.get("real_name", user_id)
            else:
                logger.error(
                    "POLLER | Slack API error for user %s: %s",
                    user_id,
                    data.get("error"),
                )
                return user_id
    except Exception as e:
        logger.error("POLLER | name lookup failed for %s: %s", user_id, e)
    return user_id
