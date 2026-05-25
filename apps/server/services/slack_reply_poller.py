"""
slack_reply_poller.py
---------------------
Polls Slack's conversations.replies / conversations.history API to fetch
host replies. No webhook, no tunnel, no Events API needed.

How it works:
  1. When a Slack notification is sent, notify_slack calls register_thread()
     for EACH message posted (DM and/or channel). Multiple threads can be
     registered for the same session.
  2. slack_watcher calls poll_for_reply(session_id) every 3 seconds.
  3. poll_for_reply checks ALL registered threads for that session and
     returns the first new reply from a non-bot user it finds.

Connection strategy:
  A single module-level AsyncClient is reused across all polls so that
  TCP connections are kept alive and connection-pool exhaustion
  (which causes ConnectTimeout on rapid repeated calls) is avoided.
"""

import logging
import os
import asyncio
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")

# ── Shared persistent client — reused across every poll call ─────────────────
_http_client: Optional[httpx.AsyncClient] = None
_http_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it if needed."""
    global _http_client
    async with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                timeout=httpx.Timeout(connect=10.0, read=10.0, write=5.0, pool=5.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30,
                ),
            )
        return _http_client


async def close_client() -> None:
    """Call on app shutdown to cleanly close the shared client."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ─────────────────────────────────────────────────────────────────────────────

# session_id -> LIST of thread entries, so we poll every place the message was sent.
# Each entry: {"channel": str, "thread_ts": str, "last_seen_ts": str}
_thread_registry: dict[str, list[dict]] = {}

# Cache display names to avoid repeated users.info calls
_display_name_cache: dict[str, str] = {}


def register_thread(session_id: str, channel_id: str, thread_ts: str) -> None:
    """
    Call this right after posting each Slack notification (DM or channel).
    Multiple calls for the same session are additive — all threads are polled.
    """
    entry = {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "last_seen_ts": thread_ts,  # skip the original bot message
    }

    if session_id not in _thread_registry:
        _thread_registry[session_id] = []

    # Avoid registering the exact same (channel, thread_ts) twice
    already = any(
        e["channel"] == channel_id and e["thread_ts"] == thread_ts
        for e in _thread_registry[session_id]
    )
    if not already:
        _thread_registry[session_id].append(entry)
        logger.info(
            "POLLER | registered | session=%s channel=%s thread_ts=%s (total threads: %d)",
            session_id,
            channel_id,
            thread_ts,
            len(_thread_registry[session_id]),
        )
    else:
        logger.debug(
            "POLLER | duplicate register ignored | session=%s channel=%s thread_ts=%s",
            session_id,
            channel_id,
            thread_ts,
        )


def rollback_last_seen(session_id: str) -> None:
    """Reset last_seen_ts on all threads so reply is re-delivered on next reconnect."""
    entries = _thread_registry.get(session_id, [])
    for entry in entries:
        entry["last_seen_ts"] = entry["thread_ts"]
    if entries:
        logger.info(
            "POLLER | rolled back last_seen_ts for session=%s (%d threads)",
            session_id,
            len(entries),
        )


def unregister_thread(session_id: str) -> None:
    _thread_registry.pop(session_id, None)


async def poll_for_reply(session_id: str) -> Optional[tuple[str, str]]:
    if not SLACK_BOT_TOKEN:
        logger.error("POLLER | SLACK_BOT_TOKEN not set")
        return None

    entries = _thread_registry.get(session_id)
    if not entries:
        logger.debug("POLLER | no thread registered for session=%s", session_id)
        return None

    # Poll every registered thread (DM + channel, or just one — doesn't matter)
    for entry in entries:
        result = await _poll_single_thread(session_id, entry)
        if result:
            return result

    return None


async def _poll_single_thread(
    session_id: str, entry: dict
) -> Optional[tuple[str, str]]:
    """Poll one (channel, thread_ts) pair and return (sender, text) if a new reply exists."""
    channel = entry["channel"]
    thread_ts = entry["thread_ts"]
    last_seen_ts = entry["last_seen_ts"]

    try:
        client = await _get_client()

        # Always use conversations.replies when we have a thread_ts.
        # This works for both public/private channels AND DM app conversations
        # (D... channels). conversations.history only returns top-level messages
        # and will miss replies made inside a DM thread with the App.
        resp = await client.get(
            "https://slack.com/api/conversations.replies",
            params={
                "channel": channel,
                "ts": thread_ts,
                "oldest": last_seen_ts,
                "limit": 10,
            },
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
                "POLLER | reply found | session=%s channel=%s sender='%s' text='%s'",
                session_id,
                channel,
                sender,
                reply_text,
            )
            return (sender, reply_text)

    except httpx.ConnectTimeout:
        logger.error(
            "POLLER | ConnectTimeout for session=%s channel=%s — will retry",
            session_id,
            channel,
        )
        global _http_client
        _http_client = None

    except httpx.ReadTimeout:
        logger.warning(
            "POLLER | ReadTimeout for session=%s channel=%s — will retry",
            session_id,
            channel,
        )

    except Exception as e:
        logger.error(
            "POLLER | poll failed for session=%s channel=%s: %s",
            session_id,
            channel,
            repr(e) if not str(e) else e,
        )

    return None


async def _get_display_name(user_id: str) -> str:
    if not user_id or not SLACK_BOT_TOKEN:
        return user_id

    if user_id in _display_name_cache:
        return _display_name_cache[user_id]

    try:
        client = await _get_client()
        resp = await client.get(
            "https://slack.com/api/users.info",
            params={"user": user_id},
        )
        data = resp.json()
        if data.get("ok"):
            p = data["user"]["profile"]
            name = p.get("display_name") or p.get("real_name", user_id)
            _display_name_cache[user_id] = name
            return name
        else:
            logger.error(
                "POLLER | Slack API error for user %s: %s",
                user_id,
                data.get("error"),
            )
    except Exception as e:
        logger.error("POLLER | name lookup failed for %s: %s", user_id, repr(e))

    return user_id
