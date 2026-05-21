"""
slack_reply_poller.py
---------------------
Polls Slack's conversations.replies / conversations.history API to fetch
host replies. No webhook, no tunnel, no Events API needed.

How it works:
  1. When a Slack notification is sent, notify_slack saves the
     (thread_ts, channel_id) for that session into _thread_registry.
  2. slack_watcher calls poll_for_reply(session_id) every 3 seconds.
  3. poll_for_reply fetches the thread replies via Slack API and
     returns any new reply from a non-bot user.

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
# keepalive_expiry=30 keeps the TCP connection warm between 3-second poll cycles.
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
                    keepalive_expiry=30,  # seconds — longer than poll interval
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

# session_id -> {"channel": ..., "thread_ts": ..., "last_seen_ts": ...}
_thread_registry: dict[str, dict] = {}

# Cache display names to avoid repeated users.info calls
_display_name_cache: dict[str, str] = {}


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


def rollback_last_seen(session_id: str) -> None:
    """Reset last_seen_ts so reply is re-delivered on next reconnect."""
    entry = _thread_registry.get(session_id)
    if entry:
        entry["last_seen_ts"] = entry["thread_ts"]
        logger.info("POLLER | rolled back last_seen_ts for session=%s", session_id)


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

    # DM channels (D...) don't support threads — use conversations.history
    is_dm = channel.startswith("D")

    try:
        client = await _get_client()

        if is_dm:
            resp = await client.get(
                "https://slack.com/api/conversations.history",
                params={
                    "channel": channel,
                    "oldest": last_seen_ts,
                    "limit": 10,
                },
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

    except httpx.ConnectTimeout:
        logger.error(
            "POLLER | ConnectTimeout for session=%s — Slack unreachable, will retry",
            session_id,
        )
        # Force client recreation on next call in case the pool is stale
        global _http_client
        _http_client = None

    except httpx.ReadTimeout:
        logger.warning("POLLER | ReadTimeout for session=%s — will retry", session_id)

    except Exception as e:
        logger.error(
            "POLLER | poll failed for session=%s: %s",
            session_id,
            repr(e) if not str(e) else e,
        )

    return None


async def _get_display_name(user_id: str) -> str:
    if not user_id or not SLACK_BOT_TOKEN:
        return user_id

    # Return cached name if available
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
            _display_name_cache[user_id] = name  # cache for future calls
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
