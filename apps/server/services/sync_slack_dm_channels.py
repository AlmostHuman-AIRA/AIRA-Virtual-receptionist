"""
sync_slack_dm_channels.py
--------------------------
Fetches every employee's Slack user ID (via users.lookupByEmail) and
their DM channel ID (via conversations.open) and saves both into the
`employees` table.

Usage:
    1. Run migrate_add_slack_columns.py first (only needed once).
    2. Then run this script whenever new employees are added:
           python sync_slack_dm_channels.py

It is safe to re-run — it skips employees whose slack_dm_channel is
already populated and only processes new/missing ones.
Add --force flag to re-sync everyone:
           python sync_slack_dm_channels.py --force
"""

import asyncio
import logging
import os
import sqlite3
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── adjust this path to point at your actual office.db ──────────────────────
DB_PATH = Path(
    r"C:/Users/Administrator/Desktop/CPU-compatible-AI-/apps/server/receptionist/office.db"
)
# ─────────────────────────────────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def lookup_slack_user_id(client: httpx.AsyncClient, email: str) -> str | None:
    """Returns Slack user_id for the given email, or None if not found."""
    resp = await client.get(
        "https://slack.com/api/users.lookupByEmail",
        params={"email": email},
        headers=HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data.get("ok"):
        return data["user"]["id"]
    logger.warning("  ↳ lookupByEmail failed for %s: %s", email, data.get("error"))
    return None


async def open_dm_channel(client: httpx.AsyncClient, slack_user_id: str) -> str | None:
    """Opens (or retrieves) the DM channel between the bot and the given user."""
    resp = await client.post(
        "https://slack.com/api/conversations.open",
        json={"users": slack_user_id},
        headers=HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data.get("ok"):
        return data["channel"]["id"]
    logger.warning(
        "  ↳ conversations.open failed for %s: %s", slack_user_id, data.get("error")
    )
    return None


async def sync(force: bool = False):
    if not SLACK_BOT_TOKEN:
        logger.error("SLACK_BOT_TOKEN not set in .env — aborting.")
        sys.exit(1)

    if not DB_PATH.exists():
        logger.error("Database not found at: %s", DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Verify columns exist
    cur.execute("PRAGMA table_info(employees)")
    cols = {row["name"] for row in cur.fetchall()}
    if "slack_user_id" not in cols or "slack_dm_channel" not in cols:
        logger.error(
            "Columns slack_user_id / slack_dm_channel not found. "
            "Run migrate_add_slack_columns.py first."
        )
        conn.close()
        sys.exit(1)

    # Fetch employees to process
    if force:
        cur.execute(
            "SELECT id, name, email FROM employees WHERE email IS NOT NULL AND email != ''"
        )
        logger.info("--force: re-syncing ALL employees with an email.")
    else:
        cur.execute(
            "SELECT id, name, email FROM employees "
            "WHERE (slack_dm_channel IS NULL OR slack_dm_channel = '') "
            "AND email IS NOT NULL AND email != ''"
        )
        logger.info("Syncing only employees with missing slack_dm_channel.")

    employees = cur.fetchall()

    if not employees:
        logger.info(
            "✅ Nothing to sync — all employees already have Slack DM channels."
        )
        conn.close()
        return

    logger.info("Found %d employee(s) to sync.", len(employees))

    synced = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for emp in employees:
            emp_id, name, email = emp["id"], emp["name"], emp["email"]
            logger.info("Processing: %s (%s)", name, email)

            # Step 1: get Slack user ID from email
            slack_user_id = await lookup_slack_user_id(client, email)
            if not slack_user_id:
                logger.warning(
                    "  ✗ Could not find Slack user for %s — skipping.", email
                )
                failed += 1
                continue

            # Step 2: open DM channel
            dm_channel = await open_dm_channel(client, slack_user_id)
            if not dm_channel:
                logger.warning("  ✗ Could not open DM channel for %s — skipping.", name)
                failed += 1
                continue

            # Step 3: save to DB
            cur.execute(
                "UPDATE employees SET slack_user_id = ?, slack_dm_channel = ? WHERE id = ?",
                (slack_user_id, dm_channel, emp_id),
            )
            conn.commit()
            logger.info(
                "  ✓ %s → slack_user_id=%s  slack_dm_channel=%s",
                name,
                slack_user_id,
                dm_channel,
            )
            synced += 1

            # Small delay to avoid Slack rate limits
            await asyncio.sleep(0.5)

    conn.close()
    logger.info("─" * 50)
    logger.info("Done. Synced: %d  |  Failed/Skipped: %d", synced, failed)


if __name__ == "__main__":
    force = "--force" in sys.argv
    asyncio.run(sync(force=force))
