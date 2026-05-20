"""
migrate_add_slack_columns.py
-----------------------------
One-time migration: adds `slack_user_id` and `slack_dm_channel` columns
to the `employees` table if they don't already exist.

Run ONCE before running sync_slack_dm_channels.py:
    python migrate_add_slack_columns.py
"""

import sqlite3
from pathlib import Path

# ── adjust this path to point at your actual office.db ──────────────────────
DB_PATH = Path(
    r"C:/Users/Administrator/Desktop/CPU-compatible-AI-/apps/server/receptionist/office.db"
)
# ─────────────────────────────────────────────────────────────────────────────


def migrate():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check existing columns
    cur.execute("PRAGMA table_info(employees)")
    existing_columns = {row[1] for row in cur.fetchall()}

    added = []

    if "slack_user_id" not in existing_columns:
        cur.execute("ALTER TABLE employees ADD COLUMN slack_user_id TEXT")
        added.append("slack_user_id")

    if "slack_dm_channel" not in existing_columns:
        cur.execute("ALTER TABLE employees ADD COLUMN slack_dm_channel TEXT")
        added.append("slack_dm_channel")

    conn.commit()
    conn.close()

    if added:
        print(f"✅ Migration complete. Added columns: {', '.join(added)}")
    else:
        print("ℹ️  Columns already exist. No changes made.")


if __name__ == "__main__":
    migrate()
