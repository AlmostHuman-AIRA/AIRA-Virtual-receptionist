"""
seed_data.py
------------
Seeds the normalized database with initial mock data, including 12 employees.
After seeding, automatically syncs Slack user IDs and DM channel IDs for all
employees — no need to run sync_slack_dm_channels.py separately.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from receptionist.database import SessionLocal, init_db
from receptionist.models import Employee, Settings

load_dotenv()

logger = logging.getLogger(__name__)


# ── Slack helpers (inlined from sync_slack_dm_channels.py) ───────────────────


async def _lookup_slack_user_id(client, email: str) -> str | None:
    """Returns Slack user_id for the given email, or None if not found."""
    import httpx

    token = os.getenv("SLACK_BOT_TOKEN", "")
    resp = await client.get(
        "https://slack.com/api/users.lookupByEmail",
        params={"email": email},
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    data = resp.json()
    if data.get("ok"):
        return data["user"]["id"]
    logger.warning("lookupByEmail failed for %s: %s", email, data.get("error"))
    return None


async def _open_dm_channel(client, slack_user_id: str) -> str | None:
    """Opens (or retrieves) the DM channel between the bot and the given user."""
    token = os.getenv("SLACK_BOT_TOKEN", "")
    resp = await client.post(
        "https://slack.com/api/conversations.open",
        json={"users": slack_user_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    data = resp.json()
    if data.get("ok"):
        return data["channel"]["id"]
    logger.warning(
        "conversations.open failed for %s: %s", slack_user_id, data.get("error")
    )
    return None


async def _sync_slack_for_employees(employees: list[Employee]) -> None:
    """
    Fetches Slack user IDs and DM channel IDs for the given Employee ORM objects
    and saves them in-place. Skips employees that already have a dm_channel set.
    Silently skips the whole sync if SLACK_BOT_TOKEN is not configured.
    """
    import httpx

    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        print(
            "⚠️  SLACK_BOT_TOKEN not set — skipping Slack sync. "
            "Set it in your .env and re-run seed_data.py (or run sync_slack_dm_channels.py) later."
        )
        return

    session = SessionLocal()
    synced = 0
    failed = 0

    try:
        async with httpx.AsyncClient() as client:
            for emp in employees:
                # Skip employees that already have both values populated
                if emp.slack_dm_channel:
                    continue

                if not emp.email:
                    logger.warning("No email for %s — skipping Slack sync.", emp.name)
                    failed += 1
                    continue

                print(f"  Syncing Slack for {emp.name} ({emp.email})…")

                slack_user_id = await _lookup_slack_user_id(client, emp.email)
                if not slack_user_id:
                    print(f"    ✗ Slack user not found for {emp.email}")
                    failed += 1
                    continue

                dm_channel = await _open_dm_channel(client, slack_user_id)
                if not dm_channel:
                    print(f"    ✗ Could not open DM channel for {emp.name}")
                    failed += 1
                    continue

                # Persist directly on the ORM object — session is still open
                db_emp = session.query(Employee).filter(Employee.id == emp.id).first()
                if db_emp:
                    db_emp.slack_user_id = slack_user_id
                    db_emp.slack_dm_channel = dm_channel
                    session.commit()

                print(
                    f"    ✓ {emp.name} → slack_user_id={slack_user_id}  "
                    f"slack_dm_channel={dm_channel}"
                )
                synced += 1

                # Small delay to respect Slack rate limits
                await asyncio.sleep(0.5)

        print(
            f"\n✅ Slack sync complete. Synced: {synced}  |  Failed/Skipped: {failed}"
        )

    except Exception as exc:
        logger.error("Slack sync error: %s", exc)
        print(f"⚠️  Slack sync encountered an error: {exc}")
    finally:
        session.close()


# ── Main seeding logic ────────────────────────────────────────────────────────


def seed_database():
    init_db()
    session = SessionLocal()

    try:
        # Check if already seeded
        if session.query(Employee).first():
            print("Database already seeded.")
            return

        print("Seeding initial company settings...")
        settings = [
            Settings(
                key="company_name",
                value="Sharp Software Development India Private Limited",
            ),
            Settings(key="company_address", value="123 Innovation Drive, Tech Park"),
            Settings(key="company_phone", value="+91-80-5555-0199"),
            Settings(key="company_email", value="contact@sharpsoftware.in"),
            Settings(key="company_website", value="www.sharpsoftware.in"),
        ]
        session.add_all(settings)

        print("Seeding 12 employees...")
        employees = [
            Employee(
                name="Priya",
                email="krutikakanchani847+priya@gmail.com",
                department="HR",
                role="HR Manager",
                location="Floor 2, Room 201",
                extension="101",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Arjun",
                email="sannidhivk15+arjun@gmail.com",
                department="Engineering",
                role="Lead Engineer",
                location="Floor 3, Desk 35",
                extension="102",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Suresh",
                email="krutikakanchani847+suresh@gmail.com",
                department="Management",
                role="CEO",
                location="Floor 5, Executive Suite",
                extension="100",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Jack",
                email="sannidhivk15+sannidhi@gmail.com",
                department="Sales",
                role="Sales Director",
                location="Floor 1, Room 105",
                extension="104",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="john",
                email="krutikakanchani847+john@gmail.com",
                department="IT Support",
                role="IT Administrator",
                location="Floor 1, Tech Bar",
                extension="110",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Virat",
                email="krutikaak07+virat@gmail.com",
                department="Design",
                role="UX/UI Designer",
                location="Floor 2, Creative Studio",
                extension="115",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Ravi",
                email="sannidhivk15+ravi@gmail.com",
                department="Finance",
                role="Chief Financial Officer",
                location="Floor 5, Room 502",
                extension="120",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Rahul",
                email="sannidhivk15+rahul@gmail.com",
                department="Marketing",
                role="Marketing Coordinator",
                location="Floor 2, Desk 12",
                extension="125",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Ramesh",
                email="sannidhivk15+ramesh@gmail.com",
                department="Engineering",
                role="Data Scientist",
                location="Floor 3, Desk 42",
                extension="130",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Lucy",
                email="lucy62648446@gmail.com",
                department="Legal",
                role="Legal Counsel",
                location="Floor 4, Room 410",
                extension="140",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Cookie",
                email="krutikakanchani847+cookie@gmail.com",
                department="Finance",
                role="Accountant",
                location="Floor 4, Desk 8",
                extension="145",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
            Employee(
                name="Jim",
                email="krutikaak07+jim@gmail.com",
                department="Operations",
                role="Operations Manager",
                location="Floor 1, Room 112",
                extension="150",
                is_public=True,
                slack_user_id=None,
                slack_dm_channel=None,
            ),
        ]
        session.add_all(employees)
        session.commit()

        # Refresh so each object has its auto-assigned .id
        for emp in employees:
            session.refresh(emp)

        print("✅ Database seeded successfully with 12 employees!")

    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
        return
    finally:
        session.close()

    # ── Auto-sync Slack after seeding ─────────────────────────────────────────
    print("\nStarting automatic Slack sync for all seeded employees…")
    asyncio.run(_sync_slack_for_employees(employees))


if __name__ == "__main__":
    seed_database()
