import httpx, os

from dotenv import load_dotenv
import os

# Point this to where your .env actually is
load_dotenv(r"C:\Users\Administrator\Desktop\CPU-compatible-AI-\apps\server\.env")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
print("Token loaded:", SLACK_BOT_TOKEN[:10] if SLACK_BOT_TOKEN else "NOT FOUND")


def verify_dm_channel(slack_user_id: str, slack_dm_channel: str):
    # Check user exists
    r = httpx.get(
        "https://slack.com/api/users.info",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        params={"user": slack_user_id},
    )
    print("User lookup:", r.json().get("ok"), r.json().get("error", ""))

    # Try opening DM — should return same channel ID
    r2 = httpx.post(
        "https://slack.com/api/conversations.open",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={"users": slack_user_id},
    )
    live_dm = r2.json().get("channel", {}).get("id")
    print(f"DB has dm_channel:   {slack_dm_channel}")
    print(f"Slack returns:       {live_dm}")
    print(f"Match: {slack_dm_channel == live_dm}")


verify_dm_channel("U0123ABC", "D0456DEF")  # paste values from Step 1
