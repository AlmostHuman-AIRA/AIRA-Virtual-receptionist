# test_slack_lookup.py — run this standalone to verify IDs
import httpx, os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("SLACK_BOT_TOKEN")
EMAIL = "lucy62648446@gmail.com"  # from your logs

# Step 1: lookup by email
r = httpx.get(
    "https://slack.com/api/users.lookupByEmail",
    headers={"Authorization": f"Bearer {TOKEN}"},
    params={"email": EMAIL},
)
data = r.json()
print("lookupByEmail response:", data)

if data.get("ok"):
    user_id = data["user"]["id"]
    print(f"User ID: {user_id}")
    print(f"Display name: {data['user']['profile'].get('display_name')}")
    print(f"Real name: {data['user']['profile'].get('real_name')}")

    # Step 2: open DM channel
    r2 = httpx.post(
        "https://slack.com/api/conversations.open",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"users": user_id},
    )
    data2 = r2.json()
    print("conversations.open response:", data2)
    if data2.get("ok"):
        print(f"DM channel ID: {data2['channel']['id']}")
        print(f"Is user DM (starts with D): {data2['channel']['id'].startswith('D')}")
