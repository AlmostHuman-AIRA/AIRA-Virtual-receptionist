import httpx, os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("SLACK_BOT_TOKEN")
r = httpx.get(
    "https://slack.com/api/auth.test", headers={"Authorization": f"Bearer {token}"}
)
print("Bot identity:", r.json())
r2 = httpx.get(
    "https://slack.com/api/users.lookupByEmail",
    headers={"Authorization": f"Bearer {token}"},
    params={"email": "lucy62648446@gmail.com"},
)
print("Lucy lookup:", r2.json())
