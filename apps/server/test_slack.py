import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

token = os.getenv("SLACK_BOT_TOKEN")
channel = os.getenv("SLACK_CHANNEL_ID")

print(f"Token loaded: {'Yes' if token else 'No'}")
print(f"Channel loaded: {channel}")

if not token or not channel:
    print("❌ ERROR: Missing SLACK_BOT_TOKEN or SLACK_CHANNEL_ID in .env")
    exit()

# Try to send a message
print("Sending test message to Slack...")
resp = httpx.post(
    "https://slack.com/api/chat.postMessage",
    headers={"Authorization": f"Bearer {token}"},
    json={"channel": channel, "text": "✅ Hello! AIRA Slack integration is working!"},
)

data = resp.json()
if data.get("ok"):
    print("✅ SUCCESS! Message sent to Slack.")
else:
    print(f"❌ SLACK API ERROR: {data.get('error')}")
    if data.get("error") == "not_in_channel":
        print("Fix: Go to Slack and type @YourBotName in the channel to invite it.")
    elif data.get("error") == "invalid_auth":
        print("Fix: Your SLACK_BOT_TOKEN is incorrect or expired.")
