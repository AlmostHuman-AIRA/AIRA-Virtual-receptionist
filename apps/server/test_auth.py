import logging

logging.basicConfig(level=logging.INFO)

from services.calendar_service import get_calendar_service

print("--- Starting Auth Flow ---")
print(
    "A browser window should open. Please log in with your Google account and grant permissions."
)
service = get_calendar_service()
if service:
    print("\nSUCCESS! token.json was generated successfully.")
else:
    print("\nFAILED to authenticate.")
