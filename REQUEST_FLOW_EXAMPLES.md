# AIRA: Request & Execution Flow Examples

This document details the lifecycle of different request types in the AIRA system, from physical input to final action.

---

## 🕒 Workflow 1: AI-Initiated Activation (Auto-Wake)
*Triggered when a person stands in front of the kiosk.*

1.  **Presence Monitoring (Frontend):** While in `PASSIVE` mode, the `usePresenceDetection` hook captures a webcam frame every 1.5s.
2.  **Detection (Backend):** The frame is sent to `/ws/presence_frame`. `person_detection_service.py` (MediaPipe) checks for a face.
3.  **Confirmation:** If a face is seen for 2 consecutive frames, the backend emits `person_detected`.
4.  **Greeting:** The backend automatically generates a greeting ("Good Morning! Welcome to Sharp Software...").
5.  **Transition:** The system moves to `FOLLOWUP` mode, and the avatar begins speaking.

**Example:**
> *User walks up to the desk.*
> **AIRA (Visual):** Glows green.
> **AIRA (Audio):** "Good Afternoon! I am Jarvis, how can I assist you today?"

---

## 🗣️ Workflow 2: Voice Request (Conversational)
*Triggered when the user speaks to Jarvis.*

1.  **Capture:** The browser streams 16kHz mono audio to the backend via WebSocket.
2.  **Endpointing:** `Silero VAD` detects when the user starts and stops speaking.
3.  **Transcription:** `Faster-Whisper` converts the audio segment into text.
4.  **Intent Extraction:** The text is sent to `Groq LLM`. It identifies the **Intent** (e.g., `schedule_meeting`) and **Entities** (e.g., "Arjun", "Tomorrow at 2pm").
5.  **State Machine:** `query_router.py` checks availability for Arjun in the SQLite DB.
6.  **Action:**
    *   **External:** Sends a Google Calendar invite.
    *   **Notification:** Posts a Slack message to Arjun: *"New Meeting Scheduled with [Visitor Name]..."*
7.  **Response:** `Kokoro TTS` generates the confirmation audio + lip-sync timings.
8.  **Avatar:** The `TalkingHead` component receives the audio and animates the 3D model's mouth to the timings.

**Example:**
> **User:** "I'd like to schedule a meeting with Arjun tomorrow at 3 PM for a project update."
> **Jarvis:** "Certainly! Arjun is available at 3 PM tomorrow. I've sent him an invite and notified him on Slack. Is there anything else?"

---

## 👁️ Workflow 3: Identity Verification
*Triggered when a user identifies as an employee.*

1.  **Identification:** User says "I am Suresh."
2.  **Lookup:** Backend finds "Suresh" in the `employees` table.
3.  **Face Request:** Backend sends `employee_identified` to the frontend.
4.  **Capture:** Frontend captures a high-res JPEG from the webcam.
5.  **Verification:** `DeepFace` compares the live JPEG against `receptionist/photos/employees/3.jpg`.
6.  **Result:**
    *   **If Match:** Backend updates session to `is_verified = True` and Jarvis says "Welcome back, Suresh!"
    *   **If Mismatch:** Jarvis asks the user to check in as a visitor instead.

**Example:**
> **User:** "I am Priya from HR."
> **AIRA (UI):** Shows "Verifying identity..." badge.
> **DeepFace:** Distance 0.42 (Match).
> **Jarvis:** "Hello Priya! Good to see you. How can I help you in HR today?"

---

## 🛎️ Workflow 4: Visitor Check-In
*Triggered when a visitor completes their registration.*

1.  **Collection:** Jarvis asks for the visitor's name and purpose.
2.  **Logging:** `database.py` creates a `Visitor` profile and a `ReceptionLog` entry.
3.  **Badge Generation:** System assigns a Badge ID (e.g., `VIS-2024-0042`).
4.  **Alerting:** `notify_slack.py` sends an immediate notification to the host: *"🛎️ Visitor Arrival: [Name] is here to see you for [Purpose]."*
5.  **Tracking:** Frontend starts a 3-second background loop to ensure the visitor stays at the desk until the host arrives.

**Example:**
> **User:** "I'm Jack, here to see Lucy for a legal review."
> **Jarvis:** "Welcome Jack. I've logged your visit and Lucy has been notified on Slack. Please have a seat; she'll be with you shortly."
