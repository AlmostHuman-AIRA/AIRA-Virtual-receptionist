# AIRA Virtual Receptionist: Technical Specification

## 1. System Overview
The **AlmostHuman AI (AIRA)** Virtual Receptionist is a multi-modal AI system optimized for CPU inference. It handles visitor greetings, employee identification, meeting scheduling, and office notifications through a real-time voice and vision interface.

## 2. Core Architecture
The system follows a hub-and-spoke architecture centered around a **FastAPI** backend that orchestrates several specialized AI processors.

### A. Backend (FastAPI)
- **Main Server (`main.py`):** Configures middleware, static files, and initializes the application lifespan.
- **WebSocket Route (`websocket_routes.py`):** Manages the high-frequency interaction loop, including audio streaming, VAD, and vision triggers.
- **Query Router (`services/query_router.py`):** The logic engine that extracts intents and manages the conversational state machine.
- **Connection Manager (`managers/connection_manager.py`):** Tracks active clients and manages task cancellation to prevent compute leaks.

### B. AI Inference Pipeline

The system is designed with a **Hybrid Acceleration** strategy. It is fully optimized for **CPU-only environments** (using quantized `int8` models) but will automatically detect and utilize **NVIDIA GPUs (CUDA)** if available for high-speed inference.

| Component | Technology | Role | Acceleration |
| :--- | :--- | :--- | :--- |
| **Wake Word** | `OpenWakeWord` | Triggers the system on "Hey Jarvis". | CPU (ONNX) |
| **VAD** | `Silero VAD` | Detects speech starts and ends. | CPU |
| **STT** | `Faster-Whisper` | Transcribes user speech into text. | **CUDA (fp16)** / CPU (int8) |
| **LLM** | `Groq (Llama-3)` | Handles conversational logic & NLU. | Cloud (Groq) |
| **TTS** | `Kokoro TTS` | Generates speech with lip-sync metadata. | **CUDA** / CPU |
| **Vision (Presence)** | `MediaPipe` | Detects persons in the frame. | CPU |
| **Vision (Identity)** | `DeepFace` | Verifies employees and visitors. | CPU / CUDA |

### C. Vision & Identity (`face_recognition_service.py`)
- **Library:** `DeepFace` with the `ArcFace` model and `SSD` detector.
- **Employee Verification:** Compares live frames against profile photos stored in `receptionist/photos/employees/`.
- **Visitor Tracking:** Captures reference photos for new visitors and performs session-long verification every 3 seconds to ensure the person hasn't changed.

## 3. Data & Persistence
- **Database:** SQLite (`office.db`) managed via SQLAlchemy.
- **Models:**
    - `Employee`: Directory including roles, locations, and photo paths.
    - `Visitor`: Persistent profiles for returning visitors.
    - `Meeting`: Scheduling logs linked to employees and visitors.
    - `ReceptionLog`: Audit trail of all check-ins and check-outs.
- **Retention:** Automatic deletion of visitor media and logs after 90 days.

## 4. Frontend Ecosystem
- **Virtual Kiosk (`apps/client`):**
    - **Talking Head:** 3D avatar (Three.js) with real-time viseme generation for lip-sync.
    - **Audio Streaming:** Continuous 16kHz PCM streaming over WebSockets.
    - **Hooks:** `useFaceVerification` and `usePresenceDetection` manage the interaction between the webcam and the backend vision services.
- **Admin & Analytics (`apps/dashboard`):**
    - A Next.js dashboard providing live reception views, visitor history, and employee photo management.

## 5. Integrations
- **Google Calendar:** Full integration for checking availability and booking meetings with native email invites.
- **Slack & Teams:** Automated arrival notifications sent to hosts when their visitor arrives or a meeting is booked.
