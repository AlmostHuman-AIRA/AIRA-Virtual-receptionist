/**
 * usePresenceDetection.ts
 * ───────────────────────
 * Sends periodic camera frames to the server while AIRA is in PASSIVE mode.
 * The server runs MediaPipe Face Detection and activates AIRA when a person
 * stands in front of the camera for ~4.5 s (3 consecutive positive frames).
 *
 * WHY 1.5 s interval?
 *   Fast enough to feel responsive (<5 s activation) but light enough
 *   to not waste CPU/bandwidth with redundant frames.
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { CameraStreamHandle } from '@/components/CameraStream';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

const PRESENCE_FRAME_INTERVAL_MS = 1500; // Send one frame every 1.5 seconds

export function usePresenceDetection(
  cameraRef: React.RefObject<CameraStreamHandle | null>,
  serverState: string // "passive" | "listening" | "processing" | "speaking"
) {
  const { sendPresenceFrame, onPersonDetected } = useWebSocketContext();
  const [personDetected, setPersonDetected] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Send periodic frames while PASSIVE ──────────────────────────────────
  useEffect(() => {
    // Only send frames when AIRA is passive (waiting for someone to approach)
    if (serverState !== 'passive') {
      // Clear interval if we leave passive mode
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      // Reset detection flag when leaving passive
      setPersonDetected(false);
      return;
    }

    if (!sendPresenceFrame) return;

    intervalRef.current = setInterval(() => {
      const frame = cameraRef.current?.captureFrame();
      if (frame) {
        sendPresenceFrame(frame);
      }
    }, PRESENCE_FRAME_INTERVAL_MS);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [serverState, cameraRef, sendPresenceFrame]);

  // ── Listen for person_detected event from server ────────────────────────
  useEffect(() => {
    if (!onPersonDetected) return;

    onPersonDetected(() => {
      console.log('👤 Person detected by server! AIRA activating...');
      setPersonDetected(true);

      // Reset after a short delay (the UI glow will show briefly)
      setTimeout(() => setPersonDetected(false), 3000);
    });
  }, [onPersonDetected]);

  return { personDetected };
}
