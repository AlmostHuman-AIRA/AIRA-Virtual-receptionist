import { useState, useEffect } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { CameraStreamHandle } from '../components/CameraStream';

export function usePresenceDetection(
  cameraRef: React.RefObject<CameraStreamHandle | null>,
  serverState: string // "passive" | "listening" | "processing" | "speaking"
) {
  const { sendPresenceFrame, onPersonDetected } = useWebSocketContext();
  const [personDetected, setPersonDetected] = useState(false);

  useEffect(() => {
    // Only send frames when AIRA is passive (waiting for someone)
    if (serverState !== 'passive') return;

    const intervalId = setInterval(() => {
      const frame = cameraRef.current?.captureFrame();
      if (frame && sendPresenceFrame) {
        sendPresenceFrame(frame);
      }
    }, 1500); // Every 1.5 seconds

    return () => clearInterval(intervalId);
  }, [serverState, cameraRef, sendPresenceFrame]);

  // Listen for person_detected event from server
  useEffect(() => {
    if (onPersonDetected) {
      onPersonDetected(() => setPersonDetected(true));
    }
  }, [onPersonDetected]);

  // Reset personDetected when server goes back to passive
  useEffect(() => {
    if (serverState === 'passive') {
      setPersonDetected(false);
    }
  }, [serverState]);

  return { personDetected };
}
