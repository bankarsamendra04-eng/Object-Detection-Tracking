import { useState, useEffect, useRef, useCallback } from 'react';
import { VisionWebSocketClient, ConnectionStatus } from '../api/websocket';

export function useWebSocketStream() {
  const clientRef = useRef(null);
  const [status, setStatus] = useState(ConnectionStatus.DISCONNECTED);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [streamMetadata, setStreamMetadata] = useState(null);
  const [error, setError] = useState(null);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    const client = new VisionWebSocketClient();
    clientRef.current = client;

    const unsubs = [
      client.on('status_change', (newStatus) => {
        setStatus(newStatus);
        if (newStatus === ConnectionStatus.STREAMING) {
          setIsPaused(false);
          setError(null);
        } else if (newStatus === ConnectionStatus.PAUSED) {
          setIsPaused(true);
        }
      }),

      client.on('connection_ack', (data) => {
        setStatus(ConnectionStatus.CONNECTED);
      }),

      client.on('stream_started', (data) => {
        setStreamMetadata(data);
        setError(null);
      }),

      client.on('frame_result', (frameData) => {
        setCurrentFrame(frameData);
      }),

      client.on('stream_stopped', () => {
        setIsPaused(false);
      }),

      client.on('stream_error', (errData) => {
        setError(errData.message || 'Stream processing error');
      }),

      client.on('error', (err) => {
        setError(err.message || 'WebSocket connection error');
      }),
    ];

    client.connect();

    return () => {
      unsubs.forEach((unsub) => unsub());
      client.disconnect();
      clientRef.current = null;
    };
  }, []);

  const start = useCallback((options) => {
    setError(null);
    if (clientRef.current) {
      clientRef.current.startStream(options);
    }
  }, []);

  const stop = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.stopStream();
    }
  }, []);

  const pause = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.pauseStream();
    }
  }, []);

  const resume = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.resumeStream();
    }
  }, []);

  return {
    status,
    currentFrame,
    streamMetadata,
    error,
    isPaused,
    start,
    stop,
    pause,
    resume,
    isStreaming: status === ConnectionStatus.STREAMING,
    isConnected: status === ConnectionStatus.CONNECTED || status === ConnectionStatus.STREAMING || status === ConnectionStatus.PAUSED,
  };
}
