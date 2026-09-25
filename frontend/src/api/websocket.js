/**
 * Production WebSocket Streaming Client for Live Computer Vision
 * Implements full protocol, reconnection, event callbacks, and clean teardown.
 */

export const ConnectionStatus = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  STREAMING: 'streaming',
  PAUSED: 'paused',
  ERROR: 'error',
};

export class VisionWebSocketClient {
  constructor(url = null) {
    this.customUrl = url;
    this.ws = null;
    this.sessionId = null;
    this.status = ConnectionStatus.DISCONNECTED;
    this.listeners = new Map();
    this.heartbeatTimer = null;
    this.reconnectTimer = null;
    this.shouldReconnect = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  getWebSocketUrl() {
    if (this.customUrl) return this.customUrl;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/stream`;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.setStatus(ConnectionStatus.CONNECTING);
    const url = this.getWebSocketUrl();

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      this.setStatus(ConnectionStatus.ERROR);
      this.emit('error', { code: 'INIT_ERROR', message: err.message });
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.startHeartbeat();
    };

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        this.handleMessage(payload);
      } catch (err) {
        console.error('Failed to parse WebSocket message JSON:', err);
      }
    };

    this.ws.onerror = (err) => {
      this.setStatus(ConnectionStatus.ERROR);
      this.emit('error', { code: 'SOCKET_ERROR', message: 'WebSocket network error occurred.' });
    };

    this.ws.onclose = (event) => {
      this.stopHeartbeat();
      this.setStatus(ConnectionStatus.DISCONNECTED);
      this.emit('close', event);

      if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
        this.reconnectTimer = setTimeout(() => this.connect(), delay);
      }
    };
  }

  handleMessage(msg) {
    const { type, session_id, data } = msg;
    if (session_id) this.sessionId = session_id;

    switch (type) {
      case 'connection_ack':
        this.setStatus(ConnectionStatus.CONNECTED);
        this.emit('connection_ack', data);
        break;

      case 'stream_started':
        this.setStatus(ConnectionStatus.STREAMING);
        this.emit('stream_started', data);
        break;

      case 'frame_result':
        this.emit('frame_result', data);
        break;

      case 'stream_stopped':
        this.setStatus(ConnectionStatus.CONNECTED);
        this.emit('stream_stopped', data);
        break;

      case 'error':
        this.emit('stream_error', data);
        break;

      case 'pong':
        this.emit('pong', data);
        break;

      default:
        this.emit('message', msg);
    }
  }

  startStream({ source = 'webcam', cameraIndex = 0, path = null, fpsLimit = 15, conf = 0.35, iou = 0.45, annotate = true }) {
    if (!this.isConnected()) {
      this.connect();
    }

    const command = {
      action: 'start',
      source,
      camera_index: Number(cameraIndex),
      path,
      fps_limit: Number(fpsLimit),
      conf: Number(conf),
      iou: Number(iou),
      annotate: Boolean(annotate),
    };

    this.sendCommand(command);
  }

  stopStream() {
    this.sendCommand({ action: 'stop' });
  }

  pauseStream() {
    this.sendCommand({ action: 'pause' });
    this.setStatus(ConnectionStatus.PAUSED);
  }

  resumeStream() {
    this.sendCommand({ action: 'resume' });
    this.setStatus(ConnectionStatus.STREAMING);
  }

  ping() {
    this.sendCommand({ action: 'ping' });
  }

  sendCommand(cmd) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
    } else {
      console.warn('Cannot send command; WebSocket is not open.', cmd);
    }
  }

  startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.ping();
      }
    }, 25000);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus(ConnectionStatus.DISCONNECTED);
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  setStatus(newStatus) {
    if (this.status !== newStatus) {
      this.status = newStatus;
      this.emit('status_change', newStatus);
    }
  }

  on(event, handler) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(handler);
    return () => this.off(event, handler);
  }

  off(event, handler) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(handler);
    }
  }

  emit(event, payload) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach((cb) => {
        try {
          cb(payload);
        } catch (e) {
          console.error(`Error in WebSocket listener for '${event}':`, e);
        }
      });
    }
  }
}
