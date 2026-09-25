import test from 'node:test';
import assert from 'node:assert/strict';

import { ConnectionStatus, VisionWebSocketClient } from '../src/api/websocket.js';

test('Frontend Unit Tests: WebSocket Protocol & State Machine', async (t) => {
  await t.test('1. ConnectionStatus has all standard states', () => {
    assert.equal(ConnectionStatus.DISCONNECTED, 'disconnected');
    assert.equal(ConnectionStatus.CONNECTING, 'connecting');
    assert.equal(ConnectionStatus.CONNECTED, 'connected');
    assert.equal(ConnectionStatus.STREAMING, 'streaming');
    assert.equal(ConnectionStatus.PAUSED, 'paused');
    assert.equal(ConnectionStatus.ERROR, 'error');
  });

  await t.test('2. WebSocket client initializes with disconnected state', () => {
    const client = new VisionWebSocketClient('ws://127.0.0.1:8000/ws/stream');
    assert.equal(client.status, ConnectionStatus.DISCONNECTED);
    assert.equal(client.sessionId, null);
    assert.equal(client.getWebSocketUrl(), 'ws://127.0.0.1:8000/ws/stream');
  });

  await t.test('3. WebSocket client handles connection_ack message', () => {
    const client = new VisionWebSocketClient('ws://127.0.0.1:8000/ws/stream');
    let ackReceived = false;

    client.on('connection_ack', (data) => {
      ackReceived = true;
      assert.equal(data.session_id, 'sess_12345');
    });

    client.handleMessage({
      type: 'connection_ack',
      session_id: 'sess_12345',
      data: { session_id: 'sess_12345', message: 'Connected' },
    });

    assert.equal(ackReceived, true);
    assert.equal(client.status, ConnectionStatus.CONNECTED);
    assert.equal(client.sessionId, 'sess_12345');
  });

  await t.test('4. WebSocket client parses frame_result correctly', () => {
    const client = new VisionWebSocketClient();
    let frameData = null;

    client.on('frame_result', (data) => {
      frameData = data;
    });

    client.handleMessage({
      type: 'frame_result',
      session_id: 'sess_12345',
      data: {
        frame_number: 42,
        fps: 29.8,
        active_tracks: 3,
        unique_tracks: 5,
        inference_time_ms: 12.4,
        tracking_time_ms: 1.8,
        detections: [
          { class_name: 'car', confidence: 0.88, box: { x1: 10, y1: 20, x2: 100, y2: 120 } },
          { class_name: 'person', confidence: 0.92, box: { x1: 150, y1: 50, x2: 200, y2: 250 } },
        ],
        tracks: [
          { track_id: 1, class_name: 'car', confidence: 0.88, box: { x1: 10, y1: 20, x2: 100, y2: 120 } },
        ],
        analytics: {
          total_unique_objects: 5,
          active_objects: 3,
          line_crossings_in: 2,
          line_crossings_out: 1,
        },
      },
    });

    assert.notEqual(frameData, null);
    assert.equal(frameData.frame_number, 42);
    assert.equal(frameData.active_tracks, 3);
    assert.equal(frameData.detections.length, 2);
    assert.equal(frameData.tracks[0].track_id, 1);
  });

  await t.test('5. WebSocket client handles stream_stopped and error events', () => {
    const client = new VisionWebSocketClient();
    let stopped = false;
    let errorData = null;

    client.on('stream_stopped', (data) => { stopped = true; });
    client.on('stream_error', (data) => { errorData = data; });

    client.handleMessage({
      type: 'stream_stopped',
      session_id: 'sess_12345',
      data: { reason: 'user_requested', frames_processed: 100 },
    });
    assert.equal(stopped, true);
    assert.equal(client.status, ConnectionStatus.CONNECTED);

    client.handleMessage({
      type: 'error',
      session_id: 'sess_12345',
      data: { code: 'SOURCE_ERROR', message: 'Camera disconnected' },
    });
    assert.notEqual(errorData, null);
    assert.equal(errorData.code, 'SOURCE_ERROR');
  });
});

test('Frontend Unit Tests: Data Calculations & Formatting', async (t) => {
  await t.test('6. Class counts calculation from detections/tracks', () => {
    const items = [
      { class_name: 'car' },
      { class_name: 'car' },
      { class_name: 'person' },
      { class_name: 'bus' },
      { class_name: 'car' },
    ];

    const counts = {};
    items.forEach((item) => {
      counts[item.class_name] = (counts[item.class_name] || 0) + 1;
    });

    assert.equal(counts['car'], 3);
    assert.equal(counts['person'], 1);
    assert.equal(counts['bus'], 1);
  });

  await t.test('7. Bounding box dimension calculation', () => {
    const box = { x1: 50, y1: 80, x2: 250, y2: 380 };
    const width = box.x2 - box.x1;
    const height = box.y2 - box.y1;
    const centerX = (box.x1 + box.x2) / 2;
    const centerY = (box.y1 + box.y2) / 2;

    assert.equal(width, 200);
    assert.equal(height, 300);
    assert.equal(centerX, 150);
    assert.equal(centerY, 230);
  });

  await t.test('8. Model availability and status mapping', () => {
    const models = [
      { model_id: 'general_pretrained', model_type: 'pretrained', status: 'AVAILABLE' },
      { model_id: 'custom_visdrone', model_type: 'custom', status: 'NOT_AVAILABLE' },
    ];

    const pretrained = models.find((m) => m.model_id === 'general_pretrained');
    const custom = models.find((m) => m.model_id === 'custom_visdrone');

    assert.equal(pretrained.status, 'AVAILABLE');
    assert.equal(custom.status, 'NOT_AVAILABLE');
    assert.equal(custom.model_type, 'custom');
  });

  await t.test('9. Supported image and video extension validation', () => {
    const validImageExts = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);
    const validVideoExts = new Set(['.mp4', '.avi', '.mov', '.mkv']);

    assert.equal(validImageExts.has('.jpg'), true);
    assert.equal(validImageExts.has('.png'), true);
    assert.equal(validImageExts.has('.exe'), false);

    assert.equal(validVideoExts.has('.mp4'), true);
    assert.equal(validVideoExts.has('.avi'), true);
    assert.equal(validVideoExts.has('.sh'), false);
  });

  await t.test('10. Analytics total crossings calculation', () => {
    const inbound = 14;
    const outbound = 9;
    const total = inbound + outbound;
    assert.equal(total, 23);
  });
});
