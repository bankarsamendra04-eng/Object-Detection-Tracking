import json
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.streaming.manager import get_stream_manager
from backend.app.services.streaming.session import StreamSession
from backend.app.api.schemas.websocket import (
    ClientCommand,
    ServerMessageType,
    WebSocketMessage,
    FrameResultData,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def temp_sample_video():
    """Generates a small 15-frame synthetic MP4 video in data/samples/ for bounded testing."""
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    temp_video_path = samples_dir / "temp_test_stream.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video_path), fourcc, 15.0, (320, 240))

    for i in range(15):
        frame = np.full((240, 320, 3), 50, dtype=np.uint8)
        # Draw moving rectangle simulating object
        cx = 30 + i * 15
        cy = 100
        cv2.rectangle(frame, (cx, cy), (cx + 50, cy + 80), (0, 255, 0), -1)
        out.write(frame)

    out.release()
    yield "data/samples/temp_test_stream.mp4"

    # Cleanup
    if temp_video_path.exists():
        try:
            temp_video_path.unlink()
        except Exception:
            pass


# ==========================================
# 1. Connection & Lifecycle Tests
# ==========================================

def test_ws_connection_and_ack(client):
    """Verifies WebSocket connection handshake returns structured connection_ack."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        ack = ws.receive_json()
        assert ack["type"] == ServerMessageType.CONNECTION_ACK.value
        assert "session_id" in ack
        assert "supported_sources" in ack["data"]
        assert "webcam" in ack["data"]["supported_sources"]
        assert "video" in ack["data"]["supported_sources"]


def test_ws_root_alias_connection(client):
    """Verifies root alias /ws/stream connects successfully."""
    with client.websocket_connect("/ws/stream") as ws:
        ack = ws.receive_json()
        assert ack["type"] == ServerMessageType.CONNECTION_ACK.value
        assert "session_id" in ack


def test_ws_ping_pong(client):
    """Verifies ping command returns pong with server timestamp."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # connection_ack

        ws.send_json({"action": "ping"})
        msg = ws.receive_json()
        assert msg["type"] == ServerMessageType.PONG.value
        assert "timestamp" in msg["data"]


# ==========================================
# 2. Command Validation & Error Handling
# ==========================================

def test_ws_invalid_json(client):
    """Verifies sending invalid non-JSON payload returns structured error."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_text("THIS IS NOT JSON")
        err = ws.receive_json()
        assert err["type"] == ServerMessageType.ERROR.value
        assert err["data"]["code"] == "INVALID_JSON"


def test_ws_invalid_command_action(client):
    """Verifies sending unknown command action returns INVALID_COMMAND error."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({"action": "unknown_action"})
        err = ws.receive_json()
        assert err["type"] == ServerMessageType.ERROR.value
        assert err["data"]["code"] == "INVALID_COMMAND"


def test_ws_path_traversal_prevention(client):
    """Verifies that path traversal attempts in video path are rejected safely."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "../../../secret/password.txt",
        })
        err = ws.receive_json()
        assert err["type"] == ServerMessageType.ERROR.value
        # Rejection occurs either at schema validation (INVALID_COMMAND) or path resolution (SOURCE_UNAVAILABLE)
        assert err["data"]["code"] in {"INVALID_COMMAND", "SOURCE_UNAVAILABLE"}


def test_ws_missing_video_file(client):
    """Verifies attempting to stream a non-existent video returns SOURCE_UNAVAILABLE."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "data/samples/does_not_exist_12345.mp4",
        })
        err = ws.receive_json()
        assert err["type"] == ServerMessageType.ERROR.value
        assert err["data"]["code"] == "SOURCE_UNAVAILABLE"


def test_ws_unavailable_webcam(client):
    """Verifies connecting to an invalid webcam index does not crash server and returns error."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({
            "action": "start",
            "source": "webcam",
            "camera_index": 9999,
        })
        err = ws.receive_json()
        assert err["type"] == ServerMessageType.ERROR.value
        assert err["data"]["code"] == "SOURCE_UNAVAILABLE"


# ==========================================
# 3. Real-Time Streaming & Frame Results
# ==========================================

def test_ws_video_streaming_and_frame_results(client, temp_sample_video):
    """
    Tests end-to-end video streaming over WebSocket:
    Start stream -> receive stream_started -> receive frame_results -> stop -> receive stream_stopped.
    """
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        ack = ws.receive_json()
        assert ack["type"] == ServerMessageType.CONNECTION_ACK.value

        # 1. Start streaming synthetic video with annotation enabled
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": temp_sample_video,
            "fps_limit": 30.0,
            "stride": 1,
            "annotate": True,
        })

        # 2. Receive stream_started acknowledgement
        started = ws.receive_json()
        assert started["type"] == ServerMessageType.STREAM_STARTED.value
        assert started["data"]["source"] == "video"
        assert started["data"]["fps_limit"] == 30.0

        # 3. Collect multiple frame_result messages
        frame_results = []
        for _ in range(5):
            msg = ws.receive_json()
            if msg["type"] == ServerMessageType.FRAME_RESULT.value:
                frame_results.append(msg)
            elif msg["type"] == ServerMessageType.STREAM_STOPPED.value:
                break

        assert len(frame_results) >= 1
        first_frame = frame_results[0]
        data = first_frame["data"]

        # Validate structured fields
        assert "frame_number" in data
        assert "timestamp_ms" in data
        assert "inference_time_ms" in data
        assert "tracking_time_ms" in data
        assert "total_time_ms" in data
        assert "fps" in data
        assert "active_tracks" in data
        assert "unique_tracks" in data
        assert "detections" in data
        assert "tracks" in data
        assert "analytics" in data
        assert "annotated_frame" in data
        assert isinstance(data["annotated_frame"], str)  # base64 encoded

        # 4. Stop stream
        ws.send_json({"action": "stop"})
        stopped = ws.receive_json()
        # Drain any buffered frames until stream_stopped is received
        while stopped["type"] == ServerMessageType.FRAME_RESULT.value:
            stopped = ws.receive_json()

        assert stopped["type"] == ServerMessageType.STREAM_STOPPED.value
        assert stopped["data"]["reason"] == "User stopped stream"


def test_ws_pause_and_resume(client, temp_sample_video):
    """Verifies that pausing and resuming stream works cleanly."""
    with client.websocket_connect("/api/v1/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({
            "action": "start",
            "source": "video",
            "path": temp_sample_video,
            "fps_limit": 20.0,
        })

        _ = ws.receive_json()  # started
        msg = ws.receive_json()
        assert msg["type"] == ServerMessageType.FRAME_RESULT.value

        # Send pause
        ws.send_json({"action": "pause"})
        time.sleep(0.1)

        # Send resume
        ws.send_json({"action": "resume"})
        time.sleep(0.1)

        # Stop
        ws.send_json({"action": "stop"})


# ==========================================
# 4. Multi-Client Isolation & Cleanup
# ==========================================

def test_ws_multi_client_isolation(client):
    """Verifies that multiple concurrent clients maintain isolated sessions."""
    manager = get_stream_manager()
    initial_count = manager.active_connections_count

    with client.websocket_connect("/api/v1/ws/stream") as ws1:
        ack1 = ws1.receive_json()
        session1 = ack1["session_id"]
        assert manager.active_connections_count == initial_count + 1

        with client.websocket_connect("/api/v1/ws/stream") as ws2:
            ack2 = ws2.receive_json()
            session2 = ack2["session_id"]
            assert session1 != session2
            assert manager.active_connections_count == initial_count + 2

            # Client 1 pings
            ws1.send_json({"action": "ping"})
            pong1 = ws1.receive_json()
            assert pong1["type"] == ServerMessageType.PONG.value

            # Client 2 pings
            ws2.send_json({"action": "ping"})
            pong2 = ws2.receive_json()
            assert pong2["type"] == ServerMessageType.PONG.value

        # Client 2 disconnected, Client 1 should still be connected
        assert manager.active_connections_count == initial_count + 1

    # Both disconnected
    assert manager.active_connections_count == initial_count


def test_ws_backpressure_queue_dropping():
    """
    Directly tests backpressure handling in StreamSession:
    When queue is full (maxsize=10), enqueueing new frame_results drops the oldest frame
    without stalling or throwing an error.
    """
    import asyncio
    from unittest.mock import MagicMock

    async def _run():
        dummy_ws = MagicMock()
        session = StreamSession(session_id="test_bp", websocket=dummy_ws)

        # Fill queue to maximum capacity (10 items)
        for i in range(10):
            frame_msg = WebSocketMessage(
                type=ServerMessageType.FRAME_RESULT,
                session_id="test_bp",
                data=FrameResultData(
                    frame_number=i,
                    timestamp_ms=float(i * 100),
                    inference_time_ms=10.0,
                    tracking_time_ms=5.0,
                    total_time_ms=15.0,
                    fps=30.0,
                    active_tracks=0,
                    unique_tracks=0,
                    detections=[],
                    tracks=[],
                    analytics={},
                ),
            )
            await session.enqueue_message(frame_msg)

        assert session.send_queue.qsize() == 10

        # Enqueue 11th frame message: should drop oldest frame (frame 0) and remain at size 10
        msg_11 = WebSocketMessage(
            type=ServerMessageType.FRAME_RESULT,
            session_id="test_bp",
            data=FrameResultData(
                frame_number=11,
                timestamp_ms=1100.0,
                inference_time_ms=10.0,
                tracking_time_ms=5.0,
                total_time_ms=15.0,
                fps=30.0,
                active_tracks=0,
                unique_tracks=0,
                detections=[],
                tracks=[],
                analytics={},
            ),
        )
        await session.enqueue_message(msg_11)
        assert session.send_queue.qsize() == 10

        # Cleanup
        await session.close()

    asyncio.run(_run())
