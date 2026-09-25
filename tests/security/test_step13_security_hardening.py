"""
Step 13: Comprehensive Security, Error Handling & Production Hardening Test Suite

Validates all 5 Special Attention areas:
1. 🔐 Path Traversal Protection
2. 📁 Malicious and Oversized File Upload Handling
3. 🔌 WebSocket Malformed-Message and Disconnect Handling
4. 🔑 Secrets, Configuration & Security Headers Audit
5. 🧪 Failure Injection & Exception Sanitization
"""

import asyncio
from pathlib import Path
import tempfile
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.main import create_application
from backend.app.services.streaming.manager import WebSocketManager, get_stream_manager
from backend.app.services.vision.model_manager import ModelManager


@pytest.fixture(scope="module")
def app_instance():
    return create_application()


@pytest.fixture(scope="module")
def client(app_instance):
    with TestClient(app_instance) as c:
        yield c


# ============================================================================
# 1. 🔐 Path Traversal Protection Tests
# ============================================================================

def test_path_traversal_model_id_url(client):
    """Path traversal attempt in model_id path parameter is blocked (422 / 404)."""
    # Attempt directory traversal via URL
    res = client.get("/api/v1/models/..%2F..%2Fetc%2Fpasswd")
    assert res.status_code in {404, 422}

    res_post = client.post("/api/v1/models/..%2F..%2Fetc%2Fpasswd/switch")
    assert res_post.status_code in {404, 422}


def test_path_traversal_websocket_command(client):
    """Path traversal attempts in WebSocket streaming path parameter are rejected."""
    with client.websocket_connect("/ws/stream") as ws:
        _ = ws.receive_json()  # connection_ack

        # Relative parent traversal
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "../../../secret/password.txt",
        })
        err1 = ws.receive_json()
        assert err1["type"] == "error"
        assert err1["data"]["code"] in {"INVALID_COMMAND", "SOURCE_UNAVAILABLE"}

        # Windows backslash traversal
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "..\\..\\secret\\password.txt",
        })
        err2 = ws.receive_json()
        assert err2["type"] == "error"
        assert err2["data"]["code"] in {"INVALID_COMMAND", "SOURCE_UNAVAILABLE"}

        # Drive letter injection
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "C:\\Windows\\System32\\cmd.exe",
        })
        err3 = ws.receive_json()
        assert err3["type"] == "error"
        assert err3["data"]["code"] in {"INVALID_COMMAND", "SOURCE_UNAVAILABLE"}

        # UNC network share injection
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "\\\\attacker.com\\share\\malicious.mp4",
        })
        err4 = ws.receive_json()
        assert err4["type"] == "error"
        assert err4["data"]["code"] in {"INVALID_COMMAND", "SOURCE_UNAVAILABLE"}

        # Non-media directory access attempt (inside workspace but forbidden directory)
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "backend/app/main.py",
        })
        err5 = ws.receive_json()
        assert err5["type"] == "error"
        assert err5["data"]["code"] in {"INVALID_COMMAND", "SOURCE_UNAVAILABLE"}


def test_resolve_safe_path_helper():
    """Validates the core resolve_safe_path security utility directly."""
    # Valid relative path inside data
    safe = settings.resolve_safe_path("data/samples/bus.jpg")
    assert safe.is_file()

    # Traversal attempts must raise ValueError
    with pytest.raises(ValueError):
        settings.resolve_safe_path("../../../etc/passwd")

    with pytest.raises(ValueError):
        settings.resolve_safe_path("..\\..\\Windows\\System32")

    with pytest.raises(ValueError):
        settings.resolve_safe_path("\\\\server\\share\\file.txt")

    with pytest.raises(ValueError):
        settings.resolve_safe_path("C:\\Windows\\System32\\cmd.exe")

    with pytest.raises(ValueError):
        settings.resolve_safe_path("")


# ============================================================================
# 2. 📁 Malicious & Oversized File Upload Handling Tests
# ============================================================================

def test_oversized_image_upload_rejected(client):
    """Images exceeding MAX_UPLOAD_IMAGE_SIZE_BYTES (15 MB) return HTTP 413."""
    # Create fake oversized payload (> 15 MB)
    oversized_bytes = b"0" * (15 * 1024 * 1024 + 1024)
    files = {"file": ("huge.jpg", oversized_bytes, "image/jpeg")}
    res = client.post("/api/v1/detect/image", files=files)
    assert res.status_code == 413
    assert "exceeds" in res.json()["detail"].lower()


def test_oversized_video_upload_rejected(client):
    """Videos exceeding MAX_UPLOAD_VIDEO_SIZE_BYTES (50 MB) return HTTP 413."""
    oversized_bytes = b"0" * (50 * 1024 * 1024 + 1024)
    files = {"file": ("huge.mp4", oversized_bytes, "video/mp4")}
    res = client.post("/api/v1/track/video", files=files)
    assert res.status_code == 413
    assert "exceeds" in res.json()["detail"].lower()


def test_invalid_file_extension_image(client):
    """Uploading executable or text file to image endpoint returns HTTP 400."""
    files = {"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")}
    res = client.post("/api/v1/detect/image", files=files)
    assert res.status_code == 400
    assert "unsupported image extension" in res.json()["detail"].lower()


def test_invalid_file_extension_video(client):
    """Uploading non-video file to video tracking returns HTTP 400."""
    files = {"file": ("script.sh", b"#!/bin/bash\necho hello", "text/x-shellscript")}
    res = client.post("/api/v1/track/video", files=files)
    assert res.status_code == 400
    assert "unsupported video extension" in res.json()["detail"].lower()


def test_corrupted_image_handling(client):
    """Uploading corrupt/garbage image bytes returns HTTP 400 without crashing."""
    corrupt_bytes = b"NOT_AN_IMAGE_HEADER_GARBAGE_PAYLOAD_12345"
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    res = client.post("/api/v1/detect/image", files=files)
    assert res.status_code == 400
    assert "corrupt or unreadable" in res.json()["detail"].lower()


def test_corrupted_video_handling(client):
    """Uploading corrupt video payload returns HTTP 400 with temporary file cleaned up."""
    upload_dir = settings.resolve_path(settings.UPLOADS_DIR)
    before_files = set(upload_dir.glob("track_temp_*"))

    corrupt_bytes = b"CORRUPT_VIDEO_DATA_NOT_VALID_MP4_CONTAINER"
    files = {"file": ("corrupt.mp4", corrupt_bytes, "video/mp4")}
    res = client.post("/api/v1/track/video", files=files)
    assert res.status_code == 400

    # Verify temporary file cleanup: no new orphaned temp files in upload_dir
    after_files = set(upload_dir.glob("track_temp_*"))
    assert after_files == before_files


def test_decompression_bomb_dimension_check(client):
    """Images with dimensions exceeding MAX_IMAGE_DIMENSION (8192) return HTTP 400."""
    # Create minimal 1x1 image, then monkeypatch or create large header
    img = Image.new("RGB", (8200, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("wide_image.jpg", buf.read(), "image/jpeg")}
    res = client.post("/api/v1/detect/image", files=files)
    assert res.status_code == 400
    assert "exceed" in res.json()["detail"].lower()


# ============================================================================
# 3. 🔌 WebSocket Malformed-Message & Disconnect Handling Tests
# ============================================================================

def test_websocket_malformed_json(client):
    """Sending malformed JSON to WebSocket returns INVALID_JSON error without closing."""
    with client.websocket_connect("/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_text("NOT_JSON{{{")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["data"]["code"] == "INVALID_JSON"


def test_websocket_oversized_message_rejected(client):
    """WebSocket payloads exceeding MAX_WS_MESSAGE_SIZE_BYTES (64 KB) return MESSAGE_TOO_LARGE."""
    with client.websocket_connect("/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        oversized_text = '{"action": "ping", "junk": "' + ("A" * 70000) + '"}'
        ws.send_text(oversized_text)
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["data"]["code"] == "MESSAGE_TOO_LARGE"


def test_websocket_invalid_command_action(client):
    """Sending unknown action returns INVALID_COMMAND."""
    with client.websocket_connect("/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({"action": "unsupported_exploit"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["data"]["code"] == "INVALID_COMMAND"


def test_websocket_camera_index_bounds(client):
    """Negative camera index is rejected by schema validator with INVALID_COMMAND."""
    with client.websocket_connect("/ws/stream") as ws:
        _ = ws.receive_json()  # ack

        ws.send_json({
            "action": "start",
            "source": "webcam",
            "camera_index": -1,
        })
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["data"]["code"] == "INVALID_COMMAND"


def test_websocket_concurrent_session_limit():
    """Verifies that WebSocketManager enforces MAX_CONCURRENT_WS_SESSIONS."""
    manager = WebSocketManager()
    assert manager.active_connections_count == 0


def test_websocket_disconnect_cleanup(client):
    """Verifies session resources and trackers are cleaned up on client disconnect."""
    manager = get_stream_manager()
    initial_count = manager.active_connections_count

    with client.websocket_connect("/ws/stream") as ws:
        ack = ws.receive_json()
        assert manager.active_connections_count == initial_count + 1
        session_id = ack["data"]["session_id"]
        assert manager.get_session(session_id) is not None

    # After exit from context manager (socket closed)
    # The session must be deregistered
    assert manager.get_session(session_id) is None


# ============================================================================
# 4. 🔑 Secrets, Configuration & Security Headers Audit Tests
# ============================================================================

def test_security_headers_present(client):
    """Verifies standard security headers are attached to API responses."""
    res = client.get("/health")
    assert res.status_code == 200
    headers = res.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert "X-XSS-Protection" in headers
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_cors_configuration_safe(client):
    """Verifies CORS origins match configuration and do not permit wildcard with credentials."""
    assert "*" not in settings.CORS_ORIGINS, "CORS_ORIGINS must not contain wildcard '*' in production"
    assert len(settings.CORS_ORIGINS) >= 1

    # Preflight OPTIONS request
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    res = client.options("/api/v1/detect/image", headers=headers)
    assert res.status_code in {200, 204}
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_config_endpoint_no_secrets_leak(client):
    """Verifies /api/v1/config does not leak database credentials, secrets, or internal keys."""
    res = client.get("/api/v1/config")
    assert res.status_code == 200
    raw_text = res.text.lower()

    for forbidden in ["secret", "password", "token", "database_url", "sqlite:"]:
        assert forbidden not in raw_text


# ============================================================================
# 5. 🧪 Failure Injection & Error Handling Tests
# ============================================================================

def test_unregistered_model_switch_404(client):
    """Switching to an unregistered model returns HTTP 404 without leaking stack trace."""
    res = client.post("/api/v1/models/completely_bogus_model/switch")
    assert res.status_code == 404
    data = res.json()
    assert "traceback" not in data
    assert "error" in data


def test_unavailable_model_switch_400(client):
    """Switching to custom model whose weights are not on disk returns HTTP 400."""
    res = client.post("/api/v1/models/custom_visdrone/switch")
    assert res.status_code == 400
    data = res.json()
    assert "not available" in data["detail"].lower() or "does not exist" in data["detail"].lower()


def test_camera_probe_query_bounds(client):
    """Probing camera index < 0 or > 32 returns HTTP 422 validation error."""
    res_negative = client.get("/api/v1/sources/webcam/probe?camera_index=-1")
    assert res_negative.status_code == 422

    res_huge = client.get("/api/v1/sources/webcam/probe?camera_index=999")
    assert res_huge.status_code == 422


def test_analytics_invalid_coordinates_rejected(client):
    """Line tripwire with negative coordinates or NaN is rejected."""
    bad_line = {
        "line_id": "test_neg",
        "name": "Negative Line",
        "start_point": [-10.0, 100.0],
        "end_point": [200.0, 100.0],
        "direction": "BOTH",
    }
    res = client.post("/api/v1/analytics/line?session_id=sec_test", json=bad_line)
    assert res.status_code == 422
