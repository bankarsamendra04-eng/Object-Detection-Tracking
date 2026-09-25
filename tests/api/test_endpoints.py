import io
from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.app.core.config import PROJECT_ROOT

SAMPLE_BUS_IMAGE = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
SAMPLE_SYNTH_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_synthetic_shapes.mp4"


# ==========================================
# 1. Health & Status Endpoints
# ==========================================

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "api_v1" in data
    assert "docs" in data


def test_health_endpoints(client):
    # Root /health
    res_root = client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "healthy"
    assert res_root.json()["api_status"] == "operational"

    # API v1 /api/v1/health
    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "healthy"


def test_status_endpoints(client):
    # Root /status
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "effective_device" in data
    assert "uptime_seconds" in data
    assert "python_version" in data
    assert data["model"]["name"] == "yolov8n.pt"
    assert data["model"]["classes_count"] == 80

    # API v1 /api/v1/status
    res_v1 = client.get("/api/v1/status")
    assert res_v1.status_code == 200
    assert res_v1.json()["app_name"] is not None


# ==========================================
# 2. Configuration & Sources Endpoints
# ==========================================

def test_config_endpoint_no_secrets(client):
    res = client.get("/api/v1/config")
    assert res.status_code == 200
    data = res.json()
    assert "model_name" in data
    assert "confidence_threshold" in data
    assert "device" in data
    assert "supported_image_formats" in data
    assert "supported_video_formats" in data

    # Security check: Ensure internal secrets, keys, or raw db passwords are never leaked
    raw_str = res.text.lower()
    assert "secret" not in raw_str
    assert "password" not in raw_str
    assert "token" not in raw_str


def test_sources_endpoint(client):
    res = client.get("/api/v1/sources")
    assert res.status_code == 200
    data = res.json()
    assert "image" in data["supported_sources"]
    assert "video" in data["supported_sources"]
    assert "webcam" in data["supported_sources"]
    assert "webcam_config" in data


def test_camera_probe_endpoint(client):
    res = client.get("/api/v1/sources/webcam/probe?camera_index=0")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    assert data["camera_index"] == 0


# ==========================================
# 3. Detection API Endpoints
# ==========================================

def test_image_detection_endpoint_valid_image(client):
    assert SAMPLE_BUS_IMAGE.is_file()
    with open(SAMPLE_BUS_IMAGE, "rb") as f:
        files = {"file": ("bus.jpg", f, "image/jpeg")}
        res = client.post("/api/v1/detect/image", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["image_width"] > 0
    assert data["image_height"] > 0
    assert data["total_detections"] > 0
    assert isinstance(data["detections"], list)
    assert "bus" in data["class_counts"] or "person" in data["class_counts"]


def test_image_detection_endpoint_alias(client):
    # Test /api/v1/detection/image alias
    assert SAMPLE_BUS_IMAGE.is_file()
    with open(SAMPLE_BUS_IMAGE, "rb") as f:
        files = {"file": ("bus.jpg", f, "image/jpeg")}
        res = client.post("/api/v1/detection/image", files=files)
    assert res.status_code == 200


def test_image_detection_validation_errors(client):
    # 1. Invalid confidence threshold (> 1.0) -> HTTP 422
    with open(SAMPLE_BUS_IMAGE, "rb") as f:
        files = {"file": ("bus.jpg", f, "image/jpeg")}
        res = client.post("/api/v1/detect/image", files=files, data={"confidence": 1.5})
    assert res.status_code == 422

    # 2. Unsupported file extension (.txt) -> HTTP 400
    files_txt = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}
    res_txt = client.post("/api/v1/detect/image", files=files_txt)
    assert res_txt.status_code == 400
    assert "Unsupported image extension" in res_txt.json()["detail"]

    # 3. Empty 0-byte file -> HTTP 400
    files_empty = {"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    res_empty = client.post("/api/v1/detect/image", files=files_empty)
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()


# ==========================================
# 4. Tracking API Endpoints
# ==========================================

def test_video_tracking_endpoint(client):
    assert SAMPLE_SYNTH_VIDEO.is_file()
    with open(SAMPLE_SYNTH_VIDEO, "rb") as vid_file:
        files = {"file": ("test_vid.mp4", vid_file, "video/mp4")}
        data = {"stride": "2", "max_frames": "6", "confidence": "0.3"}
        res = client.post("/api/v1/track/video", files=files, data=data)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "session_id" in data
    assert data["frames_processed"] > 0
    assert "video_metadata" in data


def test_video_tracking_validation_errors(client):
    # Unsupported video format -> HTTP 400
    files_bad = {"file": ("bad_video.doc", io.BytesIO(b"abc"), "application/msword")}
    res = client.post("/api/v1/track/video", files=files_bad)
    assert res.status_code == 400
    assert "Unsupported video extension" in res.json()["detail"]


def test_track_frame_endpoint(client, dummy_frame):
    success, buffer = cv2.imencode(".jpg", dummy_frame)
    assert success is True

    files = {"file": ("frame.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")}
    res = client.post("/api/v1/track/frame", files=files, data={"session_id": "test_frame_session"})
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test_frame_session"
    assert "tracks" in data


def test_track_reset_endpoint(client):
    res = client.post("/api/v1/track/reset?session_id=test_frame_session")
    assert res.status_code == 200
    assert res.json()["status"] == "success"


# ==========================================
# 5. Analytics API Endpoints
# ==========================================

def test_analytics_current_and_reset(client):
    res = client.get("/api/v1/analytics/current?session_id=api_test_session")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "api_test_session"
    assert "active_objects" in data
    assert "total_unique_objects" in data
    assert "performance" in data

    # Reset
    res_reset = client.post("/api/v1/analytics/reset?session_id=api_test_session")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "success"


def test_analytics_line_management(client):
    line_payload = {
        "line_id": "api_line_1",
        "name": "Front Gate",
        "start_point": [0.0, 200.0],
        "end_point": [640.0, 200.0],
        "direction": "BOTH",
    }
    # Add line
    res_add = client.post("/api/v1/analytics/line?session_id=api_test_session", json=line_payload)
    assert res_add.status_code == 200
    assert res_add.json()["status"] == "success"

    # Add invalid line (identical points) -> HTTP 400
    bad_line = {
        "line_id": "bad_line",
        "name": "Bad",
        "start_point": [100.0, 100.0],
        "end_point": [100.0, 100.0],
        "direction": "BOTH",
    }
    res_bad = client.post("/api/v1/analytics/line?session_id=api_test_session", json=bad_line)
    assert res_bad.status_code == 400

    # Remove line
    res_del = client.delete("/api/v1/analytics/line/api_line_1?session_id=api_test_session")
    assert res_del.status_code == 200


def test_analytics_roi_management(client):
    roi_payload = {
        "roi_id": "api_roi_1",
        "name": "Lobby",
        "points": [[50.0, 50.0], [250.0, 50.0], [250.0, 250.0], [50.0, 250.0]],
        "anchor": "bottom_center",
    }
    # Add ROI
    res_add = client.post("/api/v1/analytics/roi?session_id=api_test_session", json=roi_payload)
    assert res_add.status_code == 200
    assert res_add.json()["status"] == "success"

    # Add invalid ROI (< 3 points) -> HTTP 400
    bad_roi = {
        "roi_id": "bad_roi",
        "name": "Bad",
        "points": [[50.0, 50.0], [250.0, 50.0]],
        "anchor": "bottom_center",
    }
    res_bad = client.post("/api/v1/analytics/roi?session_id=api_test_session", json=bad_roi)
    assert res_bad.status_code == 400

    # Remove ROI
    res_del = client.delete("/api/v1/analytics/roi/api_roi_1?session_id=api_test_session")
    assert res_del.status_code == 200


def test_analytics_events_endpoint(client):
    res = client.get("/api/v1/analytics/events?session_id=api_test_session&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_analytics_historical_summary(client):
    res = client.get("/api/v1/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "database_records_count" in data
    assert "average_inference_time_ms" in data


# ==========================================
# 6. OpenAPI Documentation Availability
# ==========================================

def test_openapi_schema_and_docs_available(client):
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200

    res_redoc = client.get("/redoc")
    assert res_redoc.status_code == 200

    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    schema = res_openapi.json()
    assert "paths" in schema
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/detect/image" in schema["paths"]
    assert "/api/v1/track/video" in schema["paths"]
    assert "/api/v1/analytics/current" in schema["paths"]
    assert "/api/v1/config" in schema["paths"]
    assert "/api/v1/sources" in schema["paths"]
