"""
Step 12: Comprehensive End-to-End Integration Test Suite
Verifies the complete integration of:
1. System Health & Diagnostics
2. Runtime Configuration & Sources Probing
3. Model Registry, Validation & Switching
4. Static Image YOLO Detection Pipeline
5. Video ByteTrack Multi-Object Tracking
6. Real-time WebSocket Streaming Lifecycle (start -> frame_result -> pause -> resume -> stop)
7. Spatial Analytics & Tripwire Reporting
8. Frontend Build Artifacts & Proxy Configuration
"""

import base64
import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.main import create_application

SAMPLE_BUS_IMAGE = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
SAMPLE_REAL_BUS_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_real_bus.mp4"
SAMPLE_SYNTH_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_synthetic_shapes.mp4"


@pytest.fixture(scope="module")
def app_instance():
    return create_application()


@pytest.fixture(scope="module")
def test_client(app_instance):
    with TestClient(app_instance) as c:
        yield c


# ============================================================================
# 1. System Health & Hardware Diagnostics E2E
# ============================================================================

def test_system_status_and_health_integration(test_client):
    """Verifies system health, status, device selection, and model pre-warm."""
    # Root /health
    res_health = test_client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "healthy"
    assert health_data["api_status"] == "operational"

    # Root /status
    res_status = test_client.get("/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["status"] == "healthy"
    assert status_data["effective_device"] in ["cuda:0", "cpu"]
    assert status_data["model"]["loaded"] is True
    assert status_data["model"]["classes_count"] == 80
    assert status_data["uptime_seconds"] >= 0


# ============================================================================
# 2. Configuration & Input Sources E2E
# ============================================================================

def test_configuration_and_sources_integration(test_client):
    """Verifies runtime config with ByteTrack hyperparameters and camera probe."""
    res_cfg = test_client.get("/api/v1/config")
    assert res_cfg.status_code == 200
    cfg = res_cfg.json()
    assert cfg["model_name"] == "yolov8n.pt"
    assert cfg["active_model_id"] == "general_pretrained"
    assert cfg["tracker_type"] == "bytetrack.yaml"
    assert cfg["track_high_thresh"] == 0.5
    assert cfg["track_low_thresh"] == 0.1
    assert cfg["track_match_thresh"] == 0.8
    assert cfg["track_persistence_buffer"] == 30
    assert ".jpg" in cfg["supported_image_formats"]
    assert ".mp4" in cfg["supported_video_formats"]

    # Sources & Camera probe
    res_sources = test_client.get("/api/v1/sources")
    assert res_sources.status_code == 200
    sources = res_sources.json()
    assert "webcam" in sources["supported_sources"]

    res_probe = test_client.get("/api/v1/sources/webcam/probe?camera_index=0")
    assert res_probe.status_code == 200
    assert "available" in res_probe.json()


# ============================================================================
# 3. Model Registry & Switching Lifecycle E2E
# ============================================================================

def test_model_registry_and_switching_integration(test_client):
    """Verifies model catalog listing, forward-pass validation, and switching."""
    # List models
    res_list = test_client.get("/api/v1/models/")
    assert res_list.status_code == 200
    models_data = res_list.json()
    assert len(models_data["models"]) >= 1
    model_ids = [m["model_id"] for m in models_data["models"]]
    assert "general_pretrained" in model_ids

    # Validate weights
    res_val = test_client.post("/api/v1/models/general_pretrained/validate")
    assert res_val.status_code == 200
    val_data = res_val.json()
    assert val_data["available"] is True
    assert val_data["loadable"] is True
    assert val_data["num_classes"] == 80

    # Switch model
    res_sw = test_client.post("/api/v1/models/general_pretrained/switch")
    assert res_sw.status_code == 200
    assert res_sw.json()["active_model_id"] == "general_pretrained"


# ============================================================================
# 4. Real Static Image Detection Pipeline E2E
# ============================================================================

def test_real_image_detection_pipeline_integration(test_client):
    """Executes full detection inference on real bus.jpg image."""
    assert SAMPLE_BUS_IMAGE.is_file(), f"Sample image missing: {SAMPLE_BUS_IMAGE}"
    
    with open(SAMPLE_BUS_IMAGE, "rb") as f:
        files = {"file": ("bus.jpg", f, "image/jpeg")}
        res = test_client.post(
            "/api/v1/detect/image?confidence_threshold=0.35&iou_threshold=0.45&return_annotated=true",
            files=files,
        )

    assert res.status_code == 200
    det_data = res.json()
    assert det_data["total_detections"] > 0
    assert det_data["image_width"] > 0
    assert det_data["image_height"] > 0
    assert det_data["inference_time_ms"] > 0

    # Verify detected labels include bus or person
    detected_classes = [d["class_name"] for d in det_data["detections"]]
    assert "bus" in detected_classes or "person" in detected_classes

    # Verify bounding boxes are valid coordinates
    for det in det_data["detections"]:
        box = det["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        assert 0 <= x1 < x2 <= det_data["image_width"]
        assert 0 <= y1 < y2 <= det_data["image_height"]
        assert det["confidence"] >= 0.35


# ============================================================================
# 5. Video ByteTrack Multi-Object Tracking Pipeline E2E
# ============================================================================

def test_real_video_tracking_analytics_pipeline_integration(test_client):
    """Executes full multi-object tracking on video file."""
    assert SAMPLE_SYNTH_VIDEO.is_file(), f"Sample video missing: {SAMPLE_SYNTH_VIDEO}"

    with open(SAMPLE_SYNTH_VIDEO, "rb") as f:
        files = {"file": ("sample_synthetic_shapes.mp4", f, "video/mp4")}
        res = test_client.post(
            "/api/v1/track/video",
            files=files,
            data={"max_frames": 10, "confidence": 0.25},
        )

    assert res.status_code == 200
    track_data = res.json()
    assert track_data["frames_processed"] > 0
    assert "cumulative_unique_tracks" in track_data


# ============================================================================
# 6. Real-time WebSocket Streaming Full Lifecycle E2E
# ============================================================================

def test_websocket_live_stream_full_lifecycle_integration(test_client):
    """
    Validates end-to-end WebSocket protocol:
    Connect -> connection_ack -> start -> frame_results with ByteTrack tracks -> pause -> resume -> stop
    """
    assert SAMPLE_REAL_BUS_VIDEO.is_file(), f"Video missing: {SAMPLE_REAL_BUS_VIDEO}"

    with test_client.websocket_connect("/ws/stream") as ws:
        # 1. Verify connection ACK
        ack = ws.receive_json()
        assert ack["type"] == "connection_ack"
        assert "session_id" in ack["data"]

        # 2. Start streaming video with ByteTrack tracking
        start_payload = {
            "action": "start",
            "source": "video",
            "path": "data/samples/sample_real_bus.mp4",
            "fps_limit": 25.0,
            "conf": 0.25,
            "annotate": True,
        }
        ws.send_json(start_payload)

        # 3. Verify stream_started
        started_msg = ws.receive_json()
        assert started_msg["type"] == "stream_started"

        # 4. Receive and validate live frame results
        frames_verified = 0
        total_tracks_observed = 0

        for _ in range(8):
            msg = ws.receive_json()
            if msg["type"] == "frame_result":
                fdata = msg["data"]
                assert fdata["frame_number"] >= 1
                assert fdata["fps"] >= 0
                assert fdata["inference_time_ms"] > 0
                assert "detections" in fdata
                assert "tracks" in fdata

                # Verify tracks schema
                for trk in fdata["tracks"]:
                    assert "track_id" in trk
                    assert "class_name" in trk
                    box = trk.get("box", trk)
                    assert box["x1"] < box["x2"]
                    assert box["y1"] < box["y2"]
                    total_tracks_observed += 1

                # Verify annotated image base64 decode
                assert "annotated_frame" in fdata
                img_data = base64.b64decode(fdata["annotated_frame"])
                assert len(img_data) > 1000

                # Verify analytics payload
                assert "analytics" in fdata
                frames_verified += 1

        assert frames_verified >= 4, f"Expected at least 4 frames, got {frames_verified}"

        # 5. Test Pause & Resume Stream
        ws.send_json({"action": "pause"})
        ws.send_json({"action": "resume"})

        # 6. Test Stop Stream
        ws.send_json({"action": "stop"})
        stopped_msg = ws.receive_json()
        while stopped_msg["type"] == "frame_result":
            stopped_msg = ws.receive_json()
        assert stopped_msg["type"] == "stream_stopped"


# ============================================================================
# 7. Spatial Analytics & Reporting E2E
# ============================================================================

def test_analytics_reporting_and_reset_integration(test_client):
    """Verifies analytics telemetry report fetching and session resetting."""
    # Report endpoint
    res_rep = test_client.get("/api/v1/analytics/report")
    assert res_rep.status_code == 200
    rep_data = res_rep.json()
    assert "overall_summary" in rep_data
    assert "class_statistics" in rep_data
    assert "line_crossing" in rep_data

    # Reset endpoint
    res_reset = test_client.post("/api/v1/analytics/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "success"


# ============================================================================
# 8. Frontend Production Bundle & Proxy Verification E2E
# ============================================================================

def test_frontend_production_build_and_routing_integration():
    """Verifies that the React production bundle exists and Vite proxy is configured."""
    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    index_html = frontend_dist / "index.html"
    assert index_html.is_file(), "Production build frontend/dist/index.html does not exist"

    html_content = index_html.read_text(encoding="utf-8")
    assert '<div id="root"></div>' in html_content
    assert "assets/" in html_content

    # Verify Vite config proxies /api and /ws
    vite_config = PROJECT_ROOT / "frontend" / "vite.config.js"
    assert vite_config.is_file()
    cfg_text = vite_config.read_text(encoding="utf-8")
    assert "'/api'" in cfg_text or '"/api"' in cfg_text
    assert "'/ws'" in cfg_text or '"/ws"' in cfg_text
    assert "8000" in cfg_text
