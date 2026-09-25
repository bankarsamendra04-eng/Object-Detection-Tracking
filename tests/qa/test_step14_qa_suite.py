"""
Step 14: Comprehensive QA, Testing & Quality Assurance Test Suite
Validates the entire application across Test Pyramid Levels 1 to 8:
- Level 1: Unit edge cases (Detector, Tracker, Analytics, Input sources)
- Level 2: Integration pipeline
- Level 3: REST API boundary values and structured error handling
- Level 4: WebSocket protocol robustness, malformed inputs & reconnects
- Level 7: Failure injection and edge cases
- Level 8 / Concurrency: Multi-threaded REST requests & concurrent WebSocket sessions
"""

import concurrent.futures
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import List

import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.main import create_application
from backend.app.services.vision.detector import (
    DetectionEngine,
    Detection,
    DetectionResult,
    InvalidImageError,
    ModelLoadError,
    get_detector,
)
from backend.app.services.vision.tracker import (
    ByteTrackTracker,
    TrackedObject,
    TrackingResult,
    TrackerConfigError,
)
from backend.app.services.analytics.counter import (
    AnalyticsEngine,
    InvalidLineError,
    InvalidROIError,
)
from backend.app.services.vision.input_sources import (
    ImageInput,
    VideoInput,
    WebcamInput,
    FileNotFoundMediaError,
    UnsupportedFormatError,
    CorruptMediaError,
    probe_camera_availability,
)
from backend.app.services.streaming.manager import get_stream_manager

SAMPLE_BUS_IMAGE = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
SAMPLE_REAL_BUS_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_real_bus.mp4"
SAMPLE_SYNTH_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_synthetic_shapes.mp4"


@pytest.fixture(scope="module")
def app():
    return create_application()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


# ============================================================================
# LEVEL 1: BACKEND UNIT TESTS & EDGE CASES
# ============================================================================

class TestDetectionEngineQA:
    """QA edge cases for YOLO DetectionEngine."""

    def test_detector_model_reuse_identity(self):
        """Inference should reuse the loaded model instance without reloading weights."""
        detector = get_detector()
        model_ref_1 = id(detector.model)
        
        # Perform inference on bus.jpg
        _ = detector.detect(SAMPLE_BUS_IMAGE)
        model_ref_2 = id(detector.model)
        assert model_ref_1 == model_ref_2, "DetectionEngine must reuse the same model instance across inferences."

    def test_detector_1x1_image_input(self):
        """1x1 image is valid geometry; should yield zero detections cleanly without crashing."""
        detector = get_detector()
        tiny_img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = detector.detect(tiny_img)
        assert isinstance(result, DetectionResult)
        assert result.total_detections == 0
        assert result.image_width == 1
        assert result.image_height == 1

    def test_detector_grayscale_2d_array_input(self):
        """2D grayscale array should be handled cleanly by the detector."""
        detector = get_detector()
        gray_img = np.zeros((120, 160), dtype=np.uint8)
        result = detector.detect(gray_img)
        assert isinstance(result, DetectionResult)
        assert result.image_width == 160
        assert result.image_height == 120

    def test_detector_nonexistent_class_filter(self):
        """Filtering for a class not present in taxonomy yields 0 detections without error."""
        detector = get_detector()
        result = detector.detect(SAMPLE_BUS_IMAGE, target_classes=["non_existent_alien_craft"])
        assert result.total_detections == 0
        assert len(result.detections) == 0

    def test_detector_empty_and_none_rejection(self):
        """Empty NumPy arrays and None must raise InvalidImageError."""
        detector = get_detector()
        with pytest.raises(InvalidImageError):
            detector.detect(None)

        with pytest.raises(InvalidImageError):
            detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_detector_detection_bounds_and_validity(self):
        """Real detections on bus.jpg must have coordinates strictly inside image boundaries."""
        detector = get_detector()
        result = detector.detect(SAMPLE_BUS_IMAGE, conf=0.25)
        assert result.total_detections > 0
        for det in result.detections:
            assert 0.0 <= det.x1 < det.x2 <= result.image_width
            assert 0.0 <= det.y1 < det.y2 <= result.image_height
            assert 0.0 <= det.confidence <= 1.0
            assert det.area > 0.0
            assert det.class_name in detector.available_classes


class TestTrackingEngineQA:
    """QA edge cases for ByteTrackTracker."""

    def test_tracker_empty_detections_lifecycle(self):
        """Passing empty detections to tracker updates frame count and produces 0 active tracks."""
        tracker = ByteTrackTracker()
        empty_res = DetectionResult(
            detections=[],
            image_width=640,
            image_height=480,
            inference_time_ms=5.0,
            device_used="cpu",
            model_name="yolov8n.pt",
        )
        res = tracker.update(empty_res)
        assert isinstance(res, TrackingResult)
        assert res.active_track_count == 0
        assert res.frame_number == 1

    def test_tracker_invalid_thresholds_rejection(self):
        """Tracker must reject track_low_thresh >= track_high_thresh."""
        with pytest.raises(TrackerConfigError):
            ByteTrackTracker(track_high_thresh=0.4, track_low_thresh=0.6)

    def test_tracker_sequential_persistence_and_reset(self):
        """Tracker maintains ID on moving object and clears cleanly on reset."""
        tracker = ByteTrackTracker()
        
        # Frame 1: Object at (100, 100, 150, 150)
        det1 = Detection(class_id=0, class_name="person", confidence=0.9, x1=100, y1=100, x2=150, y2=150)
        r1 = tracker.update(DetectionResult(
            detections=[det1],
            image_width=640,
            image_height=480,
            inference_time_ms=5.0,
            device_used="cpu",
            model_name="yolov8n.pt",
        ))
        assert r1.active_track_count == 1
        t_id = r1.objects[0].track_id
        assert t_id >= 1

        # Frame 2: Object moved slightly to (105, 102, 155, 152)
        det2 = Detection(class_id=0, class_name="person", confidence=0.88, x1=105, y1=102, x2=155, y2=152)
        r2 = tracker.update(DetectionResult(
            detections=[det2],
            image_width=640,
            image_height=480,
            inference_time_ms=5.0,
            device_used="cpu",
            model_name="yolov8n.pt",
        ))
        assert r2.active_track_count == 1
        assert r2.objects[0].track_id == t_id, "Track ID must remain persistent across consecutive frames."

        # Tracker reset
        tracker.reset()
        assert tracker.frame_count == 0
        assert len(tracker.tracked_tracks) == 0


class TestAnalyticsEngineQA:
    """QA edge cases for AnalyticsEngine."""

    def test_analytics_empty_input_snapshot(self):
        """Analytics engine handles empty tracking results gracefully."""
        engine = AnalyticsEngine(session_id="qa_test_session")
        empty_tr = TrackingResult(
            frame_number=1,
            timestamp="2026-09-24T00:00:00Z",
            objects=[],
            active_track_count=0,
            processing_time_ms=1.0,
            cumulative_unique_tracks=0,
        )
        snapshot = engine.update(empty_tr)
        assert snapshot.active_objects == 0
        assert snapshot.total_unique_objects == 0
        assert snapshot.class_distribution == {}

    def test_analytics_unique_count_monotonicity(self):
        """Total unique objects count must be monotonically non-decreasing."""
        engine = AnalyticsEngine(session_id="qa_test_session")
        
        # Feed 3 different tracks across frames
        t1 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.9, x1=10, y1=10, x2=30, y2=30)
        s1 = engine.update(TrackingResult(
            frame_number=1,
            timestamp="2026-09-24T00:00:00Z",
            objects=[t1],
            active_track_count=1,
            processing_time_ms=1.0,
            cumulative_unique_tracks=1,
        ))
        assert s1.total_unique_objects == 1

        t2 = TrackedObject(track_id=2, class_id=2, class_name="car", confidence=0.85, x1=40, y1=40, x2=80, y2=80)
        s2 = engine.update(TrackingResult(
            frame_number=2,
            timestamp="2026-09-24T00:00:01Z",
            objects=[t1, t2],
            active_track_count=2,
            processing_time_ms=1.0,
            cumulative_unique_tracks=2,
        ))
        assert s2.total_unique_objects == 2

        # In frame 3, t1 leaves, t2 remains
        s3 = engine.update(TrackingResult(
            frame_number=3,
            timestamp="2026-09-24T00:00:02Z",
            objects=[t2],
            active_track_count=1,
            processing_time_ms=1.0,
            cumulative_unique_tracks=2,
        ))
        assert s3.active_objects == 1
        assert s3.total_unique_objects == 2, "Cumulative unique objects must never decrease."


class TestInputSourcesQA:
    """QA edge cases for media input sources."""

    def test_image_input_zero_byte_file(self):
        """0-byte file must raise CorruptMediaError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = Path(f.name)
        try:
            with pytest.raises(CorruptMediaError):
                ImageInput(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_image_input_unsupported_extension(self):
        """Unsupported file extensions must raise UnsupportedFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"fake binary payload")
            temp_path = Path(f.name)
        try:
            with pytest.raises(UnsupportedFormatError):
                ImageInput(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_image_input_directory_as_file(self):
        """Passing directory path must raise FileNotFoundMediaError."""
        with pytest.raises(FileNotFoundMediaError):
            ImageInput(PROJECT_ROOT / "data")

    def test_video_input_missing_file(self):
        """Missing video file must raise FileNotFoundMediaError."""
        with pytest.raises(FileNotFoundMediaError):
            VideoInput("data/samples/non_existent_video_12345.mp4")


# ============================================================================
# LEVEL 3: REST API BOUNDARY & VALIDATION TESTS
# ============================================================================

class TestRestApiQA:
    """QA boundary values and structured error handling for REST endpoints."""

    def test_image_detect_missing_file_parameter(self, client):
        """POST /api/v1/detect/image without file parameter returns HTTP 422."""
        res = client.post("/api/v1/detect/image")
        assert res.status_code == 422
        assert "detail" in res.json()

    def test_image_detect_boundary_thresholds(self, client):
        """Confidence and IoU thresholds out of [0.0, 1.0] return HTTP 422."""
        assert SAMPLE_BUS_IMAGE.is_file()
        with open(SAMPLE_BUS_IMAGE, "rb") as f:
            res_conf_high = client.post(
                "/api/v1/detect/image",
                files={"file": ("bus.jpg", f, "image/jpeg")},
                data={"confidence": 1.05},
            )
        assert res_conf_high.status_code == 422

        with open(SAMPLE_BUS_IMAGE, "rb") as f:
            res_conf_neg = client.post(
                "/api/v1/detect/image",
                files={"file": ("bus.jpg", f, "image/jpeg")},
                data={"confidence": -0.1},
            )
        assert res_conf_neg.status_code == 422

    def test_models_nonexistent_id_404(self, client):
        """GET /api/v1/models/{non_existent_id} returns HTTP 404 with structured detail."""
        res = client.get("/api/v1/models/non_existent_model_id_xyz")
        assert res.status_code == 404
        assert "not registered" in res.json()["detail"].lower()

    def test_models_validate_nonexistent_id(self, client):
        """POST /api/v1/models/{non_existent_id}/validate returns 200 with INVALID status."""
        res = client.post("/api/v1/models/non_existent_model_id_xyz/validate")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "INVALID"
        assert data["loadable"] is False

    def test_video_track_invalid_stride(self, client):
        """POST /api/v1/track/video with stride < 1 returns HTTP 422."""
        assert SAMPLE_SYNTH_VIDEO.is_file()
        with open(SAMPLE_SYNTH_VIDEO, "rb") as f:
            res = client.post(
                "/api/v1/track/video",
                files={"file": ("clip.mp4", f, "video/mp4")},
                data={"stride": 0},
            )
        assert res.status_code == 422

    def test_camera_probe_negative_index_422(self, client):
        """GET /api/v1/sources/webcam/probe with camera_index < 0 returns HTTP 422."""
        res = client.get("/api/v1/sources/webcam/probe?camera_index=-1")
        assert res.status_code == 422


# ============================================================================
# LEVEL 4: WEBSOCKET PROTOCOL & EDGE CASES
# ============================================================================

class TestWebSocketQA:
    """QA edge cases and robustness for WebSocket streaming."""

    def test_ws_empty_message_handling(self, client):
        """Empty string payload over WebSocket returns INVALID_JSON without crashing."""
        with client.websocket_connect("/ws/stream") as ws:
            _ = ws.receive_json()  # ack
            ws.send_text("")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["data"]["code"] == "INVALID_JSON"

    def test_ws_rapid_connect_stop_disconnect(self, client):
        """Connecting, pinging, and immediately disconnecting cleans up cleanly."""
        mgr = get_stream_manager()
        initial_count = mgr.active_connections_count

        with client.websocket_connect("/ws/stream") as ws:
            ack = ws.receive_json()
            assert ack["type"] == "connection_ack"
            assert mgr.active_connections_count == initial_count + 1

            # Send ping
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"

        time.sleep(0.05)
        assert mgr.active_connections_count == initial_count

    def test_ws_reconnect_lifecycle(self, client):
        """Successive reconnections obtain distinct session IDs and clean up properly."""
        mgr = get_stream_manager()
        session_ids = []

        for _ in range(3):
            with client.websocket_connect("/ws/stream") as ws:
                ack = ws.receive_json()
                assert ack["type"] == "connection_ack"
                s_id = ack["data"]["session_id"]
                session_ids.append(s_id)
            time.sleep(0.02)

        # All session IDs must be unique
        assert len(set(session_ids)) == 3
        assert mgr.active_connections_count == 0


# ============================================================================
# LEVEL 7: FAILURE INJECTION & SAFE ERROR FORMATTING
# ============================================================================

class TestFailureHandlingQA:
    """Verifies that system errors return structured JSON and never leak stack traces."""

    def test_corrupt_image_upload_returns_structured_400(self, client):
        """Uploading corrupted image bytes returns HTTP 400 with clean detail."""
        corrupt_bytes = b"FF D8 FF E0 00 00 NOT_A_REAL_JPEG_DATA_STREAM"
        files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
        res = client.post("/api/v1/detect/image", files=files)
        assert res.status_code == 400
        data = res.json()
        assert "detail" in data
        assert "Traceback" not in res.text

    def test_unavailable_custom_model_switch_returns_structured_400(self, client):
        """Attempting to switch to un-downloaded custom model returns HTTP 400 without crashing."""
        res = client.post("/api/v1/models/custom_visdrone/switch")
        assert res.status_code == 400
        data = res.json()
        assert "detail" in data
        assert "Traceback" not in res.text


# ============================================================================
# LEVEL 8 & CONCURRENCY: MULTI-CLIENT CONCURRENCY TESTS
# ============================================================================

class TestConcurrencyQA:
    """Verifies concurrent REST requests and concurrent WebSocket streaming sessions."""

    def test_concurrent_rest_detection_requests(self, client):
        """Multiple concurrent REST detection requests execute safely without state contamination."""
        assert SAMPLE_BUS_IMAGE.is_file()
        with open(SAMPLE_BUS_IMAGE, "rb") as f:
            image_bytes = f.read()

        def send_request():
            files = {"file": ("bus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            res = client.post("/api/v1/detect/image", files=files, data={"confidence": 0.25})
            return res.status_code, res.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(send_request) for _ in range(4)]
            results = [f.result() for f in futures]

        for status_code, body in results:
            assert status_code == 200
            assert body["total_detections"] > 0
            assert "bus" in [d["class_name"] for d in body["detections"]]

    def test_concurrent_websocket_streaming_sessions(self, client):
        """Two concurrent WebSocket streaming clients process frames simultaneously in isolation."""
        mgr = get_stream_manager()
        assert mgr.active_connections_count == 0

        with client.websocket_connect("/ws/stream") as ws1:
            ack1 = ws1.receive_json()
            session_1 = ack1["data"]["session_id"]

            with client.websocket_connect("/ws/stream") as ws2:
                ack2 = ws2.receive_json()
                session_2 = ack2["data"]["session_id"]
                assert session_1 != session_2
                assert mgr.active_connections_count == 2

                # Start streaming on client 1
                ws1.send_json({
                    "action": "start",
                    "source": "video",
                    "path": "data/samples/sample_real_bus.mp4",
                    "stride": 1,
                    "fps_limit": 30.0,
                    "max_frames": 10,
                })
                started1 = ws1.receive_json()
                assert started1["type"] == "stream_started"

                # Start streaming on client 2
                ws2.send_json({
                    "action": "start",
                    "source": "video",
                    "path": "data/samples/sample_real_bus.mp4",
                    "stride": 1,
                    "fps_limit": 30.0,
                    "max_frames": 10,
                })
                started2 = ws2.receive_json()
                assert started2["type"] == "stream_started"

                # Verify both receive frame_result messages
                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()
                assert msg1["type"] == "frame_result"
                assert msg2["type"] == "frame_result"
                assert msg1["session_id"] == session_1
                assert msg2["session_id"] == session_2

                # Stop both streams
                ws1.send_json({"action": "stop"})
                ws2.send_json({"action": "stop"})
                _ = ws1.receive_json()  # stopped
                _ = ws2.receive_json()  # stopped

        time.sleep(0.05)
        assert mgr.active_connections_count == 0
