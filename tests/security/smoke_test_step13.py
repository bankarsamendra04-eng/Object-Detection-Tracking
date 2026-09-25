"""
Step 13: Dedicated Real Application Smoke Test
Executes real end-to-end smoke test on the live pipeline:
1. Real image detection on bus.jpg
2. Real video tracking on sample_real_bus.mp4
3. Real WebSocket streaming with ByteTrack and analytics on sample_real_bus.mp4
4. WebSocket disconnect and session cleanup verification
5. Malicious request rejection (path traversal, oversized upload, malformed command)
6. Security headers check on HTTP responses
7. Secrets leak check
"""

import sys
import io
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient
from backend.app.core.config import PROJECT_ROOT
from backend.app.main import create_application
from backend.app.services.streaming.manager import get_stream_manager

SAMPLE_BUS_IMAGE = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
SAMPLE_REAL_BUS_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_real_bus.mp4"

def run_smoke_test():
    print("=" * 60)
    print("RUNNING STEP 13 REAL APPLICATION SMOKE TEST")
    print("=" * 60)

    app = create_application()
    
    with TestClient(app) as client:
        # 1. Security Headers Verification
        print("\n[1/7] Testing Security Headers on HTTP Responses...")
        res = client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.status_code}"
        assert res.headers.get("x-content-type-options") == "nosniff", "Missing X-Content-Type-Options"
        assert res.headers.get("x-frame-options") == "SAMEORIGIN", "Missing X-Frame-Options"
        assert res.headers.get("x-xss-protection") == "1; mode=block", "Missing X-XSS-Protection"
        assert res.headers.get("referrer-policy") == "strict-origin-when-cross-origin", "Missing Referrer-Policy"
        print("  -> PASS: All production security headers present and verified.")

        # 2. Secrets Leak Check
        print("\n[2/7] Checking Config and Diagnostics for Secrets Leaks...")
        cfg = client.get("/api/v1/config").json()
        assert "password" not in str(cfg).lower()
        assert "secret" not in str(cfg).lower()
        assert "token" not in str(cfg).lower()
        assert "private_key" not in str(cfg).lower()
        print("  -> PASS: No secrets leaked in runtime config endpoint.")

        # 3. Real Image Detection
        print("\n[3/7] Testing Real Image Detection on bus.jpg...")
        assert SAMPLE_BUS_IMAGE.exists(), f"Missing sample: {SAMPLE_BUS_IMAGE}"
        with open(SAMPLE_BUS_IMAGE, "rb") as f:
            res = client.post(
                "/api/v1/detect/image",
                files={"file": ("bus.jpg", f, "image/jpeg")},
                data={"conf": 0.25, "iou": 0.45, "annotate": True}
            )
        assert res.status_code == 200, f"Detection failed: {res.text}"
        data = res.json()
        assert data["total_detections"] > 0, "No detections found on bus.jpg"
        assert "bus" in [d["class_name"] for d in data["detections"]], "Expected 'bus' detection"
        assert len(data["detections"]) == data["total_detections"]
        print(f"  -> PASS: Detected {data['total_detections']} objects ({set(d['class_name'] for d in data['detections'])}) in {data['inference_time_ms']:.2f}ms.")

        # 4. Real Video Tracking
        print("\n[4/7] Testing Real Video Tracking on sample_real_bus.mp4...")
        assert SAMPLE_REAL_BUS_VIDEO.exists(), f"Missing sample: {SAMPLE_REAL_BUS_VIDEO}"
        with open(SAMPLE_REAL_BUS_VIDEO, "rb") as f:
            res = client.post(
                "/api/v1/track/video",
                files={"file": ("bus_clip.mp4", f, "video/mp4")},
                data={"conf": 0.25, "stride": 2, "max_frames": 10, "annotate": False}
            )
        assert res.status_code == 200, f"Tracking failed: {res.text}"
        track_data = res.json()
        assert track_data["frames_processed"] > 0
        assert track_data["cumulative_unique_tracks"] > 0
        print(f"  -> PASS: Processed {track_data['frames_processed']} frames, tracked {track_data['cumulative_unique_tracks']} unique objects.")

        # 5. Real WebSocket Streaming with ByteTrack and Analytics
        print("\n[5/7] Testing Real WebSocket Streaming with ByteTrack + Analytics...")
        mgr = get_stream_manager()
        initial_connections = mgr.active_connections_count
        assert initial_connections == 0, f"Expected 0 active sessions, got {initial_connections}"

        with client.websocket_connect("/ws/stream") as ws:
            ack = ws.receive_json()
            assert ack["type"] == "connection_ack"
            session_id = ack["data"]["session_id"]
            assert mgr.active_connections_count == 1, "Session not registered"

            # Start streaming real bus video
            ws.send_json({
                "action": "start",
                "source": "video",
                "path": "data/samples/sample_real_bus.mp4",
                "stride": 1,
                "fps_limit": 30.0,
                "max_frames": 15,
                "annotate": True
            })

            start_msg = ws.receive_json()
            assert start_msg["type"] == "stream_started"

            frames_received = 0
            while frames_received < 5:
                frame_msg = ws.receive_json()
                if frame_msg["type"] == "frame_result":
                    frames_received += 1
                    fdata = frame_msg["data"]
                    assert "frame_number" in fdata
                    assert "annotated_frame" in fdata
                    assert fdata["annotated_frame"] is not None
                    assert "detections" in fdata
                    assert "tracks" in fdata

            print(f"  -> Streamed {frames_received} frames with live tracks.")

            # Stop stream
            ws.send_json({"action": "stop"})
            stopped_msg = ws.receive_json()
            assert stopped_msg["type"] == "stream_stopped"
            print("  -> Stream stopped gracefully.")

        # 6. WebSocket Disconnect & Session Cleanup Verification
        print("\n[6/7] Verifying WebSocket Disconnect Cleanup...")
        # Give small moment for disconnect handling
        time.sleep(0.1)
        assert mgr.active_connections_count == 0, f"Expected 0 active connections after close, got {mgr.active_connections_count}"
        assert session_id not in mgr._sessions, "Session was not pruned from manager"
        print("  -> PASS: Session was cleanly pruned on client disconnect.")

        # 7. Malicious Requests Rejection Verification
        print("\n[7/7] Testing Malicious Request Rejection...")
        # Path traversal on model switch
        res_trav = client.post("/api/v1/models/..%2F..%2Fetc%2Fpasswd/switch")
        assert res_trav.status_code in {404, 422}, f"Expected 404/422 on path traversal, got {res_trav.status_code}"

        # Oversized file upload
        large_bytes = b"0" * (16 * 1024 * 1024)  # 16 MB
        res_large = client.post(
            "/api/v1/detect/image",
            files={"file": ("too_large.jpg", large_bytes, "image/jpeg")}
        )
        assert res_large.status_code == 413, f"Expected 413 on oversized upload, got {res_large.status_code}"

        # Oversized WebSocket frame
        with client.websocket_connect("/ws/stream") as ws:
            _ = ws.receive_json()
            ws.send_text('{"action": "ping", "junk": "' + ("A" * 70000) + '"}')
            err_msg = ws.receive_json()
            assert err_msg["type"] == "error"
            assert err_msg["data"]["code"] == "MESSAGE_TOO_LARGE"

        # Negative camera index
        with client.websocket_connect("/ws/stream") as ws:
            _ = ws.receive_json()
            ws.send_json({"action": "start", "source": "webcam", "camera_index": -1})
            err_msg = ws.receive_json()
            assert err_msg["type"] == "error"
            assert err_msg["data"]["code"] == "INVALID_COMMAND"

        print("  -> PASS: Malicious requests strictly rejected with accurate HTTP / WS error codes.")

    print("\n" + "=" * 60)
    print("ALL REAL APPLICATION SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
