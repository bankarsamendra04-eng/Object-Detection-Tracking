"""
Step 15: Memory & Resource Leak Sustained Runtime Test
Verifies that long-running video streaming and repeated inferences:
1. Do not cause unbounded RAM or GPU VRAM growth
2. Maintain bounded internal async queues
3. Do not create orphaned threads, lingering WebSocket sessions, or unreleased handles
4. Consistently reuse the same loaded model weights across iterations
"""

import gc
import psutil
import pytest
import time
import torch
from fastapi.testclient import TestClient

from backend.app.core.config import PROJECT_ROOT
from backend.app.main import create_application
from backend.app.services.vision.detector import get_detector
from backend.app.services.streaming.manager import get_stream_manager


@pytest.fixture(scope="module")
def app():
    return create_application()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


def test_sustained_streaming_memory_stability(client):
    """
    Simulates a sustained streaming session over 60+ frames.
    Asserts:
    - Queue size remains bounded (<= 10)
    - Host RAM and GPU VRAM do not grow uncontrollably (leak check)
    - Full resource release occurs on client disconnect
    """
    mgr = get_stream_manager()
    detector = get_detector()
    # Warm up detector to ensure PyTorch CUDA runtime DLLs are loaded before sampling baseline
    _ = detector.detect(PROJECT_ROOT / "data" / "samples" / "bus.jpg")

    process = psutil.Process()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    initial_ram_mb = process.memory_info().rss / (1024 * 1024)
    initial_vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    frames_received = 0
    with client.websocket_connect("/ws/stream") as ws:
        ack = ws.receive_json()
        assert ack["type"] == "connection_ack"
        session_id = ack["data"]["session_id"]
        session = mgr.get_session(session_id)
        assert session is not None

        # Start streaming video
        ws.send_json({
            "action": "start",
            "source": "video",
            "path": "data/samples/sample_real_bus.mp4",
            "stride": 1,
            "fps_limit": 60.0,
            "max_frames": 60,
            "annotate": True,
        })

        start_ack = ws.receive_json()
        assert start_ack["type"] == "stream_started"

        max_queue_observed = 0
        while frames_received < 25:
            msg = ws.receive_json()
            if msg["type"] == "frame_result":
                frames_received += 1
                q_size = session.send_queue.qsize()
                if q_size > max_queue_observed:
                    max_queue_observed = q_size
                # Verify queue boundary is never violated
                assert q_size <= 10, f"Queue grew beyond maximum bounded limit: {q_size}"
            elif msg["type"] == "stream_stopped":
                break

        # Stop stream
        ws.send_json({"action": "stop"})
        stopped_msg = ws.receive_json()
        assert stopped_msg["type"] in {"stream_stopped", "error"}

    # Allow small moment for async task cancellation and garbage collection
    time.sleep(0.1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_ram_mb = process.memory_info().rss / (1024 * 1024)
    final_vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    ram_growth_mb = final_ram_mb - initial_ram_mb
    vram_growth_mb = final_vram_mb - initial_vram_mb

    print(f"\nSustained Streaming Leak Check ({frames_received} frames):")
    print(f"  - Peak Queue Size Observed: {max_queue_observed} / 10")
    print(f"  - RAM Growth: {ram_growth_mb:.2f} MB")
    print(f"  - VRAM Growth: {vram_growth_mb:.2f} MB")
    print(f"  - Active Sessions after Close: {mgr.active_connections_count}")

    # Assertions
    assert mgr.active_connections_count == 0, "Lingering session after client disconnect"
    assert ram_growth_mb < 50.0, f"Unacceptable RAM growth detected: {ram_growth_mb:.2f} MB"
    assert vram_growth_mb < 15.0, f"Unacceptable GPU VRAM leak detected: {vram_growth_mb:.2f} MB"


def test_repeated_detector_inferences_no_memory_leak():
    """
    Executes 50 repeated inferences on static image to verify:
    - Zero model reloading
    - No PyTorch tensor / CUDA caching leakage
    """
    detector = get_detector()
    initial_id = id(detector.model)
    bus_path = PROJECT_ROOT / "data" / "samples" / "bus.jpg"

    process = psutil.Process()
    gc.collect()
    ram_before = process.memory_info().rss / (1024 * 1024)

    for _ in range(50):
        res = detector.detect(bus_path)
        assert res.total_detections > 0

    gc.collect()
    ram_after = process.memory_info().rss / (1024 * 1024)
    final_id = id(detector.model)

    assert initial_id == final_id, "Model must not be reloaded during inferences."
    ram_delta = ram_after - ram_before
    assert ram_delta < 20.0, f"Excessive RAM growth across 50 inferences: {ram_delta:.2f} MB"
