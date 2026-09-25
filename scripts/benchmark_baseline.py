"""
Step 14: Comprehensive Performance Baseline Measurement Script
Records real, empirical metrics for:
- Application startup time
- Detection latency (mean, min, max, p95)
- Video tracking frame rate (FPS) and latency
- WebSocket streaming round-trip frame latency
- REST API response times
- Host CPU, Memory (RAM) and GPU VRAM utilization
"""

import sys
from pathlib import Path
import time
import statistics
import psutil
import torch
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.app.core.config import PROJECT_ROOT
from backend.app.main import create_application
from backend.app.services.vision.detector import get_detector

SAMPLE_BUS = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
SAMPLE_VIDEO = PROJECT_ROOT / "data" / "samples" / "sample_real_bus.mp4"

def run_performance_baseline():
    print("=" * 65)
    print("STEP 14: MEASURING PERFORMANCE BASELINE")
    print("=" * 65)

    process = psutil.Process()
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Startup Time
    t_start = time.perf_counter()
    app = create_application()
    startup_time_sec = time.perf_counter() - t_start
    print(f"1. Application Startup Time: {startup_time_sec:.3f} s")

    # 2. Direct Detector Inference Latency
    detector = get_detector()
    device_used = detector.device
    
    # Warm-up run
    _ = detector.detect(SAMPLE_BUS)
    
    latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        _ = detector.detect(SAMPLE_BUS)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_det = statistics.mean(latencies)
    median_det = statistics.median(latencies)
    min_det = min(latencies)
    max_det = max(latencies)
    p95_det = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max_det

    print(f"2. Detection Latency ({device_used}, 15 iterations on 640x640 bus.jpg):")
    print(f"   - Mean:   {mean_det:.2f} ms")
    print(f"   - Median: {median_det:.2f} ms")
    print(f"   - Min:    {min_det:.2f} ms")
    print(f"   - Max:    {max_det:.2f} ms")
    print(f"   - P95:    {p95_det:.2f} ms")

    # 3. REST API Response Times
    with TestClient(app) as client:
        # GET /health
        health_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            res = client.get("/health")
            assert res.status_code == 200
            health_times.append((time.perf_counter() - t0) * 1000.0)
        mean_health = statistics.mean(health_times)

        # POST /api/v1/detect/image
        detect_api_times = []
        with open(SAMPLE_BUS, "rb") as f:
            bus_bytes = f.read()
        for _ in range(10):
            t0 = time.perf_counter()
            res = client.post("/api/v1/detect/image", files={"file": ("bus.jpg", bus_bytes, "image/jpeg")})
            assert res.status_code == 200
            detect_api_times.append((time.perf_counter() - t0) * 1000.0)
        mean_detect_api = statistics.mean(detect_api_times)

        print(f"3. REST API Response Times:")
        print(f"   - GET /health:             {mean_health:.2f} ms")
        print(f"   - POST /api/v1/detect/image: {mean_detect_api:.2f} ms")

        # 4. WebSocket Streaming Round-trip Latency & FPS
        ws_frame_intervals = []
        with client.websocket_connect("/ws/stream") as ws:
            _ = ws.receive_json()  # connection_ack
            ws.send_json({
                "action": "start",
                "source": "video",
                "path": "data/samples/sample_real_bus.mp4",
                "stride": 1,
                "fps_limit": 60.0,
                "max_frames": 20,
                "annotate": False,
            })
            _ = ws.receive_json()  # stream_started

            t_last = time.perf_counter()
            frame_count = 0
            while frame_count < 15:
                msg = ws.receive_json()
                if msg["type"] == "frame_result":
                    now = time.perf_counter()
                    ws_frame_intervals.append((now - t_last) * 1000.0)
                    t_last = now
                    frame_count += 1

            ws.send_json({"action": "stop"})
            _ = ws.receive_json()  # stream_stopped

        mean_ws_interval = statistics.mean(ws_frame_intervals[1:]) if len(ws_frame_intervals) > 1 else ws_frame_intervals[0]
        effective_ws_fps = 1000.0 / mean_ws_interval if mean_ws_interval > 0 else 0.0

        print(f"4. WebSocket Real-Time Streaming Performance (sample_real_bus.mp4):")
        print(f"   - Mean Frame Interval: {mean_ws_interval:.2f} ms")
        print(f"   - Effective Stream FPS: {effective_ws_fps:.1f} FPS")

    # 5. Resource Utilization
    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.5)
    print(f"5. Hardware Resource Utilization:")
    print(f"   - CPU Utilization:     {cpu_percent:.1f}%")
    print(f"   - Host RAM Usage:      {ram_end_mb:.1f} MB (Delta: +{ram_end_mb - ram_start_mb:.1f} MB)")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        gpu_reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
        print(f"   - GPU Device:          {gpu_name}")
        print(f"   - GPU VRAM Allocated:  {gpu_mem_mb:.1f} MB (Reserved: {gpu_reserved_mb:.1f} MB)")
    else:
        print("   - GPU: Not available (Running on CPU)")

    print("=" * 65)

if __name__ == "__main__":
    run_performance_baseline()
