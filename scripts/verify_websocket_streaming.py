"""
Real WebSocket Streaming Smoke Test Script.
Starts the FastAPI server live on http://127.0.0.1:8000 and connects using a real websockets client.
Tests:
1. Connection handshake & connection_ack
2. Heartbeat ping/pong
3. Real video stream (data/samples/sample_real_bus.mp4) with detection, ByteTrack tracking & analytics
4. Controlled stop & stream_stopped
5. Live hardware webcam stream (if available) with bounded frame capture
6. Clean disconnect & resource release
"""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
import httpx
import uvicorn
import websockets

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.main import app
from backend.app.services.vision.input_sources import probe_camera_availability

logger = get_logger("verify_ws_streaming")

HOST = "127.0.0.1"
PORT = 8000
WS_URL = f"ws://{HOST}:{PORT}/api/v1/ws/stream"
HTTP_HEALTH_URL = f"http://{HOST}:{PORT}/health"


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


async def wait_for_server(timeout: float = 15.0):
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(HTTP_HEALTH_URL)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return False


async def run_smoke_test():
    print(f"\n=======================================================")
    print(f"STEP 8: LIVE WEBSOCKET STREAMING SMOKE TEST")
    print(f"Target: {WS_URL}")
    print(f"=======================================================\n")

    # 1. Connect to WebSocket
    print("1. Connecting to WebSocket endpoint...")
    async with websockets.connect(WS_URL) as ws:
        # Expect connection_ack
        raw_ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(raw_ack)
        assert ack["type"] == "connection_ack", f"Expected connection_ack, got {ack}"
        session_id = ack["session_id"]
        print(f"   [PASS] Received connection_ack | Session ID: {session_id}")

        # 2. Ping / Pong
        print("2. Testing Heartbeat ping/pong...")
        await ws.send(json.dumps({"action": "ping"}))
        raw_pong = await asyncio.wait_for(ws.recv(), timeout=5.0)
        pong = json.loads(raw_pong)
        assert pong["type"] == "pong", f"Expected pong, got {pong}"
        print(f"   [PASS] Received pong | Timestamp: {pong['data']['timestamp']}")

        # 3. Stream Real Video File
        video_sample = "data/samples/sample_real_bus.mp4"
        print(f"3. Starting Real Video Stream ({video_sample})...")
        await ws.send(json.dumps({
            "action": "start",
            "source": "video",
            "path": video_sample,
            "fps_limit": 15.0,
            "stride": 1,
            "annotate": True,
        }))

        raw_started = await asyncio.wait_for(ws.recv(), timeout=5.0)
        started = json.loads(raw_started)
        assert started["type"] == "stream_started", f"Expected stream_started, got {started}"
        meta = started["data"]["metadata"]
        print(f"   [PASS] Stream started | Resolution: {meta.get('width')}x{meta.get('height')} | FPS Limit: 15.0")

        # Collect 5 consecutive frame_result messages
        print("4. Collecting real-time streaming frame_result messages...")
        frames_received = 0
        total_objects_detected = 0

        for i in range(5):
            raw_frame = await asyncio.wait_for(ws.recv(), timeout=8.0)
            msg = json.loads(raw_frame)
            if msg["type"] == "frame_result":
                frames_received += 1
                data = msg["data"]
                fn = data["frame_number"]
                inf_time = data["inference_time_ms"]
                trk_time = data["tracking_time_ms"]
                fps = data["fps"]
                dets = data["detections"]
                trks = data["tracks"]
                analytics = data["analytics"]
                active_trks = data["active_tracks"]
                unique_trks = data["unique_tracks"]
                annotated = bool(data.get("annotated_frame"))
                total_objects_detected += len(dets)

                class_names = [d["class_name"] for d in dets]
                track_ids = [t["track_id"] for t in trks]
                print(f"   [FRAME {fn}] Inf: {inf_time:.1f}ms | Track: {trk_time:.1f}ms | FPS: {fps} | "
                      f"Detections: {class_names} | Active Tracks: {track_ids} | Unique Count: {unique_trks} | Annotated: {annotated}")

        assert frames_received >= 5, f"Expected at least 5 frames, got {frames_received}"
        print(f"   [PASS] Successfully streamed {frames_received} frames with real detection, ByteTrack IDs & analytics!")

        # 4. Stop Video Stream
        print("5. Stopping stream cleanly...")
        await ws.send(json.dumps({"action": "stop"}))
        stopped_msg = None
        while True:
            raw_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw_msg)
            if msg["type"] == "stream_stopped":
                stopped_msg = msg
                break
        print(f"   [PASS] Received stream_stopped | Reason: {stopped_msg['data']['reason']} | Processed: {stopped_msg['data']['frames_processed']} frames")

    # 5. Hardware Webcam Smoke Test (if available)
    probe_res = probe_camera_availability(0)
    cam_available = probe_res.get("available", False) and probe_res.get("readable", False)
    if cam_available:
        print(f"\n6. Testing Hardware Webcam Streaming (Camera Index 0: {probe_res.get('resolution')} @ {probe_res.get('fps')} fps)...")
        async with websockets.connect(WS_URL) as ws_cam:
            _ = await ws_cam.recv()  # ack

            await ws_cam.send(json.dumps({
                "action": "start",
                "source": "webcam",
                "camera_index": 0,
                "fps_limit": 15.0,
            }))

            started_cam = json.loads(await asyncio.wait_for(ws_cam.recv(), timeout=5.0))
            assert started_cam["type"] == "stream_started"
            print("   [PASS] Webcam stream started successfully!")

            # Collect 3 live webcam frames
            cam_frames = 0
            for _ in range(3):
                msg = json.loads(await asyncio.wait_for(ws_cam.recv(), timeout=5.0))
                if msg["type"] == "frame_result":
                    cam_frames += 1
                    data = msg["data"]
                    print(f"   [WEBCAM FRAME {data['frame_number']}] Inf: {data['inference_time_ms']:.1f}ms | Tracks: {data['active_tracks']}")

            assert cam_frames >= 3
            print(f"   [PASS] Successfully processed {cam_frames} live webcam frames!")

            await ws_cam.send(json.dumps({"action": "stop"}))
            # Drain until stopped
            while True:
                msg = json.loads(await ws_cam.recv())
                if msg["type"] == "stream_stopped":
                    break
            print("   [PASS] Webcam stream stopped and hardware camera released cleanly!")
    else:
        print("\n6. Hardware webcam (index 0) not available on this environment; skipped webcam test.")

    print("\n=======================================================")
    print("ALL REAL WEBSOCKET SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=======================================================\n")


def main():
    server_thread = ServerThread()
    server_thread.start()

    try:
        ready = asyncio.run(wait_for_server(15.0))
        if not ready:
            print("ERROR: Server failed to start within timeout.")
            sys.exit(1)

        asyncio.run(run_smoke_test())
    finally:
        print("Shutting down live server...")
        server_thread.stop()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
