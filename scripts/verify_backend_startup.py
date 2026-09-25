import sys
import time
import threading
from pathlib import Path
import httpx
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.core.config import settings

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        config = uvicorn.Config(
            app=app,
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()


def main():
    print(f"Launching FastAPI Server Smoke Test on {BASE_URL} ...")
    server_thread = ServerThread()
    server_thread.start()

    success = False
    start_time = time.time()

    # Wait for server readiness
    while time.time() - start_time < 10:
        try:
            res = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if res.status_code == 200:
                print("1. [PASS] Root /health responded: 200 OK")
                success = True
                break
        except Exception:
            time.sleep(0.5)

    if not success:
        print("[FAIL] Server failed to start within timeout.")
        server_thread.server.should_exit = True
        sys.exit(1)

    try:
        # 2. Check /status
        res_status = httpx.get(f"{BASE_URL}/status", timeout=2.0)
        assert res_status.status_code == 200
        st_data = res_status.json()
        print(f"2. [PASS] /status responded: 200 OK | Device: {st_data['effective_device']} | Model: {st_data['model']['name']}")

        # 3. Check /docs
        res_docs = httpx.get(f"{BASE_URL}/docs", timeout=2.0)
        assert res_docs.status_code == 200
        print("3. [PASS] /docs Swagger UI is accessible: 200 OK")

        # 4. Check /openapi.json
        res_openapi = httpx.get(f"{BASE_URL}/openapi.json", timeout=2.0)
        assert res_openapi.status_code == 200
        print("4. [PASS] /openapi.json schema generated and accessible: 200 OK")

        # 5. Real Image Detection API Call
        sample_img = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
        assert sample_img.is_file(), f"Missing sample image at {sample_img}"
        with open(sample_img, "rb") as f:
            files = {"file": ("bus.jpg", f, "image/jpeg")}
            res_detect = httpx.post(f"{BASE_URL}/api/v1/detect/image", files=files, timeout=10.0)

        assert res_detect.status_code == 200
        det_data = res_detect.json()
        assert det_data["total_detections"] > 0
        print(f"5. [PASS] Real Image Detection API succeeded: 200 OK | Detected {det_data['total_detections']} objects ({det_data['class_counts']}) in {det_data['inference_time_ms']}ms")

        # 6. Check /api/v1/config
        res_cfg = httpx.get(f"{BASE_URL}/api/v1/config", timeout=2.0)
        assert res_cfg.status_code == 200
        cfg_data = res_cfg.json()
        assert "password" not in res_cfg.text.lower()
        print(f"6. [PASS] /api/v1/config responded: 200 OK | Safe configuration verified")

        print("\nALL API SMOKE TESTS PASSED SUCCESSFULLY!")

    except Exception as e:
        print(f"\n[FAIL] Smoke test encountered an assertion error: {e}")
        success = False
    finally:
        server_thread.server.should_exit = True
        server_thread.join(timeout=3.0)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
