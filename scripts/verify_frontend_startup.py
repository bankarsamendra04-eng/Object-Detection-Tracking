import subprocess
import time
import sys
import httpx

def main():
    print("Starting frontend dev server verification...")
    proc = subprocess.Popen(
        ["E:\\npm.cmd", "run", "dev", "--", "--port", "5173"],
        cwd=r"d:\Object-Detection-Tracking\frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )

    success = False
    start_time = time.time()
    try:
        while time.time() - start_time < 10:
            try:
                res = httpx.get("http://localhost:5173", timeout=2.0)
                if res.status_code == 200:
                    print("Frontend dev server successfully responded with HTTP 200 OK!")
                    success = True
                    break
            except Exception:
                time.sleep(0.5)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)

    if success:
        print("FRONTEND STARTUP VERIFICATION: SUCCESS")
        sys.exit(0)
    else:
        print("FRONTEND STARTUP VERIFICATION: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
