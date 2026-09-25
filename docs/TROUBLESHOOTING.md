# Troubleshooting & Diagnostics Guide

This document catalogs real operational issues, error diagnostics, and verified solutions for the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Computer Vision & Hardware Inference

### Issue: `CUDA device requested but torch.cuda.is_available() is False`
- **Symptom:** Backend starts on `cpu` despite having an NVIDIA GPU.
- **Root Cause:** PyTorch was installed without CUDA runtime wheels, or the NVIDIA host driver is missing.
- **Solution:**
  1. Verify host driver: `nvidia-smi`.
  2. Verify PyTorch installation has CUDA support:
     ```powershell
     python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
     ```
  3. Reinstall PyTorch with CUDA 12.4 support if needed:
     ```powershell
     pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
     ```

### Issue: `ModelUnavailableError: Model weights file not found on disk`
- **Symptom:** Switching to `custom_visdrone` returns HTTP 400.
- **Root Cause:** The model is cataloged in `models/registry/models.yaml`, but the `.pt` weights file has not been trained or placed in `models/custom/`.
- **Solution:**
  1. Check model status in `models/registry/models.yaml` (it will indicate `NOT_AVAILABLE`).
  2. Switch back to the active pretrained model: `general_pretrained`.

---

## 2. Media Ingestion & Camera Input

### Issue: `DeviceUnavailableError: Cannot open hardware camera at index 0`
- **Symptom:** Live stream fails to start with webcam.
- **Root Cause:** Another application (e.g. Zoom, Teams) is holding an exclusive DirectShow lock on the webcam, or camera permissions are disabled in Windows Settings.
- **Solution:**
  1. Probe available cameras via REST API:
     ```bash
     curl http://localhost:8000/api/v1/sources/webcam/probe?camera_index=0
     ```
  2. Close any background video conferencing applications.
  3. In Windows Settings, navigate to **Privacy & Security -> Camera** and enable **"Let desktop apps access your camera"**.

### Issue: `HTTP 413 Request Entity Too Large` on Upload
- **Symptom:** Large video or image uploads fail immediately.
- **Root Cause:** File exceeds the configured security limits (15 MB for images, 50 MB for videos).
- **Solution:**
  1. Compress the video file using FFmpeg:
     ```bash
     ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
     ```
  2. If deploying in Docker, ensure Nginx `client_max_body_size` matches the backend setting (default: 60M).

---

## 3. Real-Time WebSocket Streaming

### Issue: WebSocket Disconnects with Code 1006 / Broken Pipe
- **Symptom:** Live stream abruptly drops out during playback.
- **Root Cause:** Network latency caused client frame consumption to fall behind the server's streaming frame rate, or the reverse proxy terminated idle connections.
- **Solution:**
  1. The platform automatically discards stale frames when the queue exceeds 10 frames (`asyncio.Queue(maxsize=10)`).
  2. Lower the client streaming target FPS in the dashboard (e.g. from 60 FPS to 15 or 30 FPS).
  3. Ensure reverse proxy has long timeouts configured:
     ```nginx
     proxy_read_timeout 3600s;
     proxy_send_timeout 3600s;
     ```

---

## 4. Docker & Container Deployment

### Issue: `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`
- **Symptom:** `docker compose build` fails with pipe connection error on Windows.
- **Root Cause:** Docker Desktop is installed on Windows 11 Home, but the **WSL2 Subsystem** has not been installed or enabled with Administrator privileges.
- **Solution:**
  1. Open PowerShell as Administrator and run:
     ```powershell
     wsl.exe --install --no-distribution
     ```
  2. Launch Docker Desktop and accept the Subscription Service Agreement.
  3. Verify engine response: `docker info`.

### Issue: Docker Images Consuming System Drive (C:) Space
- **Symptom:** C: drive free space drops rapidly during Docker builds.
- **Root Cause:** Docker Desktop's default virtual disk (`ext4.vhdx`) is located in `%LOCALAPPDATA%\Docker\wsl`.
- **Solution:**
  1. Open Docker Desktop -> **Settings** -> **Resources** -> **Advanced**.
  2. Set **Disk image location** to an alternate data drive (e.g. `D:\Docker\data`).
  3. Click **Apply & restart**.
