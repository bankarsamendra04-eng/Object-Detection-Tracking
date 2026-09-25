# Docker Deployment & Containerization Guide

## 1. Overview
The **Real-Time Object Detection and Tracking System** features a containerized architecture engineered for reproducible deployment across local workstations, on-premise servers, and cloud environments (Linux, macOS, Windows).

### Architecture
```
                         +-----------------------------------+
                         |           Client Browser          |
                         +-----------------+-----------------+
                                           | HTTP (Port 80)
                                           v
                         +-----------------------------------+
                         |          vision-frontend          |
                         |          (Nginx 1.27 +            |
                         |        Compiled React SPA)        |
                         +--------+-----------------+--------+
                                  |                 |
                       /api/...   |                 |   /ws/...
                   (REST Proxy)   |                 | (WebSocket)
                                  v                 v
                         +-----------------------------------+
                         |          vision-backend           |
                         |      (Python 3.11 + FastAPI +     |
                         |       YOLOv8 + ByteTrack +        |
                         |        Spatial Analytics)         |
                         +-----------------+-----------------+
                                           |
                                           v
                     +---------------------------------------------+
                     |              Persistent Volumes             |
                     |  • vision-backend-data (SQLite DB, Uploads) |
                     |  • vision-backend-outputs (Exported media)  |
                     |  • vision-backend-logs (Application logs)   |
                     |  • ./models (Host bind mount for weights)   |
                     +---------------------------------------------+
```

---

## 2. Prerequisites & Windows Installation (D: Drive Storage Configuration)

### 2.1 Host Requirements
- **Operating System:** Windows 10/11 (with WSL2), Ubuntu 20.04/22.04/24.04, Debian 11/12, or macOS 12+
- **Docker Engine:** Version 24.0+ or Docker Desktop 4.25+
- **Docker Compose:** Version 2.20+ (included in modern Docker CLI)
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Disk Space:** 5 GB free disk space for container images and base layers

### 2.2 Optional Host Storage Isolation Strategy (Secondary Drive Setup)
> **Note on Machine-Level Host Storage:** Docker Desktop storage configuration (such as locating container data on an alternate drive like `D:`) is an optional host-level machine optimization and is completely independent of the project repository. The application runs seamlessly with standard default Docker installations as well.

When working with machines that have separate OS (`C:`) and data (`D:`) drives, you can configure Docker Desktop to locate heavy container data on an alternate partition:
- **Installer Path:** `D:\Docker\Docker Desktop Installer.exe`
- **Application Directory:** `D:\Docker\Docker Desktop`
- **WSL2 / Container Data Root:** `D:\Docker\data`
- **Isolated from Codebase:** Docker's internal data resides in `D:\Docker`, strictly separate from the project directory (`D:\Object-Detection-Tracking`).

### 2.3 Windows Installation Steps (Requires Administrator Elevation)
Because installing Docker Desktop and the WSL2 subsystem configures system services and virtualization components, an elevated Administrator session is required:

```powershell
# In PowerShell (Run as Administrator):

# Step A: Ensure WSL2 virtualization is installed
wsl.exe --install --no-distribution

# Step B: Execute Docker Desktop Installer configured directly to D:
& "D:\Docker\Docker Desktop Installer.exe" install --accept-license --installation-dir="D:\Docker\Docker Desktop" --wsl-default-data-root="D:\Docker\data" --hyper-v-default-data-root="D:\Docker\data" --windows-containers-default-data-root="D:\Docker\data"
```

### 2.4 Verifying / Relocating WSL2 Virtual Disk Location (D:\Docker\data)
If Docker Desktop is installed or uses default WSL2 distributions, relocate the heavy virtual disk (`ext4.vhdx`) to `D:\Docker\data`:
1. **Via Docker Desktop GUI:**
   - Navigate to `Settings` -> `Resources` -> `Advanced`.
   - Set **Disk image location** to `D:\Docker\data`.
   - Click **Apply & restart**.
2. **Via WSL CLI Migration:**
   ```powershell
   wsl --shutdown
   wsl --export docker-desktop-data "D:\Docker\docker-desktop-data.tar"
   wsl --unregister docker-desktop-data
   wsl --import docker-desktop-data "D:\Docker\data" "D:\Docker\docker-desktop-data.tar" --version 2
   del "D:\Docker\docker-desktop-data.tar"
   ```

### 2.5 Optional GPU Requirements
To run inference accelerated by an NVIDIA GPU inside Docker:
1. NVIDIA driver installed on host (v535+).
2. NVIDIA Container Toolkit (`nvidia-container-toolkit` / `nvidia-docker2`).
3. Verification: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

---

## 3. Quickstart (CPU Execution)

### 3.1 Starting the Stack
```bash
# Clone and enter directory
git clone https://github.com/YourRepo/Object-Detection-Tracking.git
cd Object-Detection-Tracking

# Launch complete multi-container stack in detached mode
docker compose up -d --build
```

### 3.2 Accessing Services
- **React Frontend Dashboard:** `http://localhost` (or `http://localhost:80`)
- **FastAPI REST API Docs (Swagger):** `http://localhost:8000/docs` or `http://localhost/docs`
- **ReDoc Interactive Docs:** `http://localhost:8000/redoc` or `http://localhost/redoc`
- **System Health Check:** `http://localhost:8000/health` or `http://localhost/health`
- **Real-Time WebSocket Stream:** `ws://localhost/ws/stream`

### 3.3 Verifying Service Health
```bash
# Check status of containers
docker compose ps

# Check backend health
curl -f http://localhost:8000/health

# Check frontend proxy health
curl -f http://localhost/healthz
```

---

## 4. Optional GPU Acceleration (NVIDIA Container Toolkit)

When deployed on a host equipped with an NVIDIA GPU and the NVIDIA Container Toolkit:

### 4.1 Launch with GPU Compose Override
```bash
# Launch with the GPU resource reservation overlay
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### 4.2 Verifying GPU Inference
Check the container system status:
```bash
curl http://localhost:8000/status
```
The response will indicate `"effective_device": "cuda:0"` and report active GPU VRAM allocation.

---

## 5. Storage & Volume Management

Persistent data is decoupled from container lifecycles via named volumes and bind mounts:

| Volume / Mount | Type | Target in Container | Purpose |
| :--- | :--- | :--- | :--- |
| `vision-backend-data` | Named Volume | `/app/data` | Stores SQLite database (`detection_tracking.db`) and uploaded media (`/app/data/uploads`). Persists across container restarts and rebuilds. |
| `vision-backend-outputs`| Named Volume | `/app/outputs` | Stores generated detection recordings, analytical snapshots, and reports. |
| `vision-backend-logs` | Named Volume | `/app/logs` | Persistent application runtime logs (`application.log`). |
| `./models` | Host Bind Mount | `/app/models` | Host directory containing `models.yaml` registry and custom models (`models/custom/`). Adding new weights does not require an image rebuild. |
| `./yolov8n.pt` | Host Bind Mount | `/app/yolov8n.pt:ro` | Read-only bind mount for the default pretrained YOLOv8 nano model checkpoint. |

### Inspecting Persistent Data
```bash
# View contents of database volume
docker run --rm -v vision-backend-data:/data alpine ls -la /data

# Backup SQLite database from Docker volume
docker run --rm -v vision-backend-data:/data -v $(pwd):/backup alpine cp /data/detection_tracking.db /backup/backup_detection.db
```

---

## 6. Management & Operational Commands

### 6.1 Viewing Container Logs
```bash
# Stream all logs
docker compose logs -f

# Stream only backend logs
docker compose logs -f backend

# Stream only frontend/Nginx logs
docker compose logs -f frontend
```

### 6.2 Restarting Services
```bash
# Restart entire stack
docker compose restart

# Restart only backend service
docker compose restart backend
```

### 6.3 Graceful Teardown
```bash
# Stop containers (preserves volume data)
docker compose down

# Stop containers and remove volumes (clean reset)
docker compose down -v
```

### 6.4 Shell Access into Containers
```bash
# Interactive bash shell inside backend container
docker compose exec -it backend /bin/bash

# Interactive sh shell inside frontend container
docker compose exec -it frontend /bin/sh
```

---

## 7. Model Management in Docker
- **Pretrained Checkpoints:** The default `yolov8n.pt` weights file (~6.2 MB) is baked into the image and mounted read-only from the host.
- **Custom Trained Models:** To add a custom model (e.g. VisDrone aerial weights `yolov8_visdrone.pt`):
  1. Place the checkpoint in `./models/custom/yolov8_visdrone.pt`.
  2. Verify that [`models/registry/models.yaml`](file:///d:/Object-Detection-Tracking/models/registry/models.yaml) references `weights_path: models/custom/yolov8_visdrone.pt`.
  3. The model becomes immediately available in the React dashboard (`Model Management`) without restarting containers.

---

## 8. Troubleshooting & Common Issues

### Issue 1: Port Conflict (Port 80 or 8000 already in use)
**Symptom:** `Error starting userland proxy: listen tcp4 0.0.0.0:80: bind: address already in use`  
**Solution:** Change host port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:80"   # Access frontend at http://localhost:8080
```

### Issue 2: Permission Denied on Writable Volumes
**Symptom:** `PermissionError: [Errno 13] Permission denied: '/app/data/detection_tracking.db'`  
**Solution:** The backend container runs as non-root `appuser` (UID 1000). Ensure host directories have appropriate write permissions:
```bash
chmod -R 775 data outputs logs
```

### Issue 3: WebSocket Connection Fails Behind Corporate Proxy
**Symptom:** React dashboard displays `WebSocket Disconnected`.  
**Solution:** Check that Nginx reverse proxy upgrade headers are transmitted:
- Verify `Upgrade: websocket` and `Connection: upgrade` headers.
- If using an external reverse proxy (Cloudflare, AWS ALB), enable WebSocket protocol support.

---

## 9. Local Development vs. Docker Deployment

| Feature | Local Development (`.venv`) | Docker Deployment (`docker compose`) |
| :--- | :--- | :--- |
| **Launch Command** | `python scripts/run_backend.py` | `docker compose up -d` |
| **Frontend** | `cd frontend; npm run dev` (Vite 5173) | Nginx Alpine (Port 80) |
| **Python Environment** | Native Windows `.venv` | Linux `python:3.11-slim` container |
| **Hardware Device** | Native CUDA 12.4 (RTX 2050) | CPU default (or GPU via Container Toolkit) |
| **Reverse Proxy** | Vite dev server proxy | Nginx 1.27 production proxy |
| **Code Changes** | Hot reload on file save | Volume-mounted or rebuild |
