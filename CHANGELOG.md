# Changelog

All notable changes, engineering milestones, and releases for the **Real-Time Object Detection & Tracking Platform** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-25

### Step 19 — Final GitHub Release & Portfolio Publication
- **Automatic Application Screenshot Generation:** Programmatically captured 7 high-resolution production views from the running Docker deployment (`docs/screenshots/01-dashboard.png` through `07-docker-runtime.png`).
- **README Screenshot Showcase:** Integrated visual demonstration sections with repository-relative paths and technical descriptions.
- **Repository Hygiene & Credentials Audit:** Verified strict exclusion of secrets, `.env` files, raw benchmark datasets, and build artifacts.
- **Release Packaging:** Finalized clean Git repository state, verified remotes, and prepared production release.

### Step 18 — Final Professional Audit & Quality Gate
- **Comprehensive Quality Audit:** Verified 17 architecture, CV, API, WebSocket, security, Docker, performance, and documentation dimensions.
- **Zero Critical / High Defects:** Certified 0 Critical, 0 High, 0 Medium, and 1 documented Low (hardware camera in headless Docker container; by design).
- **Restart Persistence Verification:** Tested and confirmed SQLite data persistence across Docker container restart via `vision-backend-data` volume.
- **Full Test Suite Integrity:** Verified 194 passing automated tests and 9 real runtime smoke tests.

### Step 17 — Professional Documentation & Portfolio Readiness
- **Root README.md Overhaul:** Structured full 30-section technical documentation, hero badges, architecture diagrams, and portfolio engineering highlights.
- **Dedicated Documentation Modules:** Created modular specifications across `docs/` (`API.md`, `USER_GUIDE.md`, `DEVELOPER_GUIDE.md`, `DATASETS.md`, `MODELS.md`, `TRACKING.md`, `ANALYTICS.md`, `PERFORMANCE.md`, `SECURITY.md`, `TESTING.md`, `DOCKER.md`, `CONFIGURATION.md`, `TROUBLESHOOTING.md`, `INTERVIEW_NOTES.md`).
- **Open-Source Licensing:** Established `LICENSE.md` with MIT licensing and third-party attribution.

### Step 16 — Dockerization & Containerization
- **Multi-Container Stack:** Authored `docker/backend.Dockerfile` (`python:3.11-slim`), `docker/frontend.Dockerfile` (multi-stage Node 20 build + Nginx Alpine runtime), and `docker-compose.yml`.
- **Nginx Reverse Proxy Gateway:** Implemented `docker/nginx.conf` routing REST API (`/api/v1/`), WebSocket live streams (`/ws/stream` with 3600s timeouts), and SPA client fallback.
- **D: Drive Storage Strategy:** Configured and documented external data root (`D:\Docker\data`) to protect host system drive (`C:`) from container image bloat.
- **Context Optimization:** Configured `.dockerignore` to filter 99.7% of repository files, reducing Docker build context to 16.82 MB.
- **GPU Override:** Provided `docker-compose.gpu.yml` for optional NVIDIA Container Toolkit acceleration.

### Step 15 — Performance Optimization & Benchmarking
- **Zero-Copy PCIe Transfer:** Replaced 3 fragmented GPU-to-CPU copies with a single contiguous PCIe tensor transfer (`res.boxes.data.cpu().numpy()`), cutting inference latency by 9.1% (14.88 ms → 13.52 ms).
- **`torch.inference_mode()` & Native FP16:** Accelerated model evaluation and eliminated Ultralytics quantization deprecation warnings.
- **$O(1)$ Ring Buffer Tracking:** Replaced Python list shifts in `SingleTrack.trajectory` with `collections.deque(maxlen=30)`.
- **Event Loop Protection:** Offloaded CPU-bound OpenCV bounding box rendering and JPEG compression via `asyncio.to_thread()`, reducing event loop blocking time from ~4 ms to <0.1 ms.
- **Memory Leak Proofing:** Verified 0.00 MB VRAM leakage over sustained 25+ frame streaming sessions.

### Step 14 — Quality Assurance & Testing Architecture
- **Multi-Level Test Pyramid:** Established 182 automated tests across 8 testing levels (Unit, Integration, REST API, WebSocket Protocol, Frontend State Machine, E2E Integration, Failure Injection, Concurrency/Load).
- **Automated QA Suite:** Verified 1x1 image inference, 2D grayscale arrays, zero-byte uploads, rapid reconnects, and monotonic analytics counting.

### Step 13 — Security Hardening & Error Handling
- **Path Traversal Defenses:** Built `Settings.resolve_safe_path()` rejecting directory traversal (`..`), UNC paths (`\\\\`), and null bytes (`\x00`).
- **Upload Boundaries:** Implemented MIME type validation and strict file size limits (15 MB image, 50 MB video, 8192px maximum dimension).
- **WebSocket Protection:** Capped WebSocket message payloads (`MAX_WS_MESSAGE_SIZE_BYTES = 64KB`) and enforced concurrency boundaries (`MAX_CONCURRENT_WS_SESSIONS = 10`).
- **Security Headers Middleware:** Added `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, and `Referrer-Policy`.

### Step 12 — Full End-to-End System Integration
- **Full-Stack Connectivity:** Verified seamless data flow from media inputs through YOLO detection, ByteTrack tracking, spatial analytics, FastAPI REST/WebSocket endpoints, to the React dashboard.
- **Real-Time Telemetry:** Bound live FPS, inference latency, tracking latency, active tracks, and cumulative counts to UI canvas and stat cards.

### Step 11 — React Object Detection Dashboard
- **React 18 & Vite SPA:** Implemented modern web dashboard with modular workspaces: Live Detection, Image Detection, Video Detection, Spatial Analytics, Model Registry, and System Settings.
- **Synchronized HTML5 Canvas:** Rendered bounding boxes, confidence tags, and persistent ByteTrack IDs synchronized to video frames.

### Step 10 — Model Registry & Custom Model Architecture
- **Declarative YAML Registry:** Authored `models/registry/models.yaml` tracking model identifiers, task modalities, recommended thresholds, and availability statuses.
- **Dynamic Hot-Switching:** Enabled on-the-fly model switching with cache clearance (`torch.cuda.empty_cache()`) without server restarts.

### Step 9 — Dataset Pipeline & VisDrone Audit
- **Standardized YOLO Normalization:** Developed converters transforming native bounding box formats into normalized YOLO coordinates `[cls cx cy nw nh]`.
- **Data Validation & Leakage Check:** Implemented cross-split image hashing to detect dataset leakage across training, validation, and test partitions.

### Step 8 — Real-Time WebSocket Streaming
- **Bi-Directional Streaming:** Implemented `StreamSession` and `WebSocketManager` supporting `start`, `stop`, `pause`, `resume`, and `ping` commands.
- **Backpressure Protection:** Integrated bounded frame queues (`asyncio.Queue(maxsize=10)`) with drop-stale policies under slow network conditions.

### Step 7 — FastAPI REST API Tier
- **Structured REST Controllers:** Implemented endpoints for health probes, image detection, video file tracking, spatial analytics reports, and system configuration.
- **Pydantic v2 Schemas:** Standardized structured request and response contracts.

### Step 6 — Tracked Object Analytics Engine
- **Spatial Tripwires:** Implemented 2D vector cross-product geometry (`ccw()`) for virtual line-crossing detection.
- **Stateful Entity Tracking:** Maintained cumulative unique entity counters, active track registries, and class distributions.

### Step 5 — Multi-Object Tracking Engine (ByteTrack)
- **Two-Stage Association:** Implemented ByteTrack association matching high-confidence detections first, followed by low-confidence recovery of occluded targets.
- **Persistent ID Lifecycle:** Retained lost tracks in persistence buffers before deletion.

### Step 4 — Multi-Source Input Pipeline
- **Unified Ingestion:** Developed context-managed input abstraction supporting images (`ImageInput`), video files (`VideoInput`), and hardware cameras (`WebcamInput`).

### Step 3 — YOLO Detection Engine
- **Inference Service:** Integrated Ultralytics YOLOv8 with dynamic hardware device selection (`cuda:0` / `cpu`), dynamic confidence, and NMS IoU filtering.

### Steps 1–2 — Environment Foundation & Project Architecture
- **Repository Setup:** Established project structure, virtual environment, dependencies (`torch`, `ultralytics`, `fastapi`, `react`), and baseline configuration.
