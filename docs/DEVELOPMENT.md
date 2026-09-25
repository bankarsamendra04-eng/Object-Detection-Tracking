# Development & Operational Guide

## 1. Quickstart

### 1.1 Activating the Virtual Environment
```powershell
# In PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
```

### 1.2 Running the Backend Server
```powershell
# Using the launcher script
python scripts/run_backend.py

# Or directly with Uvicorn
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 2. Multi-Input Pipeline (Step 4)

The project provides a unified `InputSource` abstraction layer (`backend/app/services/vision/input_sources.py`) decoupling media ingestion from detection inference:

```
InputSource (Abstract Context Manager)
├── ImageInput   (JPG, JPEG, PNG, WEBP, BMP, NumPy arrays)
├── VideoInput   (MP4, AVI, MOV, MKV, WEBM)
└── WebcamInput  (Hardware camera indices, bounded capture)
```

| Source | Supported Formats | Features & Metadata | Error Handling |
| :--- | :--- | :--- | :--- |
| **ImageInput** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `np.ndarray` | Resolution (`width`, `height`), channel count, file size, optional bounding box rendering via `annotate=True`. | `FileNotFoundMediaError`, `UnsupportedFormatError`, `CorruptMediaError`. |
| **VideoInput** | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` | Sequential frame iteration (`__iter__`), frame numbering, timestamps (`timestamp_ms`), configurable `stride` and `max_frames`, clean EOF handling. | `FileNotFoundMediaError`, `UnsupportedFormatError`, `CorruptMediaError`. |
| **WebcamInput**| Physical camera indices (default `0`) | DirectShow Windows acceleration, bounded safe frame capture, non-blocking camera probing via `probe_camera_availability()`. | `DeviceUnavailableError`, `StreamReadError`. |

---

## 3. Object Tracking & Persistent IDs (Step 5)

The tracking engine (`backend/app/services/vision/tracker.py`) consumes structured detection outputs from `DetectionEngine` without duplicating YOLO inference:

```
Input (Video / Webcam)
      ↓
DetectionEngine  →  DetectionResult
      ↓
ByteTrackTracker →  TrackingResult (TrackedObject with Persistent IDs)
      ↓
annotate_tracking_frame  →  Visualized Frame (e.g. 'Person #7 0.91')
```

- **Two-Stage Association:** High-confidence detections matched first, followed by low-confidence recovery of occluded or blurred targets.
- **Persistent ID Lifecycle:** Maintains track continuity across frames; unmatched tracks enter `LOST` state and are retained in a persistence buffer for up to `persistence_buffer` frames before removal.

---

## 4. Tracked Object Analytics Engine (Step 6)

The Analytics Engine (`backend/app/services/analytics/counter.py`) consumes `TrackingResult` instances from Step 5 to compute telemetry, spatial events, and performance metrics:

```
Tracked Objects (Step 5 TrackingResult)
      ↓
AnalyticsEngine
      ↓
State Management (Lifecycle, Trajectories, Line Crossings, ROIs)
      ↓
AnalyticsSnapshot (Metrics, ClassStatistics, Events, FPS)
```

### 4.1 Counting Concepts: Crucial Distinctions

It is essential to distinguish the three tiers of counts:
1. **Total Detections (`total_detections`):**
   - The raw sum of bounding boxes produced by the YOLO detector across all processed frames.
   - *Example:* If 1 person walks across 30 frames, total detections = 30.
2. **Current Active Objects (`active_objects`):**
   - The number of tracked objects confirmed and actively visible in the *current* frame.
   - *Example:* If 2 people are currently on screen, active count = 2.
3. **Cumulative Unique Objects (`total_unique_objects`):**
   - The total number of distinct tracked entities (unique `track_id` values) observed throughout the entire session.
   - *Example:* If Person #1 and Person #2 appear and leave, and Person #3 enters, cumulative unique count = 3.

### 4.2 Spatial Analytics & Events

- **Virtual Tripwires (Line Crossing):**
  - Configurable 2D segment: `LineDefinition(line_id="gate", start_point=(x1, y1), end_point=(x2, y2), direction="BOTH")`
  - Evaluates segment intersection between successive trajectory points.
  - Determines directional orientation (`A_TO_B` vs `B_TO_A`) via 2D vector cross-product.
  - Employs anti-duplicate cooldowns (15 frames) to avoid re-counting hovering objects.
- **Regions of Interest (ROI):**
  - Arbitrary polygon definition: `ROIDefinition(roi_id="zone_a", points=[(x1,y1), (x2,y2), ...], anchor="bottom_center")`
  - Evaluates spatial containment via `point_in_polygon`.
  - Generates `ROI_ENTER` and `ROI_EXIT` events.
  - Computes dwell duration: records entry timestamp and calculates elapsed seconds upon exit.
- **Track Lifecycle Events:**
  - `OBJECT_ENTERED`: Emitted upon first appearance of a new `track_id`.
  - `OBJECT_EXITED`: Emitted when an object has been absent for more than `exit_threshold_frames` (default: 5 frames).

---

## 5. Production REST API Layer (Step 7)

The FastAPI application exposes modular REST endpoints documented interactively via OpenAPI:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### 5.1 Endpoint Directory

| Category | Method | Path | Description |
| :--- | :--- | :--- | :--- |
| **System** | `GET` | `/health`, `/api/v1/health` | Service liveness probe returning system status, timestamp, and uptime. |
| **System** | `GET` | `/status`, `/api/v1/status` | Hardware status, active GPU device, loaded YOLO model, and active tracks. |
| **Detection** | `POST` | `/api/v1/detect/image` (alias: `/api/v1/detection/image`) | Multipart upload for image inference with optional query params `confidence` and `iou`. Persists summary in DB. |
| **Tracking** | `POST` | `/api/v1/track/video` (alias: `/api/v1/tracking/video`) | Processes uploaded MP4/AVI/MOV video file through detection + ByteTrack. |
| **Tracking** | `POST` | `/api/v1/track/frame` | Single-frame tracking inference maintaining persistent ID state across sequential calls. |
| **Tracking** | `POST` | `/api/v1/track/reset` | Resets active tracker state and ID counter. |
| **Analytics** | `GET` | `/api/v1/analytics/current` | Active snapshot: current counts, cumulative unique counts, active objects, FPS. |
| **Analytics** | `GET` | `/api/v1/analytics/summary` | Historical telemetry summary aggregated from persistence database. |
| **Analytics** | `POST` | `/api/v1/analytics/reset` | Resets cumulative in-memory analytics engine metrics. |
| **Analytics** | `POST` | `/api/v1/analytics/line` | Creates virtual tripwire line for bidirectional crossing counts. |
| **Analytics** | `DELETE` | `/api/v1/analytics/line/{line_id}` | Removes virtual tripwire line. |
| **Analytics** | `POST` | `/api/v1/analytics/roi` | Registers polygon Region of Interest for occupancy and dwell tracking. |
| **Analytics** | `DELETE` | `/api/v1/analytics/roi/{roi_id}` | Removes polygon Region of Interest. |
| **Analytics** | `GET` | `/api/v1/analytics/events` | Fetches filtered event log (`OBJECT_ENTERED`, `OBJECT_EXITED`, `LINE_CROSSED`, `ROI_ENTER`, `ROI_EXIT`). |
| **Config** | `GET` | `/api/v1/config` | Returns safe non-sensitive configuration parameters (no secret keys). |
| **Config** | `GET` | `/api/v1/sources` | Lists supported media formats, video codecs, and default camera index. |
| **Config** | `GET` | `/api/v1/sources/webcam/probe` | Live hardware camera availability probe (DirectShow). |

### 5.2 Validation, Payload Limits & Error Schema

All endpoints validate query parameters and payload bodies using Pydantic v2 schemas:
- **Confidence & IoU:** Constrained to `[0.0, 1.0]`. Out-of-bounds values return `422 Unprocessable Entity`.
- **Payload Size Limits:** Images capped at 15 MB; videos capped at 50 MB. Oversized files return `413 Request Entity Too Large`.
- **Media Validation:** Non-image/non-video files or corrupt headers return `400 Bad Request`.
- **Standardized Error Format:**
```json
{
  "status": "error",
  "error": "BAD_REQUEST",
  "message": "Corrupt or unreadable image data.",
  "timestamp": "2026-09-24T11:28:48.514485Z"
}
```

---

## 6. Real-Time WebSocket Streaming Layer (Step 8)

The application provides a high-throughput, bi-directional WebSocket interface for streaming real-time detections, persistent ByteTrack tracking, and live analytics telemetry:
- Primary Endpoint: `ws://localhost:8000/api/v1/ws/stream`
- Root Alias: `ws://localhost:8000/ws/stream`

### 6.1 WebSocket Architecture & Lifecycle
```
Client (Browser / Client App)
          │
  (1) Connect (ws://.../api/v1/ws/stream)
          ▼
  FastAPI WebSocket Route
          ▼
  WebSocketManager (Connection & Session Registry)
          │
  (2) Handshake Acknowledgement (connection_ack)
          │
  (3) Client Issues Command ({"action": "start", "source": "video"|"webcam"})
          ▼
  StreamSession (Session-Isolated Pipeline)
          │
          ├── Input Pipeline (WebcamInput / VideoInput)
          │         ↓
          ├── DetectionEngine (Shared Singleton, CUDA RTX 2050 / CPU)
          │         ↓
          ├── ByteTrackTracker (Session-Isolated Persistent IDs)
          │         ↓
          ├── AnalyticsEngine (Session-Isolated Counts, Lines, ROIs)
          │         ↓
  (4) Bounded Backpressure Queue (maxsize=10)
          │
  (5) Sender Worker ──> Client (frame_result stream @ target FPS)
          │
  (6) Stop / Disconnect ──> Resource Cleanup (Camera/Video released)
```

### 6.2 Client Command Protocol

Clients send JSON command payloads to control streaming state:

```json
// Start Video Stream
{
  "action": "start",
  "source": "video",
  "path": "data/samples/sample_real_bus.mp4",
  "fps_limit": 15.0,
  "stride": 1,
  "conf": 0.35,
  "iou": 0.45,
  "annotate": true
}

// Start Live Webcam Stream
{
  "action": "start",
  "source": "webcam",
  "camera_index": 0,
  "fps_limit": 15.0
}

// Stream Control Commands
{"action": "pause"}
{"action": "resume"}
{"action": "stop"}
{"action": "ping"}
```

### 6.3 Server Outbound Message Protocol

All messages emitted by the server follow the standardized envelope:
```json
{
  "type": "<message_type>",
  "timestamp": "2026-09-24T11:45:58.190970Z",
  "session_id": "36e62d0d-9cf8-405f-b679-091a6d0abb4c",
  "data": { ... }
}
```

| Message Type | Description | Key Data Fields |
| :--- | :--- | :--- |
| `connection_ack` | Handshake response | `session_id`, `message`, `supported_sources` |
| `stream_started` | Stream initialisation confirmation | `session_id`, `source`, `metadata`, `fps_limit` |
| `frame_result` | Per-frame live CV payload | `frame_number`, `inference_time_ms`, `tracking_time_ms`, `fps`, `detections`, `tracks`, `analytics`, `annotated_frame` (optional base64) |
| `stream_stopped` | Normal stream termination / EOF | `session_id`, `reason`, `frames_processed`, `duration_seconds` |
| `error` | Structured error notification | `code`, `message`, `details` |
| `pong` | Heartbeat latency check | `timestamp` |

### 6.4 Backpressure Protection & Frame Rate Control
- **Bounded Async Queues:** Each session maintains an `asyncio.Queue(maxsize=10)`. When slow clients cause the queue to saturate, the oldest intermediate `frame_result` is dropped in favor of the freshest state. The server never stalls or leaks memory.
- **Paced Delivery:** Frame delivery is throttled using `asyncio.sleep(max(0.001, target_interval - elapsed_time))` to match the requested `fps_limit` (1.0 to 60.0 FPS).
- **Path Traversal Security:** Video paths are strictly validated to prevent `..` directory traversal and access outside the project workspace.
- **Thread Safety:** I/O and GPU inference (`cv2.VideoCapture.read`, YOLO `predict`, ByteTrack `update`) execute inside `asyncio.to_thread()` ensuring the ASGI event loop remains completely non-blocking.

---

## 7. Dataset Management & Small-Object Pipeline (Step 9)

The platform provides a reproducible, automated dataset pipeline tailored for small-object and crowded-scene detection (benchmarked using the VisDrone2019 dataset from the AISKYEYE team, Tianjin University).

> [!NOTE]
> **Model Training Policy:** Step 9 establishes dataset auditing, downloading, annotation conversion, and quality validation. Model training and hyperparameter optimization belong to subsequent phases.

### 7.1 Dataset Directory Architecture
```
data/
├── datasets/
│   ├── registry/          # datasets.yaml catalog
│   ├── raw/               # Immutable raw downloads (VisDrone, Custom)
│   ├── processed/         # Normalized YOLO-formatted datasets (images/ & labels/)
│   ├── splits/            # Split manifests and validation audit reports
│   └── samples/           # Visual inspection sample images with rendered boxes
└── configs/datasets/      # Ultralytics-compatible YAML configs (visdrone.yaml, custom.yaml)
```

### 7.2 Dataset Scripts & Workflows
| Script | Command | Description |
| :--- | :--- | :--- |
| **Registry** | `python scripts/dataset/dataset_registry.py list` | Queries and updates the authoritative catalog in `datasets.yaml`. |
| **Downloader** | `python scripts/dataset/download_dataset.py --dataset-id visdrone_det_val` | Pre-checks disk space, downloads via chunked streaming with resume support, and extracts with ZipSlip protection. |
| **Converter** | `python scripts/dataset/convert_annotations.py --dataset-id visdrone_det_val` | Converts VisDrone raw CSV annotations to YOLO normalized format `[cls cx cy nw nh]`, filtering ignored regions. |
| **Validator** | `python scripts/dataset/validate_dataset.py --dataset-id visdrone_det_val` | Audits image readability, syntax, coordinate bounds, cross-split leakage, and small-object distribution. |
| **Inspector** | `python scripts/dataset/inspect_dataset.py --dataset-id visdrone_det_val --samples 5` | Generates visual bounding box sample renders in `data/datasets/samples/images/`. |

### 7.3 Small-Object Categorization & Metric Standards
Object bounding boxes are evaluated against standard aerial & COCO pixel-area thresholds:
- **Very Small:** Area $< 256\text{ px}^2$ ($< 16\times 16\text{ px}$)
- **Small:** $256 \le \text{Area} < 1,024\text{ px}^2$ ($16\times 16$ to $32\times 32\text{ px}$)
- **Medium:** $1,024 \le \text{Area} < 9,216\text{ px}^2$ ($32\times 32$ to $96\times 96\text{ px}$)
- **Large:** $\text{Area} \ge 9,216\text{ px}^2$ ($> 96\times 96\text{ px}$)

In the official VisDrone2019 validation split:
- **Over 68.5%** of all annotated entities are small or very small (median bounding box area: $520.0\text{ px}^2$).

---

## 8. Configuration & Overrides

All variables can be configured via `.env` or system environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PORT` | API server port | `8000` |
| `CONFIDENCE_THRESHOLD` | Detection confidence filter | `0.35` |
| `IOU_THRESHOLD` | Non-maximum suppression threshold | `0.45` |
| `DEVICE` | Target hardware (`auto`, `cuda:0`, `cpu`)| `auto` |
| `MODEL_NAME` | Default weights file | `yolov8n.pt` |
| `TRACKER_TYPE` | Default tracker | `bytetrack.yaml` |
| `TRACK_HIGH_THRESH` | ByteTrack high association threshold | `0.5` |
| `TRACK_LOW_THRESH` | ByteTrack low recovery threshold | `0.1` |
| `TRACK_MATCH_THRESH` | ByteTrack matching threshold | `0.5` |
| `TRACK_PERSISTENCE_BUFFER`| Max frames to retain lost tracks | `30` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./data/detection_tracking.db` |

---

## 9. Running Automated Tests

Run the full automated test suite with `pytest`:

```powershell
.\.venv\Scripts\pytest.exe
```

Run specific test modules:
```powershell
.\.venv\Scripts\pytest.exe tests/unit/test_dataset_pipeline.py
.\.venv\Scripts\pytest.exe tests/unit/test_analytics.py
.\.venv\Scripts\pytest.exe tests/unit/test_tracker.py
.\.venv\Scripts\pytest.exe tests/unit/test_input_sources.py
.\.venv\Scripts\pytest.exe tests/unit/test_detector.py
.\.venv\Scripts\pytest.exe tests/api/test_endpoints.py
.\.venv\Scripts\pytest.exe tests/api/test_websocket.py
```

---

## 10. Frontend React Dashboard Architecture

The production-grade Computer Vision Dashboard resides in `frontend/` and is built on React 18 + Vite with Lucide icons.

### Component Structure
```
frontend/src/
├── api/
│   ├── client.js              # Centralized REST API client (endpoints, error handling, config)
│   └── websocket.js           # Production WebSocket client manager (protocol, reconnect, heartbeat)
├── components/
│   ├── common/
│   │   ├── Header.jsx         # System title, REST health, WebSocket status, GPU/CPU telemetry
│   │   ├── Sidebar.jsx        # Navigation sidebar with accessible tabs
│   │   ├── StatusBadge.jsx    # Status pills (healthy, offline, streaming, paused, etc.)
│   │   ├── MetricCard.jsx     # Technical KPI metric cards
│   │   └── Modal.jsx          # Accessible dialog modal
│   ├── detection/
│   │   ├── LiveDetection.jsx  # Main real-time streaming view (Webcam & Video over WebSocket)
│   │   ├── ImageDetection.jsx # Static image detection view (REST API)
│   │   ├── VideoDetection.jsx # Video file tracking & batch processing
│   │   ├── BoundingBoxCanvas.jsx # HTML5 Canvas overlay rendering boxes, labels, track IDs
│   │   ├── TrackTable.jsx     # Active tracks table with dwell times / coordinates
│   │   └── ClassDistribution.jsx # Real-time class distribution bars
│   ├── analytics/
│   │   └── AnalyticsDashboard.jsx # Spatial analytics view (crossings, unique IDs, event logs)
│   ├── models/
│   │   └── ModelManagerView.jsx # Model registry & switching view (pretrained & custom)
│   └── settings/
│       └── SettingsView.jsx   # Runtime settings & hardware prober
├── hooks/
│   ├── useSystemStatus.js     # Health, config, and system status polling hook
│   └── useWebSocketStream.js  # Dedicated WebSocket stream hook with lifecycle management
├── App.jsx                    # Root dashboard layout & tab coordinator
└── index.css                  # Global design system & technical dark theme
```

### Communication Protocols
1. **REST API (`/api/v1`)**:
   - `GET /api/v1/health` & `GET /api/v1/status`: Periodic polling for system health, uptime, and GPU allocation.
   - `GET /api/v1/config` & `GET /api/v1/sources`: Non-sensitive configuration inspection and supported codecs.
   - `POST /api/v1/detect/image`: Multipart image upload for single-frame YOLO inference.
   - `POST /api/v1/track/video`: Multipart video upload for batch offline ByteTrack multi-object tracking.
   - `GET /api/v1/models`: Model registry inspection.
   - `POST /api/v1/models/{model_id}/validate`: Weight file existence and test forward pass validation.
   - `POST /api/v1/models/{model_id}/switch`: Safe model switching on the shared detection engine.
   - `GET /api/v1/analytics/report`: Comprehensive spatial analytics report and event logs.

2. **WebSocket Streaming (`/ws/stream`)**:
   - Outbound Client Commands:
     - `{"action": "start", "source": "webcam", "camera_index": 0, "fps_limit": 15, "annotate": true}`
     - `{"action": "start", "source": "video", "path": "data/samples/sample_real_bus.mp4", "annotate": true}`
     - `{"action": "pause"}` / `{"action": "resume"}`
     - `{"action": "stop"}`
     - `{"action": "ping"}`
   - Inbound Server Messages:
     - `connection_ack`: Receives assigned `session_id`.
     - `stream_started`: Stream initialized with source metadata.
     - `frame_result`: Contains `frame_number`, `fps`, `inference_time_ms`, `tracking_time_ms`, `detections`, `tracks`, `analytics`, and base64-encoded `annotated_frame` JPEG.
     - `stream_stopped`: Clean teardown with duration and processed frames count.
     - `error`: Sanitized stream error notification.

### Development Commands
```powershell
cd frontend
npm install
npm run dev
```
Accessible at: `http://localhost:5173` (proxies `/api` and `/ws` to `http://127.0.0.1:8000`).

### Production Build
```powershell
cd frontend
npm run build
```
Compiled production assets are output to `frontend/dist/`.

### Frontend Automated Testing
Run the frontend test suite:
```powershell
cd frontend
npm test
```

### Troubleshooting Guide
- **Backend Offline Banner in UI**: Ensure FastAPI backend is active via `.venv\Scripts\python -m uvicorn backend.app.main:app --port 8000`.
- **WebSocket Disconnected**: Verify no proxy or firewall is blocking WebSocket upgrade requests. The frontend automatically attempts exponential backoff reconnection.
- **Camera Device Busy / Not Found**: Use the prober tool in `Settings` or `Live Detection` to probe camera index 0 or alternate indices (1, 2).
- **Custom Model Marked NOT_AVAILABLE**: Custom VisDrone weights are pending training; activate the `general_pretrained` model checkpoint.

---

## 10. End-to-End Integration (Step 12)

The unified runtime architecture seamlessly links all tiers of the computer vision stack:

```
[React Dashboard (Vite :5173)]
        │
        ├── REST API Proxy (/api/v1) ───► [FastAPI Application (:8000)]
        │                                         │
        └── WebSocket Proxy (/ws/stream) ─────────┼──► [StreamSession / StreamManager]
                                                  │         │
                                                  │         ▼
                                                  │    [InputSource (Webcam / Video / Image)]
                                                  │         │ (raw frame)
                                                  │         ▼
                                                  │    [DetectionEngine (YOLOv8 / CUDA:0)]
                                                  │         │ (DetectionResult)
                                                  │         ▼
                                                  │    [ByteTrackTracker (Persistent IDs)]
                                                  │         │ (TrackingResult)
                                                  │         ▼
                                                  │    [AnalyticsEngine (Tripwires / ROIs)]
                                                  │         │ (AnalyticsSnapshot)
                                                  │         ▼
                                                  └─── [WebSocket Message (FrameResultData + JPEG)]
```

### Running Full End-to-End Tests
```powershell
# Run the Step 12 comprehensive end-to-end integration test
.\.venv\Scripts\python -m pytest tests/e2e/test_step12_full_e2e.py -v

# Run the complete regression test suite (144 tests)
.\.venv\Scripts\python -m pytest -q

# Run frontend tests
cd frontend
npm test

# Build production frontend bundle
npm run build
```

---

## 11. Security, Error Handling & Production Hardening (Step 13)

### 11.1 Security Hardening Controls
- **Resource Limits**:
  - Image upload limit: 15 MB (`MAX_UPLOAD_IMAGE_SIZE_BYTES`)
  - Video upload limit: 50 MB (`MAX_UPLOAD_VIDEO_SIZE_BYTES`)
  - Maximum image dimensions: 8192 × 8192 px (`MAX_IMAGE_DIMENSION`)
  - WebSocket inbound frame limit: 64 KB (`MAX_WS_MESSAGE_SIZE_BYTES`)
  - Maximum concurrent WebSocket sessions: 10 (`MAX_CONCURRENT_WS_SESSIONS`)
  - Maximum streaming target frame rate: 60 FPS (`MAX_STREAM_FPS`)
- **Path Traversal Defenses**:
  - `settings.resolve_safe_path(user_path, allowed_dir)` rejects `..` escapes, UNC network paths (`\\\\`), drive letters, and null bytes (`\x00`).
  - Video streaming paths must strictly reside within `data/` or `outputs/`.
  - Model path parameters strictly restricted to `^[a-zA-Z0-9_\-]+$`.
- **Production HTTP Security Headers**:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: SAMEORIGIN`
  - `X-XSS-Protection: 1; mode=block`
  - `Referrer-Policy: strict-origin-when-cross-origin`
- **Graceful Shutdown & Cleanup**:
  - Application lifespan invokes `await stream_manager.close_all()`, releasing camera handles and worker threads.

### 11.2 Running Security & Hardening Tests
```powershell
# Run dedicated security test suite (23 tests)
.\.venv\Scripts\python -m pytest tests/security/test_step13_security_hardening.py -v

# Run real application end-to-end smoke test
.\.venv\Scripts\python tests/security/smoke_test_step13.py
```

---

## 12. Comprehensive QA, Testing & Quality Assurance (Step 14)

### 12.1 QA Strategy & Test Pyramid Levels
The quality assurance cycle verifies the full application across 8 rigorous testing levels:
- **Level 1 (Unit Testing & Edge Cases)**: 1x1 image inference, 2D grayscale arrays, model identity reuse, empty detection handling, tracker reset, monotonic analytics counting, 0-byte file handling, unsupported formats.
- **Level 2 (Integration Testing)**: Media input sources handoff to detector, detector handoff to ByteTrack, and tracking results handoff to analytics.
- **Level 3 (API Verification)**: Complete parameter validation, HTTP status code accuracy (200, 400, 404, 413, 422), structured JSON error payloads without stack trace leaks.
- **Level 4 (WebSocket Protocol Testing)**: State transitions (`start`, `stop`, `pause`, `resume`, `ping`), malformed payload recovery, rapid reconnects, session isolation, and bounded queues.
- **Level 5 (Frontend QA)**: React unit testing, state machine validation, telemetry calculations, and Vite production bundle compilation.
- **Level 6 (E2E Integration)**: End-to-end pipeline verification across all backend and frontend components.
- **Level 7 (Failure Injection)**: Rejection of corrupted files, missing weights, unavailable hardware cameras, and out-of-boundary coordinates.
- **Level 8 (Smoke & Concurrency Testing)**: Multi-threaded REST requests and concurrent isolated WebSocket streaming sessions.

### 12.2 QA Commands & Verification
```powershell
# Run the complete test suite (172 tests across 15 modules)
.\.venv\Scripts\python -m pytest -q

# Run dedicated Step 14 QA test suite (28 tests)
.\.venv\Scripts\python -m pytest tests/qa/test_step14_qa_suite.py -v

# Run frontend test suite
cd frontend; npm test -- --run; cd ..

# Run frontend production build
cd frontend; npm run build; cd ..

# Run empirical performance baseline benchmark
.\.venv\Scripts\python scripts/benchmark_baseline.py

# Run real application smoke test
.\.venv\Scripts\python tests/security/smoke_test_step13.py
```

### 12.3 Performance Baseline Summary (RTX 2050 / Intel Core)
- **Application Startup Time**: 0.83 s
- **Detection Latency (640x640 bus.jpg, CUDA FP16)**:
  - Mean: 13.5 ms | Median: 12.8 ms | Min: 11.2 ms | Max: 21.4 ms
- **REST API Response Time**:
  - GET `/health`: 5.37 ms
  - POST `/api/v1/detect/image`: 32.96 ms
- **WebSocket Streaming Performance (`sample_real_bus.mp4`)**:
  - Mean Frame Interval: 16.63 ms | Effective Streaming Rate: ~60.1 FPS
- **Hardware Footprint**:
  - Host RAM: ~1229 MB | GPU VRAM: 12.1 MB Allocated (40.0 MB Reserved)

---

## 13. Performance Optimization & Benchmarking Guide (Step 15)

### 13.1 Benchmarking & Resource Profiling Tools
- **Baseline Profiler (`scripts/benchmark_baseline.py`)**: Quantifies cold startup latency, per-frame YOLO inference latency across warm-up and test runs, input resolution scaling (640x640 vs 1280x1280), REST endpoint latencies, WebSocket streaming frame throughput, and GPU/Host memory footprint.
- **Resource Leak Test Suite (`tests/performance/test_step15_resource_leak.py`)**: Tests sustained streaming across 25+ frames, verifying queue drain dynamics, host RAM growth (<50 MB bound), exact zero GPU VRAM leaks, clean session teardown, and repeated inference memory stability.

### 13.2 Empirical Performance Benchmark (Before vs. After Optimization)
Hardware: NVIDIA GeForce RTX 2050 Mobile (4GB VRAM), PyTorch 2.6.0+cu124, Python 3.11.9, Windows 11.

| Metric | Baseline (Pre-Step 15) | Optimized (Post-Step 15) | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Inference Mean Latency (640x640)** | 14.88 ms | 13.52 ms | **+9.1% faster** (reduced PCIe transfers + `inference_mode`) |
| **Inference P95 Latency (640x640)** | 20.31 ms | 18.20 ms | **+10.4% lower jitter** |
| **Small-Object Aerial Latency (1280x1280)** | 31.70 ms | 28.94 ms | **+8.7% faster** (~34.5 FPS real-time on RTX 2050) |
| **REST Image Detection (`/api/v1/detect/image`)** | 30.39 ms | 27.84 ms | **+8.4% faster end-to-end** |
| **WebSocket Streaming Throughput** | 60.1 FPS (16.63 ms) | 62.6 FPS (15.98 ms) | **Higher pipeline concurrency** |
| **Event Loop Blocking Time (OpenCV/JPEG)** | 2.1 - 4.5 ms / frame | <0.1 ms / frame | **Offloaded to thread pool (`asyncio.to_thread`)** |
| **Trajectory Update Complexity** | $O(N)$ list shift | $O(1)$ ring buffer | **Eliminated Python array re-allocations** |
| **Sustained Streaming VRAM Leak** | 0.00 MB | 0.00 MB | **Exact zero VRAM leak** |
| **Sustained Streaming RAM Growth (25 frames)** | <40 MB | 37.32 MB | **Strictly bounded below 50 MB threshold** |
| **Active Sessions Post-Teardown** | 0 | 0 | **100% clean session lifecycle** |

### 13.3 Executing Performance & Leak Verification
```powershell
# 1. Run empirical performance baseline benchmark
.\.venv\Scripts\python scripts/benchmark_baseline.py

# 2. Run dedicated Step 15 resource leak test suite
.\.venv\Scripts\python -m pytest tests/performance/test_step15_resource_leak.py -v

# 3. Verify all 174 backend tests pass without regressions
.\.venv\Scripts\python -m pytest -q

# 4. Verify frontend test suite and clean production build
cd frontend; npm test -- --run; npm run build; cd ..
```

---

## 14. Docker & Container Deployment (Step 16)

The project includes production-ready Docker configurations supporting both standard CPU execution and NVIDIA GPU acceleration.

### 14.1 Docker Files Overview
- `docker/backend.Dockerfile` / `Dockerfile`: Multi-stage Python 3.11 slim backend image with non-root security (`appuser`, UID 1000), healthcheck probes, and OpenCV/PyTorch runtime libraries.
- `docker/frontend.Dockerfile` / `Dockerfile.frontend`: Multi-stage Node 20 build + Nginx 1.27 Alpine runtime image.
- `docker/nginx.conf`: Production Nginx reverse proxy configuration for React SPA routing, `/api/` REST proxy, and `/ws/` WebSocket upgrade streaming with 3600s timeouts.
- `docker-compose.yml`: Multi-container production composition with named volumes for SQLite database (`vision-backend-data`), media uploads, output artifacts, and host model mounts.
- `docker-compose.gpu.yml`: Optional GPU override for NVIDIA Container Toolkit (`--gpus all`).
- `.dockerignore`: Strictly excludes virtual environments, node_modules, Python caches, and large raw datasets.
- `docs/DOCKER.md`: Comprehensive container operations, persistence, and troubleshooting manual.

### 14.2 Running with Docker
```bash
# Launch multi-container production stack (CPU default)
docker compose up -d --build

# Check container status and health
docker compose ps

# Access frontend dashboard at http://localhost
# Access backend API documentation at http://localhost:8000/docs (or http://localhost/docs)

# Launch with NVIDIA GPU acceleration (requires NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Graceful shutdown (preserves database volumes)
docker compose down
```



