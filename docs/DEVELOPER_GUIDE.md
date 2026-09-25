# Developer & Engineering Guide

This guide provides technical onboarding for engineers extending, maintaining, or debugging the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Codebase Organization

```
Object-Detection-Tracking/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI application entrypoint & lifespan
│       ├── core/                    # Settings (Pydantic), logging & security
│       ├── db/                      # SQLAlchemy 2.0 ORM models & session
│       ├── api/
│       │   ├── v1/
│       │   │   ├── endpoints/       # REST controllers (health, detect, track, etc.)
│       │   │   └── router.py        # Centralized v1 routing
│       │   └── schemas/             # Pydantic v2 validation contracts
│       └── services/
│           ├── vision/              # YOLOv8 Detector, ByteTrack Tracker, ModelManager
│           ├── analytics/           # Spatial line-crossing tripwire & counter
│           └── streaming/           # WebSocket session manager & frame pipeline
├── frontend/
│   ├── src/
│   │   ├── main.jsx                 # React DOM mount point
│   │   ├── App.jsx                  # Main workspace layout & view router
│   │   ├── api/                     # REST & WebSocket client abstractions
│   │   └── components/              # Modular UI components (detection, analytics, etc.)
├── models/
│   ├── registry/models.yaml         # Authoritative model catalog
│   ├── pretrained/                  # Pretrained model weights
│   └── custom/                      # Custom trained domain checkpoints
├── data/                            # Samples, uploads, datasets & SQLite DB
├── docker/                          # Dockerfiles, Nginx configuration & requirements
├── tests/                           # Multi-level test pyramid (unit, API, E2E, QA)
└── docs/                            # Architecture, API & operational manuals
```

---

## 2. Key Architecture Patterns & Responsibilities

### 2.1 The Vision Pipeline
1. **`DetectionEngine` (`backend/app/services/vision/detector.py`):**
   - Decoupled from HTTP context.
   - Evaluates frames inside `torch.inference_mode()`.
   - Uses zero-copy PCIe extraction (`res.boxes.data.cpu().numpy()`) to minimize host-to-device synchronization.
   - Converts weights to native FP16 when running on CUDA.

2. **`ByteTrackTracker` (`backend/app/services/vision/tracker.py`):**
   - Consumes structured `DetectionResult` objects without re-running YOLO.
   - Implements two-stage association matching high-confidence detections first, followed by low-confidence recovery.
   - Maintains trajectories using bounded circular deques (`collections.deque(maxlen=max_trajectory)`) for $O(1)$ updates.

3. **`AnalyticsEngine` (`backend/app/services/analytics/counter.py`):**
   - Consumes `TrackingResult` instances.
   - Computes 2D vector cross-product geometry (`ccw()`) to register directional line crossings without external geometry libraries.

4. **`StreamSession` (`backend/app/services/streaming/session.py`):**
   - Manages client streaming lifecycle.
   - Offloads OpenCV rendering and JPEG compression to worker threads via `asyncio.to_thread()`, keeping the FastAPI ASGI event loop responsive.
   - Implements bounded queue backpressure (`asyncio.Queue(maxsize=10)`).

---

## 3. How-To Developer Recipes

### 3.1 Adding a New Pretrained or Custom Model
1. Place the weights checkpoint (`.pt` or `.onnx`) in `models/custom/` or `models/pretrained/`.
2. Register the model in `models/registry/models.yaml`:
   ```yaml
   models:
     my_new_model:
       model_id: my_new_model
       model_name: "YOLOv8 Medium Fine-Tuned"
       model_type: custom
       version: "1.0"
       framework: ultralytics_yolo
       task: object_detection
       weights_path: models/custom/my_new_model.pt
       input_size: 640
       recommended_confidence: 0.40
       recommended_iou: 0.45
       device: auto
       status: AVAILABLE
       num_classes: 80
       classes:
         0: person
         1: car
       notes: "Fine-tuned vehicle detector."
   ```
3. The new model will immediately appear in the React dashboard's **Model Registry** tab and can be switched dynamically via `POST /api/v1/models/my_new_model/switch`.

### 3.2 Adding a New Input Ingestion Modality
1. Open `backend/app/services/vision/input_sources.py`.
2. Inherit from the abstract base class `InputSource`:
   ```python
   class RTSPInput(InputSource):
       def __init__(self, rtsp_url: str):
           self.url = rtsp_url
           self.cap = None

       def __enter__(self):
           self.cap = cv2.VideoCapture(self.url)
           return self

       def __exit__(self, exc_type, exc_val, exc_tb):
           if self.cap:
               self.cap.release()

       def read(self) -> Optional[np.ndarray]:
           ret, frame = self.cap.read()
           return frame if ret else None
   ```
3. Update `backend/app/services/streaming/session.py` to route the new modality inside `_init_source()`.

### 3.3 Adding a New Analytics Spatial Rule
1. Open `backend/app/services/analytics/counter.py`.
2. Define the analytical condition inside `update(tracks)`:
   - Example: Polygon Zone ROI containment using ray-casting geometry.
   - Store metrics in memory dictionaries.
3. Expose the metric in `AnalyticsReportSchema` in `backend/app/api/schemas/analytics.py`.
4. Add the metric card to `frontend/src/components/analytics/AnalyticsDashboard.jsx`.

---

## 4. Testing & Verification

### Running Backend Tests
```powershell
# Run the complete test suite (182 tests)
.\.venv\Scripts\python -m pytest -q

# Run specific domain test module
.\.venv\Scripts\python -m pytest tests/unit/test_detector.py -v
.\.venv\Scripts\python -m pytest tests/unit/test_tracker.py -v
.\.venv\Scripts\python -m pytest tests/performance/test_step15_resource_leak.py -v
```

### Running Frontend Tests
```powershell
cd frontend
npm test -- --run
npm run build
cd ..
```

### Running Security & Application Smoke Tests
```powershell
.\.venv\Scripts\python tests/security/smoke_test_step13.py
```
