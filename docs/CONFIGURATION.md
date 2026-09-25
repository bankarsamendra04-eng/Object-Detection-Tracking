# System Configuration & Environment Reference

This document provides a complete reference for all operational settings and environment variables in the **Real-Time Object Detection & Tracking Platform**.

All settings are managed via Pydantic Settings (`backend/app/core/config.py`) and can be overridden via `.env` files or system environment variables.

---

## 1. Server & Runtime Settings

| Variable | Type | Default | Description | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `APP_NAME` | str | `"Real-Time Object Detection & Tracking System"` | Human-readable application title shown in OpenAPI and UI. | Optional |
| `APP_ENV` | str | `"development"` | Environment mode: `development`, `staging`, `production`. | Optional |
| `DEBUG` | bool | `true` | Enables verbose debug logging and traceback outputs. | Optional |
| `HOST` | str | `"0.0.0.0"` | Network bind address for the Uvicorn ASGI server. | Optional |
| `PORT` | int | `8000` | Port for the backend REST and WebSocket server. | Optional |
| `API_V1_PREFIX` | str | `"/api/v1"` | URL prefix for version 1 REST routes. | Optional |
| `CORS_ORIGINS` | JSON list | `["http://localhost:3000","http://localhost:5173"]` | Authorized origin URLs allowed to make cross-origin browser requests. | Optional |

---

## 2. Computer Vision & Inference Settings

| Variable | Type | Default | Description | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `ACTIVE_MODEL_ID` | str | `"general_pretrained"` | Model identifier matching an entry in `models/registry/models.yaml`. | Optional |
| `MODEL_NAME` | str | `"yolov8n.pt"` | Checkpoint filename or registered model key. | Optional |
| `MODELS_DIR` | str | `"models"` | Base directory containing model weights and registry. | Optional |
| `MODELS_REGISTRY_PATH` | str | `"models/registry/models.yaml"` | Path to the authoritative YAML model registry. | Optional |
| `DEVICE` | str | `"auto"` | Hardware compute device: `auto`, `cuda:0`, or `cpu`. Auto selects CUDA if available. | Optional |
| `CONFIDENCE_THRESHOLD` | float | `0.35` | Minimum detection confidence score (0.01 to 1.0). | Optional |
| `IOU_THRESHOLD` | float | `0.45` | Non-Maximum Suppression (NMS) intersection-over-union threshold. | Optional |
| `HALF_PRECISION` | bool | `false` | Enables FP16 half-precision on CUDA devices for accelerated inference. | Optional |
| `MAX_DETECTIONS` | int | `100` | Maximum number of bounding boxes retained per frame. | Optional |
| `INFERENCE_IMG_SIZE` | int | `640` | Square input resolution: `640` (surveillance) or `1280` (aerial/small objects). | Optional |

---

## 3. Multi-Object Tracking Settings (ByteTrack)

| Variable | Type | Default | Description | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `TRACKER_TYPE` | str | `"bytetrack.yaml"` | Tracker algorithm configuration preset. | Optional |
| `TRACK_HIGH_THRESH` | float | `0.5` | Minimum confidence score for primary stage-1 data association. | Optional |
| `TRACK_LOW_THRESH` | float | `0.1` | Floor confidence score for secondary stage-2 occlusion recovery. | Optional |
| `TRACK_MATCH_THRESH`| float | `0.8` | Minimum IoU overlap required to validate a track association. | Optional |
| `TRACK_PERSISTENCE_BUFFER` | int | `30` | Number of consecutive missed frames before a lost track is removed. | Optional |

---

## 4. Storage & Persistence Tier

| Variable | Type | Default | Description | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `DATA_DIR` | str | `"data"` | Base operational data directory. | Optional |
| `UPLOADS_DIR` | str | `"data/uploads"` | Target folder for incoming user media uploads. | Optional |
| `OUTPUTS_DIR` | str | `"outputs"` | Destination folder for exported recordings and analytical artifacts. | Optional |
| `LOGS_DIR` | str | `"logs"` | Directory for rotating application log files (`application.log`). | Optional |
| `DATABASE_URL` | str | `"sqlite:///./data/detection_tracking.db"` | SQLAlchemy database connection URI. Supports SQLite and PostgreSQL. | Optional |
| `DB_ECHO` | bool | `false` | Logs all raw SQL queries to console when enabled. | Optional |

---

## 5. Security & Resource Limits

| Variable | Type | Default | Description | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `MAX_UPLOAD_IMAGE_SIZE_BYTES` | int | `15728640` (15 MB) | Maximum permitted size for uploaded image files. | Optional |
| `MAX_UPLOAD_VIDEO_SIZE_BYTES` | int | `52428800` (50 MB) | Maximum permitted size for uploaded video files. | Optional |
| `MAX_IMAGE_DIMENSION` | int | `8192` | Maximum image width or height in pixels (decompression bomb protection). | Optional |
| `MAX_CONCURRENT_WS_SESSIONS`| int | `10` | Maximum simultaneous live WebSocket streaming sessions. | Optional |
| `MAX_WS_MESSAGE_SIZE_BYTES` | int | `65536` (64 KB) | Maximum payload size for client-to-server WebSocket messages. | Optional |
| `MAX_STREAM_FPS` | float | `60.0` | Maximum frame rate cap for live streaming sessions. | Optional |

---

## 6. Spatial Analytics Settings

| Variable | Type | Default | Description | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `LINE_CROSSING_COORDS` | JSON list | `[[0, 360], [1280, 360]]` | Default virtual tripwire coordinate segment `[[x1, y1], [x2, y2]]`. | Optional |
| `TARGET_CLASSES` | JSON list | `[]` | Filter target categories for analytics (empty list tracks all detected classes). | Optional |
