# REST & WebSocket API Specification

This document provides a comprehensive technical contract for all REST and WebSocket interfaces implemented in the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Global API Conventions

### Base URLs
- **Local Development REST:** `http://localhost:8000` (or `http://127.0.0.1:8000`)
- **Docker Production REST (via Nginx):** `http://localhost/api/v1` (or `http://localhost:8000/api/v1`)
- **Local WebSocket:** `ws://localhost:8000/ws/stream`
- **Docker WebSocket (via Nginx):** `ws://localhost/ws/stream`

### Interactive Documentation
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`

### Error Format
All HTTP error responses adhere to the standard JSON error schema:
```json
{
  "error": "Error Category Name",
  "detail": "Human-readable description of the error.",
  "status_code": 400,
  "timestamp": "2026-09-25T00:00:00.000000+00:00"
}
```

---

## 2. System & Health Endpoints

### 2.1 Root Health Check
- **Method:** `GET`
- **Path:** `/health` (also aliased under `/api/v1/health`)
- **Summary:** Probes service liveness. Used by Docker and Kubernetes health probes.
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "app_name": "Real-Time Object Detection & Tracking System",
    "timestamp": "2026-09-25T00:00:00.000000+00:00"
  }
  ```

### 2.2 System & Hardware Status
- **Method:** `GET`
- **Path:** `/status` (also aliased under `/api/v1/status`)
- **Summary:** Reports runtime environment, uptime, active model, and GPU VRAM utilization.
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "app_name": "Real-Time Object Detection & Tracking System",
    "environment": "development",
    "version": "1.0.0",
    "uptime_seconds": 142.5,
    "python_version": "3.11.9",
    "effective_device": "cuda:0",
    "gpu": {
      "device_name": "NVIDIA GeForce RTX 2050",
      "memory_allocated_mb": 12.14,
      "memory_reserved_mb": 40.0
    },
    "model": {
      "name": "yolov8n.pt",
      "classes_count": 80,
      "loaded": true
    },
    "timestamp": "2026-09-25T00:00:00.000000+00:00"
  }
  ```

### 2.3 Safe Runtime Configuration
- **Method:** `GET`
- **Path:** `/api/v1/config`
- **Summary:** Returns sanitized operational configuration (secrets and internal keys redacted).
- **Response (200 OK):**
  ```json
  {
    "app_name": "Real-Time Object Detection & Tracking System",
    "environment": "development",
    "debug": true,
    "api_prefix": "/api/v1",
    "active_model_id": "general_pretrained",
    "confidence_threshold": 0.35,
    "iou_threshold": 0.45,
    "max_detections": 100,
    "max_concurrent_ws_sessions": 10,
    "line_crossing_coords": [[0, 360], [1280, 360]]
  }
  ```

---

## 3. Media Ingestion & Source Endpoints

### 3.1 Sources Overview
- **Method:** `GET`
- **Path:** `/api/v1/sources`
- **Summary:** Details supported media ingestion modalities, allowed file formats, and upload limits.
- **Response (200 OK):**
  ```json
  {
    "supported_modalities": ["image", "video", "webcam"],
    "allowed_image_formats": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
    "allowed_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "max_image_upload_mb": 15.0,
    "max_video_upload_mb": 50.0,
    "max_image_dimension_px": 8192
  }
  ```

### 3.2 Probe Hardware Camera
- **Method:** `GET`
- **Path:** `/api/v1/sources/webcam/probe?camera_index=0`
- **Query Parameters:**
  - `camera_index` (int, default `0`): Hardware camera index to test.
- **Response (200 OK):**
  ```json
  {
    "camera_index": 0,
    "available": true,
    "backend": "cv2.CAP_DSHOW",
    "resolution": "640x480"
  }
  ```

---

## 4. Object Detection & Tracking Endpoints

### 4.1 Detect Objects in Image
- **Method:** `POST`
- **Path:** `/api/v1/detect/image`
- **Content-Type:** `multipart/form-data`
- **Request Form Fields:**
  - `file` (File, Required): Image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`). Max 15 MB.
  - `confidence` (float, Optional, 0.01–1.0): Overrides default model confidence threshold.
  - `iou` (float, Optional, 0.01–1.0): Overrides default NMS IoU threshold.
- **Response (200 OK):**
  ```json
  {
    "detections": [
      {
        "bbox": [17.0, 230.0, 802.0, 770.0],
        "confidence": 0.887,
        "class_id": 5,
        "class_name": "bus"
      },
      {
        "bbox": [50.0, 390.0, 245.0, 880.0],
        "confidence": 0.832,
        "class_id": 0,
        "class_name": "person"
      }
    ],
    "counts_by_class": {
      "bus": 1,
      "person": 3
    },
    "total_detections": 4,
    "image_width": 1080,
    "image_height": 720,
    "inference_time_ms": 13.52,
    "model_name": "yolov8n.pt",
    "timestamp": "2026-09-25T00:00:00.000000+00:00"
  }
  ```
- **Error Codes:**
  - `400 Bad Request`: Corrupted image or invalid format.
  - `413 Request Entity Too Large`: File exceeds 15 MB or dimensions exceed 8192px.

### 4.2 Track Objects in Video File
- **Method:** `POST`
- **Path:** `/api/v1/track/video`
- **Content-Type:** `multipart/form-data`
- **Request Form Fields:**
  - `file` (File, Required): Video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`). Max 50 MB.
  - `confidence` (float, Optional): Confidence filter threshold.
  - `iou` (float, Optional): NMS IoU threshold.
  - `stride` (int, default `1`): Process every $N$-th frame.
- **Response (200 OK):**
  ```json
  {
    "total_frames_processed": 30,
    "unique_objects_tracked": 4,
    "counts_by_class": {
      "bus": 1,
      "person": 3
    },
    "video_width": 648,
    "video_height": 864,
    "video_fps": 15.0,
    "processing_time_ms": 482.1,
    "mean_fps": 62.2,
    "tracks": [
      {
        "track_id": 1,
        "class_id": 5,
        "class_name": "bus",
        "first_frame": 0,
        "last_frame": 29,
        "total_detections": 30
      }
    ]
  }
  ```

---

## 5. Model Registry Management Endpoints

### 5.1 List Registered Models
- **Method:** `GET`
- **Path:** `/api/v1/models`
- **Response (200 OK):**
  ```json
  [
    {
      "model_id": "general_pretrained",
      "model_name": "YOLOv8 Nano (COCO Pretrained)",
      "model_type": "pretrained",
      "status": "AVAILABLE",
      "num_classes": 80,
      "input_size": 640,
      "is_active": true
    },
    {
      "model_id": "custom_visdrone",
      "model_name": "YOLOv8 Custom VisDrone Detector",
      "model_type": "custom",
      "status": "NOT_AVAILABLE",
      "num_classes": 10,
      "input_size": 1280,
      "is_active": false
    }
  ]
  ```

### 5.2 Get Active Model
- **Method:** `GET`
- **Path:** `/api/v1/models/active`
- **Response (200 OK):** Returns detailed model metadata for the currently loaded model.

### 5.3 Switch Active Model
- **Method:** `POST`
- **Path:** `/api/v1/models/{model_id}/switch`
- **Response (200 OK):**
  ```json
  {
    "status": "switched",
    "previous_model": "general_pretrained",
    "active_model": "general_pretrained",
    "message": "Model switched successfully."
  }
  ```
- **Error Codes:**
  - `404 Not Found`: Model ID not present in registry.
  - `400 Bad Request`: Model weights file is missing on disk (`status: NOT_AVAILABLE`).
  - `422 Unprocessable Entity`: Model weights file is corrupt or invalid.

---

## 6. Spatial Analytics Endpoints

### 6.1 Get Analytics Report
- **Method:** `GET`
- **Path:** `/api/v1/analytics/report?session_id=default_session`
- **Response (200 OK):**
  ```json
  {
    "session_id": "default_session",
    "total_unique_objects": 12,
    "active_objects": 4,
    "class_breakdown": {
      "car": 7,
      "person": 4,
      "truck": 1
    },
    "tripwire_crossings": {
      "default_line": {
        "forward": 5,
        "backward": 2,
        "total": 7
      }
    },
    "timestamp": "2026-09-25T00:00:00.000000+00:00"
  }
  ```

### 6.2 Reset Analytics Session
- **Method:** `POST`
- **Path:** `/api/v1/analytics/reset?session_id=default_session`
- **Response (200 OK):**
  ```json
  {
    "status": "reset",
    "session_id": "default_session",
    "message": "Analytics counters reset successfully."
  }
  ```

---

## 7. WebSocket Live Streaming Specification

### 7.1 Connection URL
- **Endpoint:** `/ws/stream` (also aliased under `/api/v1/ws/stream`)
- **Full URL:** `ws://localhost:8000/ws/stream` (or `ws://localhost/ws/stream` through Nginx)

### 7.2 Connection Handshake
Upon establishing the WebSocket connection, the backend transmits an acknowledgment:
```json
{
  "type": "connection_ack",
  "session_id": "8a3d1c4b-721f-4b02-98e3-0c1b72e5a8f4",
  "status": "connected",
  "server_time": "2026-09-25T00:00:00.000000+00:00"
}
```

### 7.3 Client Inbound Commands

#### Command: `start`
Initiates live video streaming from a webcam or video file:
```json
{
  "action": "start",
  "source": "video",
  "video_path": "data/samples/sample_real_bus.mp4",
  "camera_index": 0,
  "confidence": 0.35,
  "iou": 0.45,
  "fps_limit": 30.0,
  "enable_tracking": true,
  "enable_analytics": true
}
```

#### Command: `stop`
Halts stream processing and safely disposes of hardware/capture handles:
```json
{
  "action": "stop"
}
```

#### Command: `pause` / `resume`
Temporarily freezes or resumes the stream loop without disconnecting:
```json
{
  "action": "pause"
}
```

#### Command: `ping`
Heartbeat probe to keep the connection active through reverse proxies:
```json
{
  "action": "ping"
}
```

### 7.4 Server Outbound Stream Messages

#### Message: `frame_result`
Emitted for every analyzed frame at the configured `fps_limit`:
```json
{
  "type": "frame_result",
  "session_id": "8a3d1c4b-721f-4b02-98e3-0c1b72e5a8f4",
  "frame_number": 14,
  "timestamp_ms": 933.3,
  "frame_jpeg_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "detections": [
    {
      "bbox": [120.0, 80.0, 450.0, 390.0],
      "confidence": 0.912,
      "class_id": 5,
      "class_name": "bus"
    }
  ],
  "tracks": [
    {
      "track_id": 3,
      "bbox": [120.0, 80.0, 450.0, 390.0],
      "confidence": 0.912,
      "class_name": "bus",
      "trajectory": [[280, 230], [285, 235]]
    }
  ],
  "telemetry": {
    "inference_ms": 13.5,
    "tracking_ms": 1.2,
    "render_ms": 2.1,
    "effective_fps": 59.8,
    "active_tracks_count": 1,
    "cumulative_count": 4
  }
}
```

#### Message: `stream_stopped`
Emitted when an input stream terminates (EOF or user cancellation):
```json
{
  "type": "stream_stopped",
  "session_id": "8a3d1c4b-721f-4b02-98e3-0c1b72e5a8f4",
  "reason": "End of video file reached",
  "timestamp": "2026-09-25T00:00:00.000000+00:00"
}
```
