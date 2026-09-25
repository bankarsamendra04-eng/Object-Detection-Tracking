# User Guide: Object Detection & Tracking Dashboard

This guide walks you through using the **Real-Time Object Detection & Tracking Platform** from an end-user perspective.

---

## 1. Launching the System

### Option A: Local Development Server
1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd D:\Object-Detection-Tracking
   .\.venv\Scripts\Activate.ps1
   ```
2. Start the FastAPI backend server:
   ```powershell
   python scripts/run_backend.py
   ```
3. Start the React frontend development server in a separate terminal:
   ```powershell
   cd frontend
   npm run dev
   ```
4. Open your browser to `http://localhost:5173`.

### Option B: Docker Multi-Container Stack
1. Ensure Docker Desktop is running.
2. Launch the stack from the repository root:
   ```bash
   docker compose up -d
   ```
3. Open your browser directly to `http://localhost`.

---

## 2. Navigating the Dashboard

The dashboard provides six dedicated navigation tabs located in the left sidebar:

| View | Purpose | Primary Controls |
| :--- | :--- | :--- |
| **Live Stream** | Real-time WebSocket streaming from webcam or video file. | Source selector, Play/Pause/Stop, FPS limiter, tracking toggle. |
| **Image Detection** | Single high-resolution image analysis. | Drag-and-drop file upload, Confidence and IoU threshold sliders. |
| **Video Analysis** | Batch analysis and tracking on uploaded video files. | Video file picker, frame stride selector, tracking table. |
| **Spatial Analytics** | Real-time line-crossing tripwires and class breakdowns. | Tripwire coordinate display, reset counter button, telemetry charts. |
| **Model Registry** | Model inspection and runtime model hot-switching. | Model cards, active model badge, validation button, switch button. |
| **System Settings** | Hardware telemetry, device selection, and operational logs. | CUDA/CPU monitor, memory meters, API prefix, and reload controls. |

---

## 3. Using Live WebSocket Streaming (`Live Stream`)

1. Navigate to the **Live Stream** tab in the sidebar.
2. In the **Stream Configuration** panel:
   - **Source Type:** Select `Webcam` for hardware cameras or `Video File` for pre-recorded media.
   - **Camera Index:** If using `Webcam`, select camera index (`0` for primary webcam). Click **Probe Camera** to verify device availability.
   - **Video Path:** If using `Video File`, enter a valid local path (e.g. `data/samples/sample_real_bus.mp4`).
   - **Target FPS:** Set the target frame rate (default: `30 FPS`).
   - **Enable Tracking:** Check to activate ByteTrack multi-object tracking.
3. Click **Start Stream**.
4. The live video canvas displays real-time annotated frames:
   - Green bounding boxes indicate active tracks with persistent IDs (e.g., `Car #4 0.89`).
   - Motion trajectories trace historical centroid paths.
   - Real-time telemetry in the header displays actual FPS, inference latency (ms), tracking latency (ms), and active entity counts.
5. Click **Pause** to freeze the stream without losing state, or click **Stop** to release camera handles cleanly.

---

## 4. Analyzing Images (`Image Detection`)

1. Click **Image Detection** in the sidebar.
2. Drag and drop an image file (`.jpg`, `.png`, `.webp`, `.bmp`) or click to browse. A sample image is available at `data/samples/bus.jpg`.
3. Adjust detection parameters:
   - **Confidence Threshold:** Filters detections below this probability (default: `0.35`).
   - **IoU Threshold:** Controls Non-Maximum Suppression overlap filtering (default: `0.45`).
4. Click **Run Detection**.
5. View results:
   - The interactive canvas renders labeled bounding boxes.
   - The **Class Distribution** panel tallies total detected entities by category.
   - Total inference latency is displayed in milliseconds.

---

## 5. Tracking Video Files (`Video Analysis`)

1. Click **Video Analysis** in the sidebar.
2. Upload a video file (`.mp4`, `.avi`, `.mov`, `.webm`).
3. Configure the **Frame Stride** (e.g., `1` for every frame, `2` for every second frame to accelerate processing).
4. Click **Process Video**.
5. Once processing finishes:
   - The **Track Registry Table** displays every unique entity tracked across the video with its persistent ID, class name, first seen frame, and last seen frame.
   - Cumulative object totals are automatically synchronized with the Analytics Engine.

---

## 6. Monitoring Spatial Analytics (`Spatial Analytics`)

1. Click **Spatial Analytics** in the sidebar.
2. The dashboard displays real-time spatial metrics:
   - **Cumulative Unique Entities:** Total distinct objects tracked during the active session.
   - **Active Objects:** Currently visible tracked objects.
   - **Virtual Line Crossing Tripwire:** Displays bidirectional crossing tallies (Forward / Backward) across configured virtual boundary lines.
3. Click **Reset Analytics** to flush active counters and begin a new monitoring session.

---

## 7. Managing Models (`Model Registry`)

1. Click **Model Registry** in the sidebar.
2. The page lists all configured models from `models/registry/models.yaml`:
   - **YOLOv8 Nano (COCO Pretrained):** Standard 80-class general-purpose detector (`AVAILABLE`).
   - **Custom VisDrone Detector:** Specialized 10-class aerial drone detector (`NOT_AVAILABLE` until weights are placed in `models/custom/`).
3. Click **Switch Model** to dynamically switch the active detector. The backend updates the detector instance and flushes GPU cache without restarting the server.
