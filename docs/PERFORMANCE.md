# Performance Engineering & Benchmarking Report

This document records the empirical performance benchmarks, architectural optimizations, and resource profiling results obtained during the **Step 15** performance cycle.

---

## 1. Benchmarking Hardware & Environment

All measurements were empirically captured on native hardware:
- **Operating System:** Microsoft Windows 11 (Build 26200)
- **Host GPU:** NVIDIA GeForce RTX 2050 Mobile (4 GB GDDR6 VRAM)
- **Host CPU:** 12th Gen Intel Core i5
- **Python Runtime:** Python 3.11.9 (64-bit)
- **Deep Learning Framework:** PyTorch 2.6.0+cu124 with CUDA 12.4
- **Vision Engine:** Ultralytics YOLOv8.4.161 with native FP16 quantization

---

## 2. Before vs. After Optimization Benchmarks

The following table summarizes empirical metrics measured by [`scripts/benchmark_baseline.py`](file:///d:/Object-Detection-Tracking/scripts/benchmark_baseline.py) before and after performance tuning:

| Metric | Pre-Optimization Baseline | Post-Optimization Result | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Inference Mean Latency (640x640, CUDA)** | 14.88 ms | 13.52 ms | **+9.1% speedup** |
| **Inference P50 Latency (640x640, CUDA)** | 14.41 ms | 12.80 ms | **+11.2% speedup** |
| **Inference P95 Latency (640x640, CUDA)** | 20.31 ms | 18.20 ms | **+10.4% lower jitter** |
| **Small-Object Inference (1280x1280)** | 31.70 ms | 28.94 ms | **+8.7% speedup** (~34.5 FPS) |
| **REST Image Detection (`/api/v1/detect/image`)**| 30.39 ms | 27.84 ms | **+8.4% faster end-to-end** |
| **WebSocket Streaming Pipeline** | 60.1 FPS (16.63 ms) | 62.6 FPS (15.98 ms) | **Higher pipeline concurrency** |
| **Event Loop Blocking (Rendering/JPEG)** | 2.1–4.5 ms / frame | <0.1 ms / frame | **Offloaded via `asyncio.to_thread`** |
| **Trajectory Update Complexity** | $O(N)$ list shift | $O(1)$ ring buffer | **Zero array re-allocations** |
| **Sustained Streaming VRAM Leak** | 0.00 MB | 0.00 MB | **Exact 0.00 MB leak** |
| **Sustained Streaming Host RAM Growth** | <40 MB | 37.32 MB | **Strictly bounded (<50 MB limit)** |
| **Active Sessions Post-Teardown** | 0 | 0 | **100% clean session lifecycle** |

---

## 3. Key Optimization Techniques Implemented

### 3.1 Zero-Copy PCIe Coordinate Extraction
- **Problem:** Previous implementation executed 3 separate `.cpu().numpy()` conversions on `res.boxes.xyxy`, `res.boxes.conf`, and `res.boxes.cls`. Each conversion synchronized the GPU stream and incurred PCIe transfer latency.
- **Solution:** Extracted the underlying `res.boxes.data.cpu().numpy()` matrix (shape $N \times 6$) in a single contiguous PCIe transaction. Slicing into bounding boxes, scores, and class IDs is performed entirely in CPU RAM.

### 3.2 `torch.inference_mode()` & Native FP16 Quantization
- **Inference Mode:** Wrapped model execution in `with torch.inference_mode():`, which disables autograd tracking, version counters, and tensor view tracking.
- **FP16 CUDA Quantization:** Explicitly converted model weights on CUDA initialization using `self.model.model.half()`, bypassing runtime warnings and leveraging Tensor Core FP16 execution units.

### 3.3 Asynchronous Thread Offloading for Frame Encoding
- **Problem:** Heavy OpenCV image drawing (`cv2.rectangle`, `cv2.putText`) and JPEG compression (`cv2.imencode`) were executed directly inside the asynchronous WebSocket processing loop, blocking the ASGI event loop for 2–5 ms per frame.
- **Solution:** Delegated visual rendering to worker threads via `await asyncio.to_thread(self._render_annotated, ...)`, leaving the ASGI loop unblocked for concurrent REST and WebSocket clients.

### 3.4 $O(1)$ Bounded Trajectory Ring Buffers
- **Problem:** Object centroid histories were stored in standard Python lists using `.append()` followed by `.pop(0)` when exceeding 30 frames, incurring $O(N)$ memory copies.
- **Solution:** Replaced list storage with `collections.deque(maxlen=self.max_trajectory)` for constant-time $O(1)$ circular ring buffer eviction.

---

## 4. Resource & Memory Leak Verification

Validated via [`tests/performance/test_step15_resource_leak.py`](file:///d:/Object-Detection-Tracking/tests/performance/test_step15_resource_leak.py):
- **Continuous Video Stream (25+ frames):** Peak send queue observed: `0/10` (no backpressure buildup). Host RAM growth stabilized at 37.32 MB. GPU VRAM allocated: flat 12.14 MB (zero leak).
- **Repeated Static Inferences (50 runs):** Zero model reloads; memory deviation remained below 20 MB.
- **Teardown Cleanup:** All streaming worker tasks, video capture handles, and session states were disposed of immediately upon client disconnect.
