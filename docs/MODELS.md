# Model Architecture & Registry Documentation

This document covers model management, pretrained weights, custom domain-adapted models, and runtime inference configurations in the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Model Registry Architecture

All models are declaratively tracked in `models/registry/models.yaml`. The registry decouples model definition from application source code, enabling hot-switching without application downtime.

```
models/
├── registry/
│   └── models.yaml      # Declarative model configuration catalog
├── pretrained/          # Pretrained general-purpose checkpoints
│   └── yolov8n.pt       # (Active COCO detector, ~6.2 MB)
├── custom/              # Domain-adapted fine-tuned checkpoints
│   └── .gitkeep         # (Reserved for custom checkpoints)
├── weights/             # General model weights staging directory
└── metadata/            # Training metrics, PR curves & class taxonomies
```

---

## 2. Configured Models & Availability

| Model ID | Framework | Task | Classes | Input Size | Recommended Conf / IoU | Local Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`general_pretrained`** | Ultralytics YOLOv8 | Object Detection | 80 (COCO) | 640px | 0.35 / 0.45 | **`AVAILABLE` / `LOADED`** |
| **`custom_visdrone`** | Ultralytics YOLOv8 | Aerial Object Detection | 10 (VisDrone) | 1280px | 0.25 / 0.45 | **`NOT_AVAILABLE`** (Planned) |

> **Architectural Distinction:**
> - `general_pretrained` is physically present on disk (`yolov8n.pt`, 6.2 MB) and loaded by default.
> - `custom_visdrone` is cataloged and fully supported by the architecture and API, but its trained weights checkpoint is pending future training.

---

## 3. Dynamic Model Switching Lifecycle

The `ModelManager` service (`backend/app/services/vision/model_manager.py`) provides safe runtime switching:

1. **Client Request:** User triggers `POST /api/v1/models/{model_id}/switch`.
2. **Registry Lookup:** Verifies `model_id` exists in `models.yaml`.
3. **Physical File Validation:** Checks that `weights_path` exists on disk and resolves strictly inside project boundaries (`resolve_safe_path`).
4. **Extension Verification:** Enforces supported formats: `.pt` (PyTorch), `.onnx` (ONNX Runtime), `.engine` (TensorRT).
5. **Memory Teardown:** Disposes previous model, invokes `torch.cuda.empty_cache()` to flush GPU VRAM, and initializes new `DetectionEngine`.
6. **Persistence:** Updates active configuration state in memory.

---

## 4. Resolution & Small-Object Detection Tuning

The platform supports dynamic input resolution scaling configured via `models.yaml` or runtime parameters:

- **Standard Surveillance (`imgsz: 640`):**
  - **Latency:** 13.52 ms on RTX 2050 Mobile (CUDA FP16).
  - **Throughput:** ~74 FPS.
  - **Best for:** Ground cameras, vehicles, pedestrians, indoor surveillance.
- **Aerial / Drone Surveillance (`imgsz: 1280`):**
  - **Latency:** 28.94 ms on RTX 2050 Mobile (CUDA FP16).
  - **Throughput:** ~34.5 FPS (real-time >30 FPS).
  - **Best for:** High-altitude drone footage, small targets ($<32\times32$ px), crowded traffic scenes.

---

## 5. Hardware Device Acceleration & Fallback

- **NVIDIA CUDA Acceleration (`cuda:0`):**
  - Evaluates models in native FP16 half-precision (`model.model.half()`).
  - Single contiguous PCIe coordinate extraction (`res.boxes.data.cpu().numpy()`).
- **CPU Fallback (`cpu`):**
  - Automatically activated if CUDA is unavailable or when running in CPU Docker containers.
  - FP16 is automatically disabled on CPU to prevent half-precision software emulation overhead.
