# Model Repository Architecture

This directory serves as the centralized model management repository for the Real-Time Object Detection & Tracking Platform.

## Directory Structure

```
models/
├── README.md               # Main model repository documentation
├── weights/                # Legacy/default weights directory (e.g., yolov8n.pt)
├── pretrained/             # Standard general-purpose pretrained models (COCO)
│   └── README.md
├── custom/                 # Specialized domain-specific models (VisDrone, aerial)
│   ├── README.md
│   └── .gitkeep
├── registry/               # Declarative model registry configurations
│   └── models.yaml
└── metadata/               # Class mappings, benchmark metrics, validation specs
    └── README.md
```

## Supported Model Formats
- **Ultralytics YOLO PyTorch Checkpoints (`.pt`)**: Primary format for YOLOv8/v9/v11 models.
- **ONNX (`.onnx`)**: Optimized cross-platform runtime weights.
- **TensorRT (`.engine`)**: High-performance GPU inference engine (where supported).

## Security & Path Integrity
All model loading is strictly mediated through the `ModelManager` and declarative `models.yaml` registry. Arbitrary user-supplied filesystem paths are blocked to prevent arbitrary code execution via untrusted pickle/PyTorch payloads.
