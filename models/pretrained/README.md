# Pretrained Models Directory

This directory stores standard pretrained detection checkpoints trained on general-purpose datasets (such as MS COCO).

## Default Pretrained Models
- **YOLOv8 Nano (`yolov8n.pt`)**:
  - Task: Object Detection (80 COCO classes)
  - Parameter Count: ~3.2M
  - Input Resolution: 640×640
  - Recommended Usage: Real-time edge inference, fast webcam tracking, testing.
  - License: AGPL-3.0 (Ultralytics)

## Storage Convention
Pretrained weights may be placed directly in this directory or referenced from the project root / `models/weights/`. The `ModelManager` checks the configured relative paths and automatically resolves their absolute paths on the local filesystem.
