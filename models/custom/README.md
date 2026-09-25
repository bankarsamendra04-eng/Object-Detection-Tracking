# Custom Models Directory

This directory is designated for domain-adapted and custom-trained model checkpoints.

## Target Custom Models
- **YOLOv8 VisDrone Detector (`models/custom/yolov8_visdrone.pt`)**:
  - Target Task: Aerial Drone Object Detection (10 VisDrone classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor).
  - Target Input Resolution: 1280×1280 (High resolution for small aerial targets).
  - Status: **Pending Step 11 Training** (Not currently present on disk).

## Model Placement
When a custom model is trained or exported:
1. Save the PyTorch `.pt` file here (e.g. `models/custom/yolov8_visdrone.pt`).
2. Register the model in `models/registry/models.yaml` with its class mapping, recommended confidence, and input size.
3. Use the `ModelManager` or API `/api/v1/models/custom_visdrone/validate` to verify weight integrity before switching.
