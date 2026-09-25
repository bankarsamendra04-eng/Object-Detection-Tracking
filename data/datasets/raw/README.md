# Raw Datasets Storage

This directory contains original, un-mutated dataset archives and raw annotations.

## Structure
- `visdrone/`: VisDrone benchmark dataset hierarchy
  - `detection/`: Image object detection archives (`VisDrone2019-DET-val`, `VisDrone2019-DET-train`)
  - `video/`: Video object detection (Task 2)
  - `mot/`: Multi-object tracking (Task 4)
  - `sot/`: Single-object tracking (Task 3)
  - `crowd/`: Crowd counting (Task 5)
- `custom/`: User-provided raw images and external annotation formats

## Invariance Principle
Raw data stored here must NEVER be directly edited, overwritten, or modified in-place. All transformations (coordinate normalization, format conversion, split redistribution) write to `data/datasets/processed/`.
