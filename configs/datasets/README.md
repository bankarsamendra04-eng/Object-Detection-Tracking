# Dataset Configurations

This directory contains YAML configurations compatible with Ultralytics YOLO dataset definitions.

## Datasets
- `visdrone.yaml`: Configuration for the VisDrone2019-DET benchmark dataset.
- `custom.yaml`: Template configuration for custom user datasets.

## Environment Variable Overrides
The base path for datasets can be specified dynamically via the `DATASET_DIR` environment variable, or resolved relative to the detected project root.
