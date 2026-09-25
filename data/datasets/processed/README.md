# Processed Datasets Storage

This directory holds sanitized, training-ready datasets in YOLO format:

```
processed/
└── visdrone/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

## Annotation Format (YOLO Normalized)
Each `.txt` label file corresponds to an image of the same basename:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where:
- `class_id`: Integer index `[0, num_classes - 1]`
- `x_center`, `y_center`, `width`, `height`: Float values normalized to `[0.0, 1.0]` relative to image width and height.
