# Dataset Management Pipeline Scripts

This suite of scripts provides an end-to-end automated dataset workflow:

1. `dataset_registry.py`: Inspect, query, and update the declarative dataset catalog (`data/datasets/registry/datasets.yaml`).
2. `download_dataset.py`: Robust, safe downloading with resume support, disk-space pre-checks, progress display, and ZipSlip-protected extraction.
3. `convert_annotations.py`: Normalizes VisDrone and custom annotations to standard YOLO format, handling ignored categories and coordinate bounding box normalization.
4. `validate_dataset.py`: Comprehensive dataset sanity and integrity verification: image readability, coordinate bounds, class indices, split overlap leakage detection, and small-object distribution statistics.
5. `inspect_dataset.py`: Visual and statistical inspection utility generating dataset distribution summaries and visual bounding box overlays.
