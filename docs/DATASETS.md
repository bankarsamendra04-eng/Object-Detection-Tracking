# Dataset Architecture & Pipeline Documentation

This document describes the dataset ingestion pipeline, normalization formats, validation tools, and current storage status for the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Dataset Architecture & Storage Layout

Datasets are managed under an isolated, structured hierarchy (`data/datasets/`):

```
data/datasets/
├── registry/
│   └── datasets.yaml        # Authoritative catalog of dataset sources and metadata
├── raw/                     # Original, immutable downloads and annotations
│   └── visdrone/
│       └── detection/
│           └── VisDrone2019-DET-val/    # (548 validation images + annotations)
├── processed/               # Normalized YOLO coordinate labels
│   └── visdrone/
│       ├── images/val/
│       └── labels/val/
└── samples/                 # Sample images and videos used for smoke tests
    ├── images/
    └── videos/
```

### Storage Isolation & Git Exclusion Policy
- **Git Invariance:** Raw dataset files (`.zip`, `.tar`, raw image folders) are strictly excluded from version control via `.gitignore` and `.dockerignore`.
- **Reproducibility:** All datasets are cataloged in `data/datasets/registry/datasets.yaml` with upstream source URLs, licenses, taxonomies, and MD5 hashes.

---

## 2. Dataset Registry & Current Status

The authoritative catalog tracks all dataset modalities and lifecycle states (`PLANNED`, `DOWNLOADING`, `DOWNLOADED`, `PROCESSED`, `VALIDATED`, `READY`):

| Dataset ID | Modality | Tasks | Local Status | Images | Storage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`visdrone_det_val`** | Aerial Images | Small Object Detection | **`READY`** | 548 | ~268 MB (Raw + Processed) |
| **`visdrone_det_train`**| Aerial Images | Small Object Detection | **`PLANNED`** | — (Pending future training) | ~1.4 GB (Not downloaded) |
| **`visdrone_mot`** | Aerial Video | Multi-Object Tracking | **`PLANNED`** | — (Pending future training) | ~2.5 GB (Not downloaded) |
| **`visdrone_sot`** | Aerial Video | Single-Object Tracking | **`PLANNED`** | — (Pending future training) | ~3.0 GB (Not downloaded) |

> **Important Note:** In accordance with repository constraints, only the **VisDrone 2019 Detection Validation Split (`visdrone_det_val`)** is downloaded and processed locally. Large training and video datasets remain cataloged but are not committed to Git.

---

## 3. Annotation Conversion & YOLO Normalization

VisDrone annotations provide bounding boxes in CSV format:
`[bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion]`

The automated converter (`scripts/dataset/convert_visdrone.py`) transforms them into standard normalized YOLO format:
`[class_id x_center y_center width height]`

Normalized coordinates satisfy:
$$x_{\text{center}} = \frac{\text{bbox\_left} + \frac{\text{bbox\_width}}{2}}{\text{image\_width}}, \quad y_{\text{center}} = \frac{\text{bbox\_top} + \frac{\text{bbox\_height}}{2}}{\text{image\_height}}$$
$$w = \frac{\text{bbox\_width}}{\text{image\_width}}, \quad h = \frac{\text{bbox\_height}}{\text{image\_height}}$$

### VisDrone 10-Class Taxonomy Remapping
1. `0: pedestrian`
2. `1: people`
3. `2: bicycle`
4. `3: car`
5. `4: van`
6. `5: truck`
7. `6: tricycle`
8. `7: awning-tricycle`
9. `8: bus`
10. `9: motor`
*(Ignored regions and others are automatically filtered during conversion).*

---

## 4. Dataset Inspection & Quality Auditing

The repository includes standalone validation and auditing scripts:

```powershell
# Inspect dataset distribution, class balance, and small-object stats
.\.venv\Scripts\python scripts/dataset/inspect_dataset.py

# Validate bounding box boundaries and perform cross-split leakage checks
.\.venv\Scripts\python scripts/dataset/validate_dataset.py
```

### Auditing Safeguards
- **Coordinate Boundary Checks:** Flags any box where $x, y, w, h \notin [0, 1]$.
- **Small-Object Scale Analysis:** Analyzes bounding box pixel areas ($A < 32^2$ pixels) to calibrate `INFERENCE_IMG_SIZE = 1280` for aerial recall.
- **Cross-Split Hash Checks:** Computes SHA-256 hashes across images to guarantee zero data leakage between splits.
