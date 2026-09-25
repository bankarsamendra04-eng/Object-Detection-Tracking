# Data Directory Architecture

This directory houses all local data assets, sample media, database persistence files, and the dataset management subsystem.

```
data/
├── README.md                          # Data directory overview & policy
├── detection_tracking.db              # SQLite operational database
├── uploads/                           # Temporary user-uploaded media files
├── samples/                           # Bundled verification fixtures (bus.jpg, zidane.jpg, sample_real_bus.mp4)
│
└── datasets/                          # Enterprise dataset storage tier
    ├── registry/                      # Central metadata registry (datasets.yaml)
    ├── raw/                           # Immutable raw downloads (VisDrone, Custom)
    ├── processed/                     # Training-ready YOLO-formatted datasets
    ├── splits/                        # Split definitions & file lists (train, val, test)
    └── samples/                       # Visual inspection samples & rendered bbox previews
```

## Storage Policy & Data Reproducibility
- **Raw Data Integrity:** Original downloaded archives and annotations are stored under `data/datasets/raw/` and are strictly immutable.
- **Processed Artifacts:** Normalised YOLO bounding boxes and organized image directories reside under `data/datasets/processed/`.
- **Git Protection:** Heavy datasets, raw archives (`*.zip`, `*.tar.gz`), and generated tensors are excluded from Git version control via `.gitignore`. Configuration YAMLs, registry catalogs, and conversion scripts remain fully versioned.
