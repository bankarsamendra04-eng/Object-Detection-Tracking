# Dataset Registry

The dataset registry acts as the authoritative catalog of all external, benchmark, and custom datasets configured for this platform.

## Configuration File
- `datasets.yaml`: Declarative YAML mapping each `dataset_id` to its metadata, upstream source, task, splits, class taxonomy, and local processing status.

## Lifecycle Statuses
1. `PLANNED`: Cataloged in registry but not yet downloaded.
2. `DOWNLOADING`: Active download task underway.
3. `DOWNLOADED`: Raw archive successfully fetched and extracted into `data/datasets/raw/`.
4. `PROCESSED`: Annotations converted into standardized YOLO format under `data/datasets/processed/`.
5. `VALIDATED`: Integrity, syntax, dimension, and split leakage verification completed.
6. `READY`: Fully certified and ready for model evaluation or training.
