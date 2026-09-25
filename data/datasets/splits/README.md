# Dataset Splits & Partition Lists

This directory maintains explicit file manifests for experimental partitions:
- `train/`: Text lists of image file paths assigned to the training split.
- `val/`: Text lists of image file paths assigned to the validation split.
- `test/`: Text lists of image file paths assigned to the test split.

Maintaining split manifests guarantees deterministic experiments and facilitates leakage auditing across training and validation sets.
