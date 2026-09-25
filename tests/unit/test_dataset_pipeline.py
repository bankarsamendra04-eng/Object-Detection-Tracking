"""
Unit tests for the Dataset Subsystem (Step 9).
Validates registry parsing, path resolution, safe extraction, annotation conversion,
coordinate normalization, dataset validation, split leakage detection, and small-object classification.
Uses lightweight synthetic fixtures without external network calls or multi-GB dependencies.
"""

import io
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from PIL import Image
import pytest

from scripts.dataset.dataset_registry import DatasetRegistry, DatasetEntry
from scripts.dataset.download_dataset import check_disk_space, safe_extract_zip
from scripts.dataset.convert_annotations import (
    parse_visdrone_line,
    convert_visdrone_annotation,
    VISDRONE_TO_YOLO_CLASS_MAP,
)
from scripts.dataset.validate_dataset import DatasetValidator, SIZE_THRESHOLDS, compute_file_hash


# ==========================================
# 1. Dataset Registry Tests
# ==========================================

def test_registry_loading_and_retrieval():
    """Verifies that the dataset registry loads datasets.yaml and returns valid entries."""
    reg = DatasetRegistry()
    entry = reg.get("visdrone_det_val")
    assert entry is not None
    assert entry.dataset_id == "visdrone_det_val"
    assert entry.num_classes == 10
    assert 0 in entry.classes
    assert entry.classes[0] == "pedestrian"
    assert entry.classes[3] == "car"
    assert entry.task == "object_detection"


def test_registry_status_update(tmp_path):
    """Verifies updating dataset status and rejection of invalid status strings."""
    temp_yaml = tmp_path / "datasets.yaml"
    reg_orig = DatasetRegistry()
    reg = DatasetRegistry(registry_file=temp_yaml)
    reg.datasets = reg_orig.datasets
    reg.save()

    # Valid update
    ok = reg.update_status("visdrone_det_val", "DOWNLOADING")
    assert ok is True
    assert reg.get("visdrone_det_val").status == "DOWNLOADING"

    # Reload from disk
    reg_reloaded = DatasetRegistry(registry_file=temp_yaml)
    assert reg_reloaded.get("visdrone_det_val").status == "DOWNLOADING"

    # Invalid status rejection
    with pytest.raises(ValueError):
        reg.update_status("visdrone_det_val", "NOT_A_VALID_STATUS")


def test_registry_path_resolution(tmp_path):
    """Verifies dynamic path resolution for raw and processed directories."""
    reg = DatasetRegistry()
    entry = reg.get("visdrone_det_val")
    raw_p = entry.resolve_raw_dir(base_override=tmp_path)
    proc_p = entry.resolve_processed_dir(base_override=tmp_path)

    assert str(raw_p).startswith(str(tmp_path))
    assert str(proc_p).startswith(str(tmp_path))
    assert "VisDrone2019-DET-val" in str(raw_p)


# ==========================================
# 2. Archive Safety & ZipSlip Prevention
# ==========================================

def test_safe_extract_zip_valid(tmp_path):
    """Verifies safe extraction of valid archives."""
    zip_path = tmp_path / "valid.zip"
    extract_dir = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("subfolder/sample.txt", "Hello World")

    ok = safe_extract_zip(zip_path, extract_dir)
    assert ok is True
    assert (extract_dir / "subfolder" / "sample.txt").exists()
    assert (extract_dir / "subfolder" / "sample.txt").read_text() == "Hello World"


def test_safe_extract_zipslip_rejection(tmp_path):
    """Verifies that ZipSlip path traversal attempts are detected and rejected."""
    malicious_zip = tmp_path / "malicious.zip"
    extract_dir = tmp_path / "extracted_safe"
    extract_dir.mkdir()

    # Create zip with directory traversal entry
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../../evil.txt", "Malicious content")

    ok = safe_extract_zip(malicious_zip, extract_dir)
    assert ok is False
    assert not (tmp_path / "evil.txt").exists()


# ==========================================
# 3. Annotation Parsing & Conversion Tests
# ==========================================

def test_parse_visdrone_line():
    """Verifies parsing of VisDrone raw annotation lines."""
    # Standard format: bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
    valid_line = "684,249,42,29,1,4,0,0"
    parsed = parse_visdrone_line(valid_line)
    assert parsed == (684, 249, 42, 29, 1, 4)

    # Empty and corrupt lines
    assert parse_visdrone_line("") is None
    assert parse_visdrone_line("invalid,data") is None
    assert parse_visdrone_line("10,20,30,40,abc,4") is None


def test_convert_visdrone_annotation(tmp_path):
    """Verifies VisDrone to YOLO conversion with normalization, category mapping, and ignored filtering."""
    ann_file = tmp_path / "sample_ann.txt"
    # Lines:
    # 1. Car (cat=4 -> yolo cls=3), score=1 (valid)
    # 2. Pedestrian (cat=1 -> yolo cls=0), score=1 (valid)
    # 3. Ignored region: score=0 (should be skipped)
    # 4. Ignored category: cat=0 (should be skipped)
    # 5. Category 11: others (should be skipped)
    # 6. Zero-width box (should be skipped)
    lines = [
        "100,50,200,100,1,4,0,0",
        "300,100,50,80,1,1,0,0",
        "50,50,30,30,0,1,0,0",
        "20,20,40,40,1,0,0,0",
        "10,10,20,20,1,11,0,0",
        "15,15,0,50,1,2,0,0",
    ]
    ann_file.write_text("\n".join(lines), encoding="utf-8")

    img_w, img_h = 1000, 500
    yolo_lines, stats = convert_visdrone_annotation(ann_file, img_w, img_h)

    assert stats["total_raw_boxes"] == 6
    assert stats["ignored_score_zero"] == 1
    assert stats["ignored_category_zero_or_other"] == 2
    assert stats["zero_area_boxes"] == 1
    assert stats["valid_converted_boxes"] == 2
    assert len(yolo_lines) == 2

    # Verify Car (cat 4 -> cls 3):
    # cx = (100 + 200/2) / 1000 = 200 / 1000 = 0.200000
    # cy = (50 + 100/2) / 500 = 100 / 500 = 0.200000
    # nw = 200 / 1000 = 0.200000
    # nh = 100 / 500 = 0.200000
    line1 = yolo_lines[0].split()
    assert line1[0] == "3"
    assert abs(float(line1[1]) - 0.2) < 1e-4
    assert abs(float(line1[2]) - 0.2) < 1e-4
    assert abs(float(line1[3]) - 0.2) < 1e-4
    assert abs(float(line1[4]) - 0.2) < 1e-4

    # Verify Pedestrian (cat 1 -> cls 0):
    # cx = (300 + 25) / 1000 = 0.325000
    # cy = (100 + 40) / 500 = 0.280000
    line2 = yolo_lines[1].split()
    assert line2[0] == "0"
    assert abs(float(line2[1]) - 0.325) < 1e-4
    assert abs(float(line2[2]) - 0.280) < 1e-4


# ==========================================
# 4. Dataset Validation & Leakage Tests
# ==========================================

@pytest.fixture
def synthetic_yolo_dataset(tmp_path):
    """Creates a tiny synthetic YOLO-formatted dataset for validation testing."""
    ds_dir = tmp_path / "mock_yolo_dataset"
    for split in ["train", "val"]:
        (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split: val (2 images)
    # Image 1: 640x480 with 2 boxes (1 small, 1 large)
    img1 = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img1.save(ds_dir / "images" / "val" / "val_001.jpg")
    (ds_dir / "labels" / "val" / "val_001.txt").write_text(
        "0 0.5 0.5 0.03 0.04\n"  # 19.2px x 19.2px area ~ 368 px^2 -> small
        "3 0.2 0.3 0.40 0.30\n", # 256px x 144px area ~ 36864 px^2 -> large
        encoding="utf-8",
    )

    # Image 2: 800x600 with 1 very small box
    img2 = Image.new("RGB", (800, 600), color=(50, 80, 120))
    img2.save(ds_dir / "images" / "val" / "val_002.jpg")
    (ds_dir / "labels" / "val" / "val_002.txt").write_text(
        "1 0.1 0.1 0.01 0.01\n", # 8px x 6px area = 48 px^2 -> very small
        encoding="utf-8",
    )

    # Split: train (1 unique image)
    img_tr = Image.new("RGB", (640, 480), color=(200, 100, 50))
    img_tr.save(ds_dir / "images" / "train" / "train_001.jpg")
    (ds_dir / "labels" / "train" / "train_001.txt").write_text(
        "3 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    return ds_dir


def test_dataset_validator_clean_split(synthetic_yolo_dataset):
    """Verifies that a well-formed dataset split passes validation completely."""
    validator = DatasetValidator(
        dataset_dir=synthetic_yolo_dataset,
        num_classes=10,
        class_names={0: "pedestrian", 1: "people", 3: "car"},
    )
    report = validator.validate_split(split="val")

    assert report["total_images"] == 2
    assert report["valid_images"] == 2
    assert len(report["corrupt_images"]) == 0
    assert report["total_boxes"] == 3
    assert report["invalid_class_ids"] == 0
    assert report["malformed_box_lines"] == 0
    assert report["out_of_bounds_coords"] == 0
    assert report["size_distribution"]["very_small"] == 1
    assert report["size_distribution"]["small"] == 1
    assert report["size_distribution"]["large"] == 1


def test_dataset_validator_corrupt_image_detection(synthetic_yolo_dataset):
    """Verifies detection of unreadable / corrupt image files."""
    corrupt_img = synthetic_yolo_dataset / "images" / "val" / "val_corrupt.jpg"
    corrupt_img.write_bytes(b"NOT_A_VALID_JPEG_HEADER_BYTES")

    validator = DatasetValidator(dataset_dir=synthetic_yolo_dataset, num_classes=10)
    report = validator.validate_split(split="val")

    assert len(report["corrupt_images"]) == 1
    assert report["corrupt_images"][0]["file"] == "val_corrupt.jpg"


def test_dataset_validator_invalid_and_out_of_bounds_boxes(synthetic_yolo_dataset):
    """Verifies detection of out-of-range class IDs and coordinates outside [0.0, 1.0]."""
    bad_lbl = synthetic_yolo_dataset / "labels" / "val" / "val_bad.txt"
    bad_lbl.write_text(
        "99 0.5 0.5 0.1 0.1\n"    # Class ID 99 (exceeds num_classes=10)
        "0 1.5 0.5 0.1 0.1\n"     # Out-of-bounds cx=1.5
        "0 0.5 0.5 0.0 0.1\n"     # Zero-width box
        "malformed row text\n",
        encoding="utf-8",
    )
    # Create matching dummy image
    img = Image.new("RGB", (100, 100), color=(10, 20, 30))
    img.save(synthetic_yolo_dataset / "images" / "val" / "val_bad.jpg")

    validator = DatasetValidator(dataset_dir=synthetic_yolo_dataset, num_classes=10)
    report = validator.validate_split(split="val")

    assert report["invalid_class_ids"] >= 1
    assert report["out_of_bounds_coords"] >= 1
    assert report["zero_area_boxes"] >= 1
    assert report["malformed_box_lines"] >= 1


def test_split_leakage_detection(synthetic_yolo_dataset):
    """Verifies that duplicate images across train and val splits are flagged as leakage."""
    validator = DatasetValidator(dataset_dir=synthetic_yolo_dataset)
    # Clean check: no leakage initially
    res_clean = validator.check_split_leakage(splits=["train", "val"])
    assert res_clean["status"] == "PASS"
    assert res_clean["leakage_count"] == 0

    # Introduce duplicate: copy val_001.jpg into train split with a different name
    val_img = synthetic_yolo_dataset / "images" / "val" / "val_001.jpg"
    train_leaked = synthetic_yolo_dataset / "images" / "train" / "train_duplicate.jpg"
    train_leaked.write_bytes(val_img.read_bytes())

    res_leaked = validator.check_split_leakage(splits=["train", "val"])
    assert res_leaked["status"] == "FAIL"
    assert res_leaked["leakage_count"] >= 1
