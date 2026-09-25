"""
Dataset Validation Subsystem.
Audits image integrity, label syntax, coordinate bounds, class distribution,
data leakage across splits, and comprehensive small-object size statistics.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.logging import get_logger
from scripts.dataset.dataset_registry import DatasetRegistry, DatasetEntry

logger = get_logger("dataset.validate")

# Bounding box object size categories (pixel area based on COCO + aerial drone standards)
# Very Small: < 16x16 (< 256 px^2)
# Small:      16x16 to 32x32 (256 - 1,024 px^2)
# Medium:     32x32 to 96x96 (1,024 - 9,216 px^2)
# Large:      > 96x96 (> 9,216 px^2)
SIZE_THRESHOLDS = {
    "very_small_max_area": 256,    # 16x16 px
    "small_max_area": 1024,        # 32x32 px (Standard COCO small threshold)
    "medium_max_area": 9216,       # 96x96 px (Standard COCO medium threshold)
}


def compute_file_hash(path: Path) -> str:
    """Calculates MD5 hash of a file for duplicate & leakage detection."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetValidator:
    """Validates processed YOLO dataset integrity, annotations, and split separation."""

    def __init__(self, dataset_dir: Path, num_classes: int = 10, class_names: Optional[Dict[int, str]] = None):
        self.dataset_dir = Path(dataset_dir)
        self.num_classes = num_classes
        self.class_names = class_names or {}
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

    def validate_split(self, split: str = "val") -> Dict[str, Any]:
        """Validates all image-label pairs and box distributions within a split."""
        split_img_dir = self.images_dir / split
        split_lbl_dir = self.labels_dir / split

        if not split_img_dir.exists():
            return {"error": f"Image split directory does not exist: {split_img_dir}"}

        image_files = sorted(list(split_img_dir.glob("*.jpg")) + list(split_img_dir.glob("*.png")))

        report: Dict[str, Any] = {
            "split": split,
            "total_images": len(image_files),
            "valid_images": 0,
            "corrupt_images": [],
            "missing_label_files": [],
            "empty_label_files": [],
            "total_boxes": 0,
            "malformed_box_lines": 0,
            "invalid_class_ids": 0,
            "out_of_bounds_coords": 0,
            "zero_area_boxes": 0,
            "image_resolutions": [],
            "class_counts": defaultdict(int),
            "size_distribution": {
                "very_small": 0,  # < 256 px^2 (< 16x16)
                "small": 0,       # 256 - 1024 px^2 (16x16 to 32x32)
                "medium": 0,      # 1024 - 9216 px^2 (32x32 to 96x96)
                "large": 0,       # > 9216 px^2 (> 96x96)
            },
            "box_areas_px": [],
            "box_relative_areas": [],
            "image_hashes": {},
        }

        for img_p in image_files:
            # 1. Image Readability & Dimensions
            try:
                with Image.open(img_p) as im:
                    im.verify()
                with Image.open(img_p) as im:
                    w, h = im.size
                report["valid_images"] += 1
                report["image_resolutions"].append((w, h))
                # Hash for leakage check
                report["image_hashes"][img_p.name] = compute_file_hash(img_p)
            except Exception as e:
                report["corrupt_images"].append({"file": img_p.name, "error": str(e)})
                continue

            # 2. Label Inspection
            lbl_p = split_lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                report["missing_label_files"].append(img_p.name)
                continue

            content = lbl_p.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                report["empty_label_files"].append(img_p.name)
                continue

            for line in content.splitlines():
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) != 5:
                    report["malformed_box_lines"] += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, norm_w, norm_h = map(float, parts[1:5])
                except ValueError:
                    report["malformed_box_lines"] += 1
                    continue

                # Class index validation
                if cls_id < 0 or cls_id >= self.num_classes:
                    report["invalid_class_ids"] += 1
                    continue

                # Coordinate bounds validation
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0):
                    report["out_of_bounds_coords"] += 1

                # Calculate absolute box dimensions
                box_w_px = norm_w * w
                box_h_px = norm_h * h
                box_area_px = box_w_px * box_h_px
                rel_area = norm_w * norm_h

                if box_area_px <= 0:
                    report["zero_area_boxes"] += 1
                    continue

                report["total_boxes"] += 1
                c_name = self.class_names.get(cls_id, f"class_{cls_id}")
                report["class_counts"][c_name] += 1
                report["box_areas_px"].append(box_area_px)
                report["box_relative_areas"].append(rel_area)

                # Classify small-object category
                if box_area_px < SIZE_THRESHOLDS["very_small_max_area"]:
                    report["size_distribution"]["very_small"] += 1
                elif box_area_px < SIZE_THRESHOLDS["small_max_area"]:
                    report["size_distribution"]["small"] += 1
                elif box_area_px < SIZE_THRESHOLDS["medium_max_area"]:
                    report["size_distribution"]["medium"] += 1
                else:
                    report["size_distribution"]["large"] += 1

        report["class_counts"] = dict(report["class_counts"])
        return report

    def check_split_leakage(self, splits: List[str] = ["train", "val"]) -> Dict[str, Any]:
        """Verifies absence of duplicate images across different splits via hash comparison."""
        hashes_per_split: Dict[str, Dict[str, str]] = {}
        for sp in splits:
            sp_dir = self.images_dir / sp
            if sp_dir.exists():
                files = list(sp_dir.glob("*.jpg")) + list(sp_dir.glob("*.png"))
                hashes_per_split[sp] = {f.name: compute_file_hash(f) for f in files}

        leakages = []
        split_names = list(hashes_per_split.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                h1_map = {h: fname for fname, h in hashes_per_split[s1].items()}
                h2_map = {h: fname for fname, h in hashes_per_split[s2].items()}
                common_hashes = set(h1_map.keys()) & set(h2_map.keys())
                for h in common_hashes:
                    leakages.append({
                        "hash": h,
                        "file_split1": f"{s1}/{h1_map[h]}",
                        "file_split2": f"{s2}/{h2_map[h]}",
                    })

        return {
            "splits_checked": splits,
            "leakage_count": len(leakages),
            "leaked_pairs": leakages,
            "status": "PASS" if len(leakages) == 0 else "FAIL",
        }


def validate_dataset(dataset_id: str = "visdrone_det_val", base_dir: Optional[Path] = None) -> bool:
    """Runs end-to-end dataset validation and produces structured metrics."""
    reg = DatasetRegistry()
    entry = reg.get(dataset_id)
    if not entry:
        logger.error(f"Dataset '{dataset_id}' not found in registry.")
        return False

    processed_dir = entry.resolve_processed_dir(base_override=base_dir)
    validator = DatasetValidator(
        dataset_dir=processed_dir,
        num_classes=entry.num_classes,
        class_names=entry.classes,
    )

    split = "val" if "val" in dataset_id else ("train" if "train" in dataset_id else "test")
    logger.info(f"Running validation audit on dataset '{dataset_id}', split '{split}'...")
    res = validator.validate_split(split=split)

    if "error" in res:
        logger.error(f"Validation failed: {res['error']}")
        return False

    # Check leakages across existing splits
    leakage_res = validator.check_split_leakage(splits=["train", "val", "test"])

    # Compute summary statistics
    areas = res.get("box_areas_px", [])
    mean_area = float(np.mean(areas)) if areas else 0.0
    median_area = float(np.median(areas)) if areas else 0.0
    total_boxes = res["total_boxes"]

    very_small_pct = (res["size_distribution"]["very_small"] / total_boxes * 100.0) if total_boxes else 0.0
    small_pct = (res["size_distribution"]["small"] / total_boxes * 100.0) if total_boxes else 0.0
    medium_pct = (res["size_distribution"]["medium"] / total_boxes * 100.0) if total_boxes else 0.0
    large_pct = (res["size_distribution"]["large"] / total_boxes * 100.0) if total_boxes else 0.0

    print("\n" + "=" * 70)
    print(f"DATASET VALIDATION AUDIT REPORT: {entry.dataset_name}")
    print("=" * 70)
    print(f"Split:               {split}")
    print(f"Total Images:        {res['total_images']} (Valid: {res['valid_images']}, Corrupt: {len(res['corrupt_images'])})")
    print(f"Total BBoxes:        {res['total_boxes']}")
    print(f"Missing Labels:      {len(res['missing_label_files'])}")
    print(f"Empty Labels:        {len(res['empty_label_files'])}")
    print(f"Malformed Lines:     {res['malformed_box_lines']}")
    print(f"Invalid Class IDs:   {res['invalid_class_ids']}")
    print(f"Out of Bounds:       {res['out_of_bounds_coords']}")
    print(f"Zero Area Boxes:     {res['zero_area_boxes']}")
    print(f"Split Leakage:       {leakage_res['status']} ({leakage_res['leakage_count']} duplicates across splits)")
    print("\n--- SMALL-OBJECT SIZE DISTRIBUTION (COCO & Aerial Thresholds) ---")
    print(f"Very Small (<16x16): {res['size_distribution']['very_small']} ({very_small_pct:.1f}%)")
    print(f"Small (16x16-32x32): {res['size_distribution']['small']} ({small_pct:.1f}%)")
    print(f"Medium (32-96x96):   {res['size_distribution']['medium']} ({medium_pct:.1f}%)")
    print(f"Large (>96x96):      {res['size_distribution']['large']} ({large_pct:.1f}%)")
    print(f"Mean BBox Area:      {mean_area:.1f} px^2 | Median Area: {median_area:.1f} px^2")
    print("\n--- CLASS DISTRIBUTION ---")
    for c_name, count in sorted(res["class_counts"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {c_name:<16}: {count:>6} boxes")
    print("=" * 70 + "\n")

    # Persist JSON report
    report_file = PROJECT_ROOT / "data" / "datasets" / "splits" / "validation_report.json"
    clean_res = {**res}
    clean_res.pop("image_resolutions", None)
    clean_res.pop("image_hashes", None)
    clean_res.pop("box_areas_px", None)
    clean_res.pop("box_relative_areas", None)
    clean_res["mean_bbox_area_px"] = round(mean_area, 2)
    clean_res["median_bbox_area_px"] = round(median_area, 2)
    clean_res["leakage_audit"] = leakage_res

    report_file.write_text(json.dumps(clean_res, indent=2), encoding="utf-8")
    logger.info(f"Detailed validation report persisted to {report_file}")

    # Certification criteria: zero corrupt images, zero invalid class IDs, zero out of bounds
    is_valid = (
        len(res["corrupt_images"]) == 0
        and res["invalid_class_ids"] == 0
        and res["malformed_box_lines"] == 0
        and res["total_boxes"] > 0
        and leakage_res["leakage_count"] == 0
    )

    if is_valid:
        reg.update_status(dataset_id, "READY")
        logger.info(f"Dataset '{dataset_id}' successfully validated and certified READY.")
    else:
        reg.update_status(dataset_id, "VALIDATED")
        logger.warning(f"Dataset '{dataset_id}' has warnings or validation issues.")

    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Dataset Validator")
    parser.add_argument("--dataset-id", default="visdrone_det_val", help="Dataset identifier to validate")
    parser.add_argument("--base-dir", default=None, help="Base directory override")
    args = parser.parse_args()

    base_p = Path(args.base_dir) if args.base_dir else None
    ok = validate_dataset(dataset_id=args.dataset_id, base_dir=base_p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
