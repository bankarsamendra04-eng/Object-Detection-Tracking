"""
Annotation Conversion Subsystem.
Converts raw VisDrone annotations into standardized, normalized YOLO format.
Preserves original raw data without mutation.
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.logging import get_logger
from scripts.dataset.dataset_registry import DatasetRegistry, DatasetEntry

logger = get_logger("dataset.convert")

# Official VisDrone Category Mapping to YOLO Classes
# VisDrone raw category index -> (YOLO class index, class name)
VISDRONE_TO_YOLO_CLASS_MAP = {
    1: (0, "pedestrian"),
    2: (1, "people"),
    3: (2, "bicycle"),
    4: (3, "car"),
    5: (4, "van"),
    6: (5, "truck"),
    7: (6, "tricycle"),
    8: (7, "awning-tricycle"),
    9: (8, "bus"),
    10: (9, "motor"),
}


def parse_visdrone_line(line: str) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Parses a single line of VisDrone annotation:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    """
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 6:
        return None

    try:
        x = int(float(parts[0]))
        y = int(float(parts[1]))
        w = int(float(parts[2]))
        h = int(float(parts[3]))
        score = int(float(parts[4]))
        cat = int(float(parts[5]))
        return (x, y, w, h, score, cat)
    except ValueError:
        return None


def convert_visdrone_annotation(
    ann_path: Path,
    img_width: int,
    img_height: int,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Converts a VisDrone .txt annotation file to normalized YOLO format.
    Returns: (list of yolo formatted lines, stats dictionary)
    """
    stats = {
        "total_raw_boxes": 0,
        "ignored_score_zero": 0,
        "ignored_category_zero_or_other": 0,
        "zero_area_boxes": 0,
        "valid_converted_boxes": 0,
    }

    if not ann_path.exists() or img_width <= 0 or img_height <= 0:
        return [], stats

    yolo_lines = []
    content = ann_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return [], stats

    for line in content.splitlines():
        if not line.strip():
            continue

        parsed = parse_visdrone_line(line)
        if not parsed:
            continue

        x, y, w, h, score, cat = parsed
        stats["total_raw_boxes"] += 1

        # Check ignored regions flag (score == 0 means ignored region)
        if score == 0:
            stats["ignored_score_zero"] += 1
            continue

        # Check category validity
        if cat not in VISDRONE_TO_YOLO_CLASS_MAP:
            stats["ignored_category_zero_or_other"] += 1
            continue

        if w <= 0 or h <= 0:
            stats["zero_area_boxes"] += 1
            continue

        yolo_cls, _ = VISDRONE_TO_YOLO_CLASS_MAP[cat]

        # Calculate normalized center coordinates and dimensions
        cx = (x + w / 2.0) / img_width
        cy = (y + h / 2.0) / img_height
        norm_w = w / float(img_width)
        norm_h = h / float(img_height)

        # Clamp coordinates to [0.0, 1.0]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        stats["valid_converted_boxes"] += 1

    return yolo_lines, stats


def convert_visdrone_split(
    raw_split_dir: Path,
    processed_base_dir: Path,
    split_name: str = "val",
    copy_images: bool = True,
) -> Dict[str, Any]:
    """
    Converts a complete VisDrone split into YOLO processed directory structure:
    processed/visdrone/images/<split>/
    processed/visdrone/labels/<split>/
    """
    raw_images_dir = raw_split_dir / "images"
    raw_annotations_dir = raw_split_dir / "annotations"

    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")

    target_images_dir = processed_base_dir / "images" / split_name
    target_labels_dir = processed_base_dir / "labels" / split_name
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png")))
    logger.info(f"Converting split '{split_name}': Found {len(img_files)} images in {raw_images_dir}...")

    aggregate_stats = {
        "split": split_name,
        "images_processed": 0,
        "labels_written": 0,
        "total_raw_boxes": 0,
        "ignored_score_zero": 0,
        "ignored_category_zero_or_other": 0,
        "zero_area_boxes": 0,
        "valid_converted_boxes": 0,
        "class_distribution": defaultdict(int),
    }

    manifest_file = PROJECT_ROOT / "data" / "datasets" / "splits" / split_name / f"visdrone_{split_name}.txt"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = []

    for img_path in img_files:
        dest_img_path = target_images_dir / img_path.name
        if copy_images and not dest_img_path.exists():
            shutil.copy2(img_path, dest_img_path)

        manifest_lines.append(str(dest_img_path.resolve()) + "\n")
        aggregate_stats["images_processed"] += 1

        # Read image size safely without loading full raster
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            logger.warning(f"Could not read image header for {img_path.name}: {e}")
            continue

        ann_path = raw_annotations_dir / f"{img_path.stem}.txt"
        yolo_lines, stats = convert_visdrone_annotation(ann_path, w, h)

        for k in ["total_raw_boxes", "ignored_score_zero", "ignored_category_zero_or_other", "zero_area_boxes", "valid_converted_boxes"]:
            aggregate_stats[k] += stats[k]

        for line in yolo_lines:
            cls_id = int(line.split()[0])
            cls_name = VISDRONE_TO_YOLO_CLASS_MAP.get(cls_id + 1, (None, f"class_{cls_id}"))[1]
            aggregate_stats["class_distribution"][cls_name] += 1

        dest_label_path = target_labels_dir / f"{img_path.stem}.txt"
        dest_label_path.write_text("".join(yolo_lines), encoding="utf-8")
        aggregate_stats["labels_written"] += 1

    manifest_file.write_text("".join(manifest_lines), encoding="utf-8")
    aggregate_stats["class_distribution"] = dict(aggregate_stats["class_distribution"])

    logger.info(
        f"Conversion complete for split '{split_name}': {aggregate_stats['images_processed']} images, "
        f"{aggregate_stats['valid_converted_boxes']} valid bounding boxes written to {target_labels_dir}."
    )
    return aggregate_stats


def convert_dataset(dataset_id: str = "visdrone_det_val", base_dir: Optional[Path] = None) -> bool:
    """Orchestrates annotation conversion for a registered dataset."""
    reg = DatasetRegistry()
    entry = reg.get(dataset_id)
    if not entry:
        logger.error(f"Unknown dataset_id '{dataset_id}'.")
        return False

    raw_dir = entry.resolve_raw_dir(base_override=base_dir)
    processed_dir = entry.resolve_processed_dir(base_override=base_dir)

    split_name = "val" if "val" in dataset_id else ("train" if "train" in dataset_id else "test")
    try:
        stats = convert_visdrone_split(
            raw_split_dir=raw_dir,
            processed_base_dir=processed_dir,
            split_name=split_name,
            copy_images=True,
        )
        reg.update_status(dataset_id, "PROCESSED")
        return True
    except Exception as e:
        logger.error(f"Conversion failed for '{dataset_id}': {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Dataset Annotation Converter")
    parser.add_argument("--dataset-id", default="visdrone_det_val", help="ID of dataset to convert")
    parser.add_argument("--base-dir", default=None, help="Base directory override")
    args = parser.parse_args()

    base_p = Path(args.base_dir) if args.base_dir else None
    ok = convert_dataset(dataset_id=args.dataset_id, base_dir=base_p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
