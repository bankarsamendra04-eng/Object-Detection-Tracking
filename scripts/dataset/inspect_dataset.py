"""
Dataset Inspection and Visualization Subsystem.
Computes dataset distributions and generates annotated preview samples
in data/datasets/samples/images/ for visual inspection.
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.logging import get_logger
from scripts.dataset.dataset_registry import DatasetRegistry, DatasetEntry

logger = get_logger("dataset.inspect")

# Distinct BGR colors for visualization
PALETTE = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),  # spring green
    (0, 128, 255),  # orange
    (255, 128, 0),  # azure
    (128, 0, 255),  # violet
]


def draw_yolo_labels_on_image(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: Dict[int, str],
) -> bool:
    """Draws YOLO normalized bounding boxes on an image and saves the preview."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    if not label_path.exists():
        return False

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, nw, nh = map(float, parts[1:])
        except ValueError:
            continue

        x1 = int((cx - nw / 2.0) * w)
        y1 = int((cy - nh / 2.0) * h)
        x2 = int((cx + nw / 2.0) * w)
        y2 = int((cy + nh / 2.0) * h)

        color = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        c_name = class_names.get(cls_id, f"class_{cls_id}")
        label_text = f"{c_name}"
        cv2.putText(
            img,
            label_text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return True


def inspect_dataset(
    dataset_id: str = "visdrone_det_val",
    split: str = "val",
    num_samples: int = 5,
    base_dir: Optional[Path] = None,
) -> bool:
    """Generates distribution summary and saves annotated sample images."""
    reg = DatasetRegistry()
    entry = reg.get(dataset_id)
    if not entry:
        logger.error(f"Dataset '{dataset_id}' not found.")
        return False

    processed_dir = entry.resolve_processed_dir(base_override=base_dir)
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split

    if not img_dir.exists():
        logger.error(f"Directory not found: {img_dir}")
        return False

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    logger.info(f"Inspecting {len(img_files)} images in {img_dir}...")

    samples_dir = PROJECT_ROOT / "data" / "datasets" / "samples" / "images"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rendered_count = 0
    for img_p in img_files[:num_samples]:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        out_p = samples_dir / f"annotated_{img_p.name}"
        if draw_yolo_labels_on_image(img_p, lbl_p, out_p, entry.classes):
            rendered_count += 1
            logger.info(f"Generated sample visualization: {out_p.name}")

    print(f"\nInspection preview complete: {rendered_count} annotated sample images rendered to {samples_dir}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Dataset Inspector")
    parser.add_argument("--dataset-id", default="visdrone_det_val", help="Dataset identifier")
    parser.add_argument("--split", default="val", help="Split to inspect (default: val)")
    parser.add_argument("--samples", type=int, default=5, help="Number of visual samples to render")
    parser.add_argument("--base-dir", default=None, help="Base directory override")
    args = parser.parse_args()

    base_p = Path(args.base_dir) if args.base_dir else None
    ok = inspect_dataset(dataset_id=args.dataset_id, split=args.split, num_samples=args.samples, base_dir=base_p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
