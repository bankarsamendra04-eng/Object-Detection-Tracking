"""
Automated Dataset Download Subsystem.
Supports robust streaming download, pre-flight disk space verification,
resumable downloads, retry handling, and ZipSlip-protected extraction.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Optional
import urllib.request
import zipfile

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.logging import get_logger
from scripts.dataset.dataset_registry import DatasetRegistry, DatasetEntry

logger = get_logger("dataset.download")


def check_disk_space(target_dir: Path, required_mb: float) -> bool:
    """Checks if destination drive has sufficient free storage space (at least 2x requirement)."""
    try:
        usage = shutil.disk_usage(target_dir.anchor or str(target_dir))
        free_mb = usage.free / (1024 * 1024)
        margin_mb = required_mb * 2.0  # Buffer for archive + extraction
        if free_mb < margin_mb:
            logger.error(
                f"Insufficient disk space on {target_dir.anchor}! Free: {free_mb:.1f} MB, Required: {margin_mb:.1f} MB"
            )
            return False
        logger.info(f"Disk space check passed on {target_dir.anchor}. Free: {free_mb:.1f} MB (Buffer needed: {margin_mb:.1f} MB)")
        return True
    except Exception as e:
        logger.warning(f"Could not verify disk space: {e}. Proceeding with caution.")
        return True


def download_file_with_resume(url: str, dest_path: Path, max_retries: int = 3, chunk_size: int = 65536) -> bool:
    """
    Downloads a file with streaming progress, retry handling, and partial file resumption.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            existing_size = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30.0) as response:
                total_size = response.getheader("Content-Length")
                total_bytes = int(total_size) + existing_size if total_size else None

                mode = "ab" if existing_size > 0 and response.status == 206 else "wb"
                if mode == "wb":
                    existing_size = 0

                downloaded = existing_size
                start_time = time.time()
                last_log_time = start_time

                with open(temp_path, mode) as out_f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 2 seconds
                        now = time.time()
                        if now - last_log_time >= 2.0:
                            if total_bytes:
                                pct = (downloaded / total_bytes) * 100.0
                                mb_down = downloaded / (1024 * 1024)
                                mb_tot = total_bytes / (1024 * 1024)
                                logger.info(f"Downloading: {mb_down:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)")
                            else:
                                mb_down = downloaded / (1024 * 1024)
                                logger.info(f"Downloading: {mb_down:.1f} MB")
                            last_log_time = now

            # Download finished successfully
            if temp_path.exists():
                temp_path.replace(dest_path)
            logger.info(f"Download complete: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            logger.warning(f"Download attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
            else:
                logger.error(f"Download failed after {max_retries} attempts.")
                return False
    return False


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """
    Extracts a zip archive safely, protecting against ZipSlip path traversal vulnerabilities.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_extract = extract_dir.resolve()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Check for path traversal attacks
                target_path = (extract_dir / member.filename).resolve()
                try:
                    target_path.relative_to(resolved_extract)
                except ValueError:
                    logger.error(f"Security Alert: ZipSlip path traversal attempt detected in '{member.filename}'! Aborting extraction.")
                    return False

            logger.info(f"Extracting {len(zf.infolist())} entries from {zip_path.name} to {extract_dir}...")
            zf.extractall(extract_dir)
            logger.info("Extraction completed successfully.")
            return True
    except Exception as e:
        logger.error(f"Archive extraction failed: {e}")
        return False


def download_dataset(dataset_id: str = "visdrone_det_val", force: bool = False, base_dir: Optional[Path] = None) -> bool:
    """
    Orchestrates pre-checks, download, and extraction for a cataloged dataset.
    """
    reg = DatasetRegistry()
    entry = reg.get(dataset_id)
    if not entry:
        logger.error(f"Unknown dataset_id '{dataset_id}'. Registered: {[d.dataset_id for d in reg.list_all()]}")
        return False

    raw_dir = entry.resolve_raw_dir(base_override=base_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already present and extracted
    if not force:
        # Check if raw images exist
        img_candidates = list(raw_dir.glob("**/*.jpg")) + list(raw_dir.glob("**/*.png"))
        if len(img_candidates) > 0 and len(img_candidates) >= (entry.extracted_images_count * 0.9):
            logger.info(f"Dataset '{dataset_id}' is already downloaded and extracted at {raw_dir} ({len(img_candidates)} images).")
            if entry.status in {"PLANNED", "DOWNLOADING"}:
                reg.update_status(dataset_id, "DOWNLOADED")
            return True

    if not entry.download_url:
        logger.error(f"Dataset '{dataset_id}' does not have an automated download URL configured.")
        return False

    # Check disk space
    if not check_disk_space(raw_dir, entry.archive_size_mb):
        return False

    reg.update_status(dataset_id, "DOWNLOADING")

    archive_name = Path(entry.download_url).name
    archive_dest = raw_dir.parent / archive_name

    logger.info(f"Starting download for {entry.dataset_name} from {entry.download_url}...")
    success = download_file_with_resume(entry.download_url, archive_dest)
    if not success:
        logger.error(f"Download failed for '{dataset_id}'.")
        return False

    # Extract archive into raw destination
    extract_target = raw_dir.parent
    extract_ok = safe_extract_zip(archive_dest, extract_target)
    if not extract_ok:
        logger.error(f"Extraction failed for '{dataset_id}'.")
        return False

    # Verify extracted content
    extracted_imgs = list(raw_dir.glob("**/*.jpg")) + list(raw_dir.glob("**/*.png"))
    logger.info(f"Extracted {len(extracted_imgs)} images to {raw_dir}.")

    reg.update_status(dataset_id, "DOWNLOADED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Automated Dataset Downloader")
    parser.add_argument("--dataset-id", default="visdrone_det_val", help="ID of dataset to download (default: visdrone_det_val)")
    parser.add_argument("--force", action="store_true", help="Force redownload even if already extracted")
    parser.add_argument("--base-dir", default=None, help="Optional base directory override for external storage")
    args = parser.parse_args()

    base_p = Path(args.base_dir) if args.base_dir else None
    ok = download_dataset(dataset_id=args.dataset_id, force=args.force, base_dir=base_p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
