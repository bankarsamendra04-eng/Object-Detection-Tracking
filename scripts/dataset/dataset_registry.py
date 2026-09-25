"""
Dataset Registry Management Subsystem.
Parses, updates, and queries the authoritative datasets.yaml catalog.
Resolves dataset paths dynamically without hardcoded paths.
"""

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import yaml

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "datasets" / "registry" / "datasets.yaml"


@dataclass
class DatasetEntry:
    dataset_id: str
    dataset_name: str
    version: str
    source_repository: str
    official_source: str
    download_url: Optional[str]
    task: str
    modality: str
    focus: str
    raw_path: str
    processed_path: str
    annotation_format: str
    converted_format: str
    num_classes: int
    classes: Dict[int, str]
    train_split: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: Optional[str] = None
    archive_size_mb: float = 0.0
    extracted_images_count: int = 0
    license: str = "Unknown"
    status: str = "PLANNED"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetEntry":
        # Ensure class keys are integers
        raw_classes = data.get("classes", {})
        parsed_classes = {int(k): str(v) for k, v in raw_classes.items()} if isinstance(raw_classes, dict) else {}
        clean_data = {**data, "classes": parsed_classes}
        return cls(**clean_data)

    def resolve_raw_dir(self, base_override: Optional[Path] = None) -> Path:
        """Resolves absolute path to raw data directory."""
        base = base_override or Path(os.environ.get("DATASET_DIR", PROJECT_ROOT))
        raw_p = Path(self.raw_path)
        return raw_p if raw_p.is_absolute() else (base / raw_p)

    def resolve_processed_dir(self, base_override: Optional[Path] = None) -> Path:
        """Resolves absolute path to processed data directory."""
        base = base_override or Path(os.environ.get("DATASET_DIR", PROJECT_ROOT))
        proc_p = Path(self.processed_path)
        return proc_p if proc_p.is_absolute() else (base / proc_p)


class DatasetRegistry:
    """Manages querying and updating the declarative datasets catalog."""

    VALID_STATUSES = {"PLANNED", "DOWNLOADING", "DOWNLOADED", "PROCESSED", "VALIDATED", "READY"}

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = Path(registry_file) if registry_file else DEFAULT_REGISTRY_PATH
        self.datasets: Dict[str, DatasetEntry] = {}
        self.load()

    def load(self) -> None:
        """Loads and parses the datasets.yaml file."""
        if not self.registry_file.exists():
            self.datasets = {}
            return

        with open(self.registry_file, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}

        raw_datasets = content.get("datasets", {})
        self.datasets = {
            ds_id: DatasetEntry.from_dict(ds_data)
            for ds_id, ds_data in raw_datasets.items()
        }

    def save(self) -> None:
        """Serializes current registry state back to datasets.yaml."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "datasets": {
                ds_id: entry.to_dict()
                for ds_id, entry in self.datasets.items()
            }
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(dump_data, f, sort_keys=False, default_flow_style=False)

    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        return self.datasets.get(dataset_id)

    def list_all(self) -> List[DatasetEntry]:
        return list(self.datasets.values())

    def update_status(self, dataset_id: str, new_status: str) -> bool:
        if new_status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {sorted(self.VALID_STATUSES)}")
        if dataset_id not in self.datasets:
            return False
        self.datasets[dataset_id].status = new_status
        self.save()
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Registry CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all cataloged datasets")

    show_parser = subparsers.add_parser("show", help="Show details for a dataset")
    show_parser.add_argument("dataset_id", help="Dataset identifier")

    update_parser = subparsers.add_parser("update-status", help="Update dataset status")
    update_parser.add_argument("dataset_id", help="Dataset identifier")
    update_parser.add_argument("status", choices=list(DatasetRegistry.VALID_STATUSES), help="New status")

    args = parser.parse_args()
    reg = DatasetRegistry()

    if args.command == "list":
        print(f"\n{'ID':<20} {'Name':<35} {'Task':<18} {'Status':<12}")
        print("-" * 88)
        for ds in reg.list_all():
            print(f"{ds.dataset_id:<20} {ds.dataset_name:<35} {ds.task:<18} {ds.status:<12}")
        print()

    elif args.command == "show":
        ds = reg.get(args.dataset_id)
        if not ds:
            print(f"Dataset '{args.dataset_id}' not found.")
            sys.exit(1)
        print(f"\nDataset: {ds.dataset_name} ({ds.dataset_id})")
        print(f"Status:       {ds.status}")
        print(f"Official:     {ds.official_source}")
        print(f"Repository:   {ds.source_repository}")
        print(f"Download URL: {ds.download_url}")
        print(f"Task:         {ds.task} | Modality: {ds.modality} | Focus: {ds.focus}")
        print(f"Raw Path:     {ds.resolve_raw_dir()}")
        print(f"Processed:    {ds.resolve_processed_dir()}")
        print(f"Classes ({ds.num_classes}): {ds.classes}")
        print(f"Notes:        {ds.notes}\n")

    elif args.command == "update-status":
        success = reg.update_status(args.dataset_id, args.status)
        if success:
            print(f"Updated '{args.dataset_id}' status to '{args.status}'.")
        else:
            print(f"Failed to find dataset '{args.dataset_id}'.")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
