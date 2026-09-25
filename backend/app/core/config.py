from pathlib import Path
from typing import List, Optional, Set, Union
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the entire repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Settings
    APP_NAME: str = "Real-Time Object Detection & Tracking System"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Computer Vision & Detection Settings
    ACTIVE_MODEL_ID: str = "general_pretrained"
    MODEL_TYPE: str = "pretrained"  # "pretrained" or "custom"
    MODEL_NAME: str = "yolov8n.pt"
    MODEL_VERSION: str = "8.0"
    MODEL_SOURCE: str = "ultralytics"
    MODEL_WEIGHTS_DIR: str = "models/weights"
    MODELS_DIR: str = "models"
    MODELS_REGISTRY_PATH: str = "models/registry/models.yaml"
    INFERENCE_IMG_SIZE: int = 640  # 640 default; configure 1280+ for small-object/aerial recall
    DEVICE: str = "auto"
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    HALF_PRECISION: bool = False
    MAX_DETECTIONS: int = 100

    # Tracker Settings
    TRACKER_TYPE: str = "bytetrack.yaml"
    TRACK_PERSISTENCE_BUFFER: int = 30
    TRACK_HIGH_THRESH: float = 0.5
    TRACK_LOW_THRESH: float = 0.1
    TRACK_MATCH_THRESH: float = 0.8

    # Storage & Paths
    DATA_DIR: str = "data"
    UPLOADS_DIR: str = "data/uploads"
    OUTPUTS_DIR: str = "outputs"
    LOGS_DIR: str = "logs"

    # Resource Limits & Security Hardening
    MAX_UPLOAD_IMAGE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    MAX_UPLOAD_VIDEO_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    MAX_IMAGE_DIMENSION: int = 8192  # 8192 px max width / height
    MAX_CONCURRENT_WS_SESSIONS: int = 10  # Max simultaneous streaming sessions
    MAX_WS_MESSAGE_SIZE_BYTES: int = 65536  # 64 KB
    MAX_STREAM_FPS: float = 60.0

    # Database Settings
    DATABASE_URL: str = "sqlite:///./data/detection_tracking.db"
    DB_ECHO: bool = False

    # Analytics Settings
    LINE_CROSSING_COORDS: List[List[int]] = [[0, 360], [1280, 360]]
    TARGET_CLASSES: List[str] = []

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            import json
            return json.loads(v)
        return v

    def resolve_path(self, relative_path: Union[str, Path]) -> Path:
        """Resolves relative project paths to absolute paths based on PROJECT_ROOT safely."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def resolve_safe_path(self, user_path: Union[str, Path], allowed_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Safely resolves a path ensuring it strictly resides within PROJECT_ROOT (or an allowed sub-directory).
        Rejects path traversal (..), UNC paths, and paths escaping the boundary.
        """
        base = self.resolve_path(allowed_dir).resolve() if allowed_dir else PROJECT_ROOT.resolve()
        p_str = str(user_path).strip()
        if not p_str:
            raise ValueError("Path cannot be empty.")
        if "\x00" in p_str:
            raise ValueError("Null bytes are forbidden in paths.")
        if p_str.startswith(("\\\\", "//")):
            raise ValueError(f"UNC network paths are forbidden: '{user_path}'")
        if len(p_str) > 1 and p_str[1] == ":":
            raise ValueError(f"Drive letters are forbidden: '{user_path}'")
        if ".." in p_str.replace("\\", "/").split("/"):
            raise ValueError(f"Path traversal ('..') is strictly forbidden: '{user_path}'")

        candidate = (base / p_str).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            raise ValueError(f"Access denied: '{user_path}' escapes allowed directory '{base}'")

        return candidate

    def get_resolved_model_path(self, model_name_or_path: Union[str, None] = None) -> str:
        """
        Resolves model path to absolute file or fallback weight.
        Checks:
        1. Explicit path if provided
        2. Direct file at PROJECT_ROOT
        3. models/weights/
        4. models/pretrained/
        5. models/custom/
        6. Ultralytics auto-resolver name
        """
        target = model_name_or_path or self.MODEL_NAME

        # If already an existing absolute or relative file path
        direct_path = Path(target)
        if direct_path.is_file():
            return str(direct_path.resolve())

        resolved_direct = self.resolve_path(target)
        if resolved_direct.is_file():
            return str(resolved_direct)

        # Check standard model search locations
        search_dirs = [
            PROJECT_ROOT,
            self.resolve_path(self.MODEL_WEIGHTS_DIR),
            self.resolve_path("models/pretrained"),
            self.resolve_path("models/custom"),
            self.resolve_path("models"),
        ]

        target_name = Path(target).name
        for sdir in search_dirs:
            candidate = sdir / target_name
            if candidate.is_file():
                return str(candidate)

        # Fallback to model name for Ultralytics auto-download / registry
        return target

    def get_effective_device(self) -> str:
        """Determines whether to execute on CUDA or CPU based on config and hardware."""
        if self.DEVICE.lower() == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.DEVICE.lower().startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.DEVICE

    def ensure_directories_exist(self) -> None:
        """Ensures all configured operational directories exist."""
        directories = [
            self.DATA_DIR,
            self.UPLOADS_DIR,
            self.OUTPUTS_DIR,
            self.LOGS_DIR,
            self.MODEL_WEIGHTS_DIR,
            "models/pretrained",
            "models/custom",
            "models/registry",
            "models/metadata",
        ]
        for path_attr in directories:
            dir_path = self.resolve_path(path_attr)
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories_exist()
