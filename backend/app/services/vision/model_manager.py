from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml

import numpy as np
import torch
from ultralytics import YOLO

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.core.logging import get_logger

logger = get_logger("vision.model_manager")


class ModelManagerError(Exception):
    """Base exception for model manager failures."""
    pass


class ModelNotFoundError(ModelManagerError):
    """Raised when a requested model ID is not registered in the registry."""
    pass


class ModelUnavailableError(ModelManagerError):
    """Raised when model weights are not physically available on disk."""
    pass


class ModelCorruptError(ModelManagerError):
    """Raised when model weights are damaged or incompatible."""
    pass


@dataclass
class ModelEntry:
    """Represents a registered model in the registry."""
    model_id: str
    model_name: str
    model_type: str  # "pretrained" or "custom"
    version: str
    framework: str
    task: str
    weights_path: str
    source: str
    input_size: int = 640
    recommended_confidence: float = 0.35
    recommended_iou: float = 0.45
    device: str = "auto"
    status: str = "NOT_AVAILABLE"
    num_classes: int = 0
    classes: Dict[int, str] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self, is_active: bool = False) -> Dict[str, Any]:
        """Converts model entry to dictionary format for API responses."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "framework": self.framework,
            "task": self.task,
            "weights_path": self.weights_path,
            "source": self.source,
            "input_size": int(self.input_size),
            "recommended_confidence": float(self.recommended_confidence),
            "recommended_iou": float(self.recommended_iou),
            "device": self.device,
            "status": self.status,
            "num_classes": int(self.num_classes or len(self.classes)),
            "classes": {int(k): str(v) for k, v in self.classes.items()} if self.classes else {},
            "is_active": is_active,
            "notes": self.notes,
        }


class ModelManager:
    """
    Production Model Manager & Model Registry layer.
    Manages declarative model configurations, weight validation, model caching,
    safe switching, and hardware resource lifecycle.
    """

    ALLOWED_WEIGHT_EXTENSIONS: Set[str] = {".pt", ".onnx", ".engine"}

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        self.registry_path = Path(registry_path) if registry_path else settings.resolve_path(settings.MODELS_REGISTRY_PATH)
        self.active_model_id: str = settings.ACTIVE_MODEL_ID
        self._model_cache: Dict[str, YOLO] = {}
        self._registry: Dict[str, ModelEntry] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Parses the YAML model registry and populates registered models."""
        self._registry.clear()

        if not self.registry_path.is_file():
            logger.warning(f"Registry file not found at {self.registry_path}. Initializing default entries.")
            self._init_default_registry()
            return

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            raw_models = data.get("models", {})
            for mid, mdata in raw_models.items():
                classes_dict = {}
                raw_classes = mdata.get("classes", {})
                if isinstance(raw_classes, dict):
                    classes_dict = {int(k): str(v) for k, v in raw_classes.items()}

                entry = ModelEntry(
                    model_id=str(mid),
                    model_name=mdata.get("model_name", mid),
                    model_type=mdata.get("model_type", "pretrained"),
                    version=str(mdata.get("version", "1.0")),
                    framework=mdata.get("framework", "ultralytics_yolo"),
                    task=mdata.get("task", "object_detection"),
                    weights_path=mdata.get("weights_path", ""),
                    source=mdata.get("source", "local"),
                    input_size=int(mdata.get("input_size", 640)),
                    recommended_confidence=float(mdata.get("recommended_confidence", 0.35)),
                    recommended_iou=float(mdata.get("recommended_iou", 0.45)),
                    device=mdata.get("device", "auto"),
                    status=mdata.get("status", "NOT_AVAILABLE"),
                    num_classes=int(mdata.get("num_classes", len(classes_dict))),
                    classes=classes_dict,
                    notes=mdata.get("notes"),
                )
                self._registry[entry.model_id] = entry

            logger.info(f"Loaded {len(self._registry)} model entries from registry ({self.registry_path.name}).")
        except Exception as e:
            logger.error(f"Failed to read model registry from {self.registry_path}: {e}")
            self._init_default_registry()

    def _init_default_registry(self) -> None:
        """Fallback initialization for default models."""
        self._registry["general_pretrained"] = ModelEntry(
            model_id="general_pretrained",
            model_name="YOLOv8 Nano (COCO Pretrained)",
            model_type="pretrained",
            version="8.0",
            framework="ultralytics_yolo",
            task="object_detection",
            weights_path="yolov8n.pt",
            source="ultralytics",
            input_size=640,
            recommended_confidence=0.35,
            recommended_iou=0.45,
            device="auto",
            status="AVAILABLE",
            num_classes=80,
            notes="Default general-purpose COCO pretrained detector.",
        )
        self._registry["custom_visdrone"] = ModelEntry(
            model_id="custom_visdrone",
            model_name="YOLOv8 Custom VisDrone Detector",
            model_type="custom",
            version="1.0-planned",
            framework="ultralytics_yolo",
            task="object_detection",
            weights_path="models/custom/yolov8_visdrone.pt",
            source="custom_training",
            input_size=1280,
            recommended_confidence=0.25,
            recommended_iou=0.45,
            device="auto",
            status="NOT_AVAILABLE",
            num_classes=10,
            classes={
                0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
                5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
            },
            notes="Pending custom training in Step 11.",
        )

    def resolve_weights_path(self, weights_path: str) -> Optional[Path]:
        """
        Safely resolves weights path to an absolute Path on disk.
        Enforces security checks to prevent arbitrary path traversal outside the project.
        """
        if not weights_path:
            return None

        p = Path(weights_path)
        # Direct check
        candidates = [
            PROJECT_ROOT / p,
            PROJECT_ROOT / "models" / "weights" / p.name,
            PROJECT_ROOT / "models" / "pretrained" / p.name,
            PROJECT_ROOT / "models" / "custom" / p.name,
            PROJECT_ROOT / "models" / p.name,
            PROJECT_ROOT / p.name,
        ]

        for cand in candidates:
            try:
                resolved = cand.resolve()
                # Security boundary: must reside within PROJECT_ROOT
                if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
                    logger.warning(f"Security violation: path '{resolved}' outside project boundary.")
                    return None
                if resolved.is_file():
                    return resolved
            except Exception:
                continue

        # If it doesn't physically exist, return canonical relative path within PROJECT_ROOT
        try:
            canonical = (PROJECT_ROOT / p).resolve()
            if not str(canonical).startswith(str(PROJECT_ROOT.resolve())):
                return None
            return canonical
        except Exception:
            return None

    def is_model_available(self, model_id: str) -> bool:
        """Checks if the model weights file physically exists on disk."""
        entry = self.get_model(model_id)
        if not entry:
            return False
        resolved = self.resolve_weights_path(entry.weights_path)
        return resolved is not None and resolved.is_file()

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Retrieves a registered model entry by ID."""
        if model_id in self._registry:
            return self._registry[model_id]
        # Case-insensitive fallback
        for mid, entry in self._registry.items():
            if mid.lower() == model_id.lower():
                return entry
        return None

    def list_models(self, refresh_status: bool = True) -> List[Dict[str, Any]]:
        """Lists all registered models with up-to-date physical availability status."""
        results: List[Dict[str, Any]] = []
        for mid, entry in self._registry.items():
            if refresh_status:
                if self.is_model_available(mid):
                    entry.status = "LOADED" if mid == self.active_model_id else "AVAILABLE"
                else:
                    entry.status = "NOT_AVAILABLE"

            is_active = (mid == self.active_model_id)
            results.append(entry.to_dict(is_active=is_active))
        return results

    def validate_model(self, model_id: str, run_test_inference: bool = True) -> Dict[str, Any]:
        """
        Validates model configuration, file existence, weight integrity, and inference compatibility.
        Returns structured validation response without leaking raw internal stack traces.
        """
        entry = self.get_model(model_id)
        if not entry:
            return {
                "model_id": model_id,
                "available": False,
                "loadable": False,
                "device": "unknown",
                "status": "INVALID",
                "classes": None,
                "num_classes": 0,
                "inference_tested": False,
                "test_inference_time_ms": None,
                "error": f"Model '{model_id}' is not registered in the model registry.",
                "notes": "Verify models/registry/models.yaml or register the model.",
            }

        resolved_path = self.resolve_weights_path(entry.weights_path)
        if resolved_path is None or not resolved_path.is_file():
            return {
                "model_id": model_id,
                "available": False,
                "loadable": False,
                "device": settings.get_effective_device(),
                "status": "NOT_AVAILABLE",
                "classes": entry.classes or None,
                "num_classes": len(entry.classes) if entry.classes else 0,
                "inference_tested": False,
                "test_inference_time_ms": None,
                "error": f"Weights file '{entry.weights_path}' does not exist on disk.",
                "notes": "Model is planned or weights are not yet downloaded/trained.",
            }

        # Validate extension
        ext = resolved_path.suffix.lower()
        if ext not in self.ALLOWED_WEIGHT_EXTENSIONS:
            return {
                "model_id": model_id,
                "available": True,
                "loadable": False,
                "device": settings.get_effective_device(),
                "status": "INVALID",
                "classes": None,
                "num_classes": 0,
                "inference_tested": False,
                "test_inference_time_ms": None,
                "error": f"Unsupported weights extension '{ext}'. Must be one of {sorted(list(self.ALLOWED_WEIGHT_EXTENSIONS))}.",
                "notes": "Only .pt, .onnx, or .engine weight files are supported.",
            }

        # Attempt to load model
        effective_device = settings.get_effective_device()
        try:
            model = YOLO(str(resolved_path))
            classes_map = model.names or entry.classes or {}
            num_classes = len(classes_map)
        except Exception as e:
            logger.error(f"Validation failure: failed loading weights from '{resolved_path}': {e}", exc_info=True)
            return {
                "model_id": model_id,
                "available": True,
                "loadable": False,
                "device": effective_device,
                "status": "INVALID",
                "classes": None,
                "num_classes": 0,
                "inference_tested": False,
                "test_inference_time_ms": None,
                "error": "Model weights are damaged, corrupt, or incompatible with Ultralytics runtime.",
                "notes": "Check logs for detailed exception trace.",
            }

        # Optional test inference pass
        test_inference_ms: Optional[float] = None
        if run_test_inference:
            try:
                synthetic_img = np.zeros((64, 64, 3), dtype=np.uint8)
                t0 = time.perf_counter()
                model.predict(
                    source=synthetic_img,
                    device=effective_device,
                    conf=entry.recommended_confidence,
                    iou=entry.recommended_iou,
                    imgsz=64,
                    verbose=False,
                )
                test_inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            except Exception as e:
                # If CUDA failed, attempt CPU fallback
                if "cuda" in effective_device.lower():
                    logger.warning(f"Validation test on {effective_device} failed, testing CPU fallback: {e}")
                    try:
                        t0 = time.perf_counter()
                        model.predict(
                            source=synthetic_img,
                            device="cpu",
                            conf=entry.recommended_confidence,
                            iou=entry.recommended_iou,
                            imgsz=64,
                            verbose=False,
                        )
                        test_inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                        effective_device = "cpu"
                    except Exception as cpu_err:
                        logger.error(f"Inference validation failed on both GPU and CPU: {cpu_err}")
                        return {
                            "model_id": model_id,
                            "available": True,
                            "loadable": True,
                            "device": effective_device,
                            "status": "INVALID",
                            "classes": classes_map,
                            "num_classes": num_classes,
                            "inference_tested": False,
                            "test_inference_time_ms": None,
                            "error": "Failed executing forward pass with loaded weights.",
                            "notes": str(cpu_err),
                        }
                else:
                    return {
                        "model_id": model_id,
                        "available": True,
                        "loadable": True,
                        "device": effective_device,
                        "status": "INVALID",
                        "classes": classes_map,
                        "num_classes": num_classes,
                        "inference_tested": False,
                        "test_inference_time_ms": None,
                        "error": "Failed executing forward pass with loaded weights.",
                        "notes": str(e),
                    }

        return {
            "model_id": model_id,
            "available": True,
            "loadable": True,
            "device": effective_device,
            "status": "VALID",
            "classes": {int(k): str(v) for k, v in classes_map.items()},
            "num_classes": num_classes,
            "inference_tested": run_test_inference,
            "test_inference_time_ms": test_inference_ms,
            "error": None,
            "notes": "Model passed validation successfully.",
        }

    def load_model(self, model_id: str) -> Tuple[YOLO, ModelEntry]:
        """
        Loads and caches a YOLO model instance by model ID.
        Reuses cached model instances to prevent repeated reloading.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise ModelNotFoundError(f"Model ID '{model_id}' is not registered.")

        if model_id in self._model_cache:
            return self._model_cache[model_id], entry

        resolved_path = self.resolve_weights_path(entry.weights_path)
        if resolved_path is None or not resolved_path.is_file():
            raise ModelUnavailableError(
                f"Model weights for '{model_id}' are not available at '{entry.weights_path}'."
            )

        try:
            logger.info(f"Loading YOLO weights for '{model_id}' from {resolved_path}...")
            model = YOLO(str(resolved_path))
            self._model_cache[model_id] = model
            return model, entry
        except Exception as e:
            logger.error(f"Failed loading YOLO weights from '{resolved_path}': {e}", exc_info=True)
            raise ModelCorruptError(f"Could not initialize YOLO model '{model_id}': {str(e)}") from e

    def switch_model(self, model_id: str, detector: Optional[Any] = None) -> Dict[str, Any]:
        """
        Switches the active model for detection and tracking pipelines.
        Safely coordinates resource cleanup and updates the detection engine singleton.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise ModelNotFoundError(f"Cannot switch to unregistered model '{model_id}'.")

        if not self.is_model_available(model_id):
            raise ModelUnavailableError(
                f"Cannot switch to '{model_id}': weights file '{entry.weights_path}' does not exist."
            )

        # Validate before switching
        val = self.validate_model(model_id, run_test_inference=False)
        if val["status"] != "VALID":
            raise ModelCorruptError(f"Validation failed for '{model_id}': {val.get('error')}")

        previous_id = self.active_model_id
        yolo_model, model_entry = self.load_model(model_id)

        # Update detection engine instance
        from backend.app.services.vision.detector import get_detector
        target_detector = detector or get_detector()

        target_detector.model = yolo_model
        target_detector.classes_map = yolo_model.names or model_entry.classes or {}
        target_detector.available_classes = set(target_detector.classes_map.values())
        target_detector.model_name = model_entry.model_name
        target_detector.model_path = str(self.resolve_weights_path(model_entry.weights_path))
        target_detector.conf_threshold = model_entry.recommended_confidence
        target_detector.iou_threshold = model_entry.recommended_iou
        target_detector.default_imgsz = model_entry.input_size

        self.active_model_id = model_id

        # Free GPU memory cache if CUDA is active
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info(
            f"Successfully switched active model: '{previous_id}' -> '{model_id}' "
            f"({model_entry.model_name}, {len(target_detector.classes_map)} classes)."
        )

        return {
            "status": "success",
            "previous_model_id": previous_id,
            "active_model_id": model_id,
            "active_model_name": model_entry.model_name,
            "device": target_detector.device,
            "classes_count": len(target_detector.classes_map),
            "message": f"Active detection model switched to '{model_entry.model_name}' successfully.",
        }


# Global model manager singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Dependency / singleton provider for ModelManager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
