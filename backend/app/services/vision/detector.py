from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from backend.app.core.config import settings
from backend.app.core.logging import get_logger

logger = get_logger("vision.detector")


# ==========================================
# Custom Exception Hierarchy
# ==========================================

class DetectionError(Exception):
    """Base exception for all detection engine failures."""
    pass


class ModelLoadError(DetectionError):
    """Raised when the specified YOLO model weights cannot be located or loaded."""
    pass


class InvalidImageError(DetectionError):
    """Raised when an input image or frame is corrupt, empty, or has invalid dimensions."""
    pass


class InferenceError(DetectionError):
    """Raised when the underlying YOLO runtime fails during the forward pass."""
    pass


# ==========================================
# Core Detection Data Structures
# ==========================================

@dataclass
class Detection:
    """Represents a single structured object detection independent of rendering or UI."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_number: Optional[int] = None
    timestamp: Optional[str] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes detection item to dictionary format."""
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "box": {
                "x1": round(float(self.x1), 2),
                "y1": round(float(self.y1), 2),
                "x2": round(float(self.x2), 2),
                "y2": round(float(self.y2), 2),
            },
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "frame_number": int(self.frame_number) if self.frame_number is not None else None,
            "timestamp": self.timestamp,
        }


@dataclass
class DetectionResult:
    """Contains all detections and execution metadata for an image or video frame."""
    detections: List[Detection]
    image_width: int
    image_height: int
    inference_time_ms: float
    device_used: str
    model_name: str
    frame_number: Optional[int] = None
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def filter_by_class(self, class_names: Union[str, List[str], Set[str]]) -> List[Detection]:
        """Filters detections by specific class name(s)."""
        target = {class_names} if isinstance(class_names, str) else set(class_names)
        return [d for d in self.detections if d.class_name in target]

    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Filters detections meeting or exceeding a minimum confidence score."""
        return [d for d in self.detections if d.confidence >= min_confidence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts result into structured dictionary compatible with API schemas."""
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": round(float(self.inference_time_ms), 2),
            "device": self.device_used,
            "device_used": self.device_used,
            "model_name": self.model_name,
            "total_detections": self.total_detections,
            "class_counts": self.class_counts,
            "detections": [d.to_dict() for d in self.detections],
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
        }


# ==========================================
# Detection Engine
# ==========================================

class DetectionEngine:
    """
    Production-grade Object Detection Engine wrapping Ultralytics YOLO models.
    Decoupled from presentation and web layers with robust error handling,
    dynamic threshold configuration, hardware auto-routing, and batch support.
    """

    MANDATORY_COCO_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        half_precision: Optional[bool] = None,
        max_detections: Optional[int] = None,
        imgsz: Optional[int] = None,
    ):
        self.model_name = model_name_or_path or settings.MODEL_NAME
        self.model_path = settings.get_resolved_model_path(model_name_or_path) if model_name_or_path is not None else settings.get_resolved_model_path()
        self.device = device or settings.get_effective_device()
        self.conf_threshold = conf_threshold if conf_threshold is not None else settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.IOU_THRESHOLD
        self.half_precision = half_precision if half_precision is not None else settings.HALF_PRECISION
        self.max_detections = max_detections if max_detections is not None else settings.MAX_DETECTIONS
        self.default_imgsz = imgsz if imgsz is not None else settings.INFERENCE_IMG_SIZE

        logger.info(
            f"Initializing DetectionEngine: model={self.model_path} ({self.model_name}), "
            f"device={self.device}, default_conf={self.conf_threshold}, default_iou={self.iou_threshold}, "
            f"imgsz={self.default_imgsz}"
        )

        self._load_yolo_model(self.model_path)

    def _load_yolo_model(self, model_path: str) -> None:
        """Loads and verifies a YOLO weights file, updating internal class maps."""
        # Check file extension
        ext = Path(model_path).suffix.lower()
        if ext and ext not in {".pt", ".onnx", ".engine"}:
            raise ModelLoadError(f"Unsupported model weight extension '{ext}'. Must be .pt, .onnx, or .engine.")

        try:
            self.model = YOLO(model_path)
            if self.half_precision and "cuda" in self.device and hasattr(self.model, "model") and self.model.model is not None:
                try:
                    self.model.model.half()
                except Exception as e:
                    logger.debug(f"Could not convert model to half precision: {e}")
            self.classes_map: Dict[int, str] = self.model.names or {}
            self.available_classes: Set[str] = set(self.classes_map.values())
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise ModelLoadError(f"Could not load detection model '{model_path}': {str(e)}") from e

        # Validate minimum classes for standard models
        missing_mandatory = self.MANDATORY_COCO_CLASSES - self.available_classes
        if missing_mandatory:
            logger.info(
                f"Model '{self.model_name}' does not contain standard COCO classes: {missing_mandatory}. "
                f"Active classes ({len(self.classes_map)}): {sorted(list(self.available_classes))[:5]}... "
                f"This is expected for specialized domain models (e.g. VisDrone aerial dataset)."
            )
        else:
            logger.info(f"Verified support for mandatory classes: {self.MANDATORY_COCO_CLASSES}")

        logger.info(f"DetectionEngine loaded successfully with {len(self.classes_map)} classes.")

    def switch_model(
        self,
        model_name_or_path: str,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        """
        Dynamically switches active model weights and updates threshold defaults.
        Releases GPU memory cache when running on CUDA.
        """
        resolved_path = settings.get_resolved_model_path(model_name_or_path)
        logger.info(f"Switching DetectionEngine model: {self.model_path} -> {resolved_path}")
        self._load_yolo_model(resolved_path)
        self.model_path = resolved_path
        self.model_name = model_name or Path(model_name_or_path).name

        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if imgsz is not None:
            self.default_imgsz = imgsz

        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def validate_input(self, image: Any) -> Union[np.ndarray, Image.Image, str]:
        """Validates input frame/image and ensures it is well-formed for inference."""
        if image is None:
            raise InvalidImageError("Input image cannot be None.")

        if isinstance(image, str):
            if not os.path.exists(image):
                raise InvalidImageError(f"Image path does not exist: {image}")
            if os.path.isdir(image):
                raise InvalidImageError(f"Target path is a directory, not an image file: {image}")
            return image

        if isinstance(image, Path):
            if not image.is_file():
                raise InvalidImageError(f"Image file does not exist: {image}")
            return str(image)

        if isinstance(image, Image.Image):
            if image.width <= 0 or image.height <= 0:
                raise InvalidImageError(f"PIL Image has invalid dimensions: {image.width}x{image.height}")
            return image

        if isinstance(image, np.ndarray):
            if image.size == 0 or len(image.shape) < 2:
                raise InvalidImageError(f"NumPy array is empty or has invalid shape: {image.shape}")
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                raise InvalidImageError(f"NumPy array dimensions are invalid: h={h}, w={w}")
            return image

        raise InvalidImageError(
            f"Unsupported image type '{type(image).__name__}'. Supported types: np.ndarray, PIL.Image, file path (str/Path)."
        )

    def detect(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
        frame_number: Optional[int] = None,
        imgsz: Optional[int] = None,
    ) -> DetectionResult:
        """
        Executes object detection on an input image/frame and returns a structured DetectionResult.
        
        Args:
            image: Image array, PIL Image, or file path.
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.
            target_classes: Optional list of class names to filter.
            frame_number: Optional frame index for video streams.
            imgsz: Optional inference resolution override (e.g. 1280 for small-object recall).
        """
        valid_input = self.validate_input(image)
        start_time = time.perf_counter()

        effective_conf = conf if conf is not None else self.conf_threshold
        effective_iou = iou if iou is not None else self.iou_threshold
        effective_imgsz = imgsz if imgsz is not None else self.default_imgsz
        target_set = set(target_classes) if target_classes else None

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=valid_input,
                    conf=effective_conf,
                    iou=effective_iou,
                    imgsz=effective_imgsz,
                    device=self.device,
                    max_det=self.max_detections,
                    verbose=False,
                )
        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            raise InferenceError(f"Failed executing YOLO model prediction: {str(e)}") from e

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        now_iso = datetime.now(timezone.utc).isoformat()

        if not results:
            return DetectionResult(
                detections=[],
                image_width=0,
                image_height=0,
                inference_time_ms=inference_time_ms,
                device_used=self.device,
                model_name=self.model_name,
                frame_number=frame_number,
                timestamp=now_iso,
            )

        res = results[0]
        orig_shape = res.orig_shape  # (height, width)
        h, w = int(orig_shape[0]), int(orig_shape[1])

        detections: List[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0 and hasattr(res.boxes, "data") and res.boxes.data is not None:
            # Single contiguous PCIe transfer for all bounding box data (N, 6): [x1, y1, x2, y2, conf, cls]
            boxes_data = res.boxes.data.cpu().numpy()

            for row in boxes_data:
                x1, y1, x2, y2, score, cls_id = row
                cls_id_int = int(cls_id)
                class_name = self.classes_map.get(cls_id_int, f"class_{cls_id_int}")

                if target_set and class_name not in target_set:
                    continue

                detections.append(
                    Detection(
                        class_id=cls_id_int,
                        class_name=class_name,
                        confidence=float(score),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        frame_number=int(frame_number) if frame_number is not None else None,
                        timestamp=now_iso,
                    )
                )

        return DetectionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            inference_time_ms=inference_time_ms,
            device_used=self.device,
            model_name=self.model_name,
            frame_number=frame_number,
            timestamp=now_iso,
        )

    def annotate(
        self,
        image: np.ndarray,
        result: DetectionResult,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Utility method to render bounding boxes and labels onto a copy of an OpenCV BGR image."""
        canvas = image.copy()
        for det in result.detections:
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(canvas, p1, p2, box_color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, p1, (p1[0] + text_w, p1[1] - text_h - 4), box_color, -1)
            cv2.putText(canvas, label, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        return canvas


# Backward-compatible alias
YOLODetector = DetectionEngine

# Global detection engine singleton
_default_engine: Optional[DetectionEngine] = None


def get_detector() -> DetectionEngine:
    """Dependency / singleton provider for DetectionEngine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = DetectionEngine()
    return _default_engine
