from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    PRETRAINED = "pretrained"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    LOADED = "LOADED"
    ERROR = "ERROR"


class ModelInfoResponse(BaseModel):
    """Detailed model metadata and runtime availability."""
    model_id: str = Field(..., description="Unique registered model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(..., description="Model category (pretrained or custom)")
    version: str = Field(..., description="Model version string")
    framework: str = Field(..., description="Framework engine (e.g. ultralytics_yolo)")
    task: str = Field(..., description="Vision task (e.g. object_detection)")
    weights_path: str = Field(..., description="Relative or resolved weights file path")
    source: str = Field(..., description="Origin source or training lineage")
    input_size: int = Field(640, description="Recommended image inference dimension")
    recommended_confidence: float = Field(0.35, description="Recommended detection confidence threshold")
    recommended_iou: float = Field(0.45, description="Recommended NMS IoU threshold")
    device: str = Field("auto", description="Configured execution device target")
    status: str = Field(..., description="Availability status (AVAILABLE, NOT_AVAILABLE, etc.)")
    num_classes: int = Field(0, description="Total number of supported classes")
    classes: Dict[int, str] = Field(default_factory=dict, description="Class ID to class name mapping")
    is_active: bool = Field(False, description="Whether this model is currently active in the DetectionEngine")
    notes: Optional[str] = Field(None, description="Operational and architecture notes")


class ModelListResponse(BaseModel):
    """List of all registered models and active detector state."""
    active_model_id: str = Field(..., description="Currently active model ID")
    total_models: int = Field(..., description="Total count of models in the registry")
    models: List[ModelInfoResponse] = Field(..., description="List of registered model profiles")


class ModelValidationResponse(BaseModel):
    """Structured validation report for a registered model."""
    model_id: str = Field(..., description="Validated model identifier")
    available: bool = Field(..., description="Whether weights file physically exists on disk")
    loadable: bool = Field(..., description="Whether weights can be instantiated without runtime errors")
    device: str = Field(..., description="Resolved execution device verified during validation")
    status: str = Field(..., description="Validation outcome (VALID, INVALID, NOT_AVAILABLE)")
    classes: Optional[Dict[int, str]] = Field(default=None, description="Extracted classes if loadable")
    num_classes: int = Field(0, description="Total classes verified")
    inference_tested: bool = Field(False, description="Whether a synthetic test forward pass succeeded")
    test_inference_time_ms: Optional[float] = Field(None, description="Latency of test forward pass in ms")
    error: Optional[str] = Field(None, description="Sanitized, safe error description if invalid")
    notes: Optional[str] = Field(None, description="Validation diagnosis notes")


class ModelSwitchRequest(BaseModel):
    """Request payload to switch the active detection model."""
    model_id: str = Field(..., description="Target registered model identifier to activate")


class ModelSwitchResponse(BaseModel):
    """Response returned upon switching active detection model."""
    status: str = Field("success", description="Switch operation status")
    previous_model_id: str = Field(..., description="Previous active model ID")
    active_model_id: str = Field(..., description="Newly activated model ID")
    active_model_name: str = Field(..., description="Newly activated model name")
    device: str = Field(..., description="Device used by new active model")
    classes_count: int = Field(..., description="Number of classes recognized by active model")
    message: str = Field(..., description="Informative status message")
