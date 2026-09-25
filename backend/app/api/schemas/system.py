from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    api_status: str = "operational"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GPUInfo(BaseModel):
    device_name: str
    memory_allocated_mb: float
    memory_reserved_mb: float


class ModelInfo(BaseModel):
    name: str
    classes_count: int
    loaded: bool = True


class StatusResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    environment: str
    version: str = "1.0.0"
    uptime_seconds: float
    python_version: str
    effective_device: str
    gpu: Optional[GPUInfo] = None
    model: ModelInfo
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConfigurationResponse(BaseModel):
    app_name: str
    environment: str
    debug: bool
    model_name: str
    active_model_id: Optional[str] = "general_pretrained"
    model_type: Optional[str] = "pretrained"
    input_size: Optional[int] = 640
    confidence_threshold: float
    iou_threshold: float
    device: str
    tracker_type: str
    track_high_thresh: Optional[float] = 0.5
    track_low_thresh: Optional[float] = 0.1
    track_match_thresh: Optional[float] = 0.8
    track_persistence_buffer: Optional[int] = 30
    supported_image_formats: List[str]
    supported_video_formats: List[str]


class SourcesResponse(BaseModel):
    supported_sources: List[str] = ["image", "video", "webcam"]
    supported_image_formats: List[str]
    supported_video_formats: List[str]
    webcam_config: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
