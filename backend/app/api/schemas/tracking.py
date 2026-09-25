from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from backend.app.api.schemas.detection import BoundingBox, DetectionItem


class TrackedItem(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    trajectory: Optional[List[Tuple[float, float]]] = None


class TrackingFrameResponse(BaseModel):
    session_id: str
    frame_number: int
    inference_time_ms: float
    device: str
    active_tracks_count: int
    cumulative_unique_tracks: int
    class_counts: Dict[str, int]
    tracks: List[TrackedItem]


class TrackingConfig(BaseModel):
    tracker_type: Optional[str] = "bytetrack.yaml"
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_classes: Optional[List[str]] = None
