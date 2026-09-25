from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from backend.app.api.schemas.detection import BoundingBox


class TrackedItemResponse(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    tracking_state: str = "active"
    trajectory: Optional[List[Tuple[float, float]]] = None


class VideoTrackingResponse(BaseModel):
    status: str = "success"
    session_id: str
    video_metadata: Dict[str, Any]
    frames_processed: int
    total_tracks_observed: int
    cumulative_unique_tracks: int
    class_counts: Dict[str, int]
    sample_frames: List[Dict[str, Any]]
