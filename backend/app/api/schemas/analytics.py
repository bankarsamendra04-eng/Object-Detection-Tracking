from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class TrajectoryPoint(BaseModel):
    frame_number: int
    timestamp: str
    x: float
    y: float


class LineDefinition(BaseModel):
    line_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(default="Tripwire", max_length=64)
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    direction: str = Field(default="BOTH", pattern=r"^(BOTH|A_TO_B|B_TO_A)$")

    @field_validator("start_point", "end_point")
    @classmethod
    def validate_finite_coords(cls, pt: Tuple[float, float]) -> Tuple[float, float]:
        import math
        x, y = pt
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Coordinates must be finite numbers.")
        if x < 0 or y < 0:
            raise ValueError("Coordinates must be non-negative.")
        return pt


class ROIDefinition(BaseModel):
    roi_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(default="Region", max_length=64)
    points: List[Tuple[float, float]] = Field(..., max_length=100)
    anchor: str = Field(default="bottom_center", pattern=r"^(bottom_center|center|top_left)$")

    @field_validator("points")
    @classmethod
    def validate_polygon_points(cls, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        import math
        for x, y in pts:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("Polygon coordinates must be finite numbers.")
            if x < 0 or y < 0:
                raise ValueError("Polygon coordinates must be non-negative.")
        return pts


class AnalyticsEvent(BaseModel):
    event_id: str
    event_type: str  # "OBJECT_ENTERED", "OBJECT_EXITED", "LINE_CROSSED", "ROI_ENTER", "ROI_EXIT"
    track_id: int
    class_name: str
    timestamp: str
    frame_number: int
    location: Tuple[float, float]
    details: Dict[str, Any] = Field(default_factory=dict)


class TrackStatistics(BaseModel):
    track_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: str
    last_seen_time: str
    duration_seconds: float
    status: str  # "active", "disappeared", "exited"
    total_frames_visible: int
    current_rois: List[str] = Field(default_factory=list)
    roi_dwell_times: Dict[str, float] = Field(default_factory=dict)


class ClassStatistics(BaseModel):
    class_name: str
    active_count: int = 0
    unique_count: int = 0
    entered_count: int = 0
    exited_count: int = 0
    line_crossing_count: int = 0
    roi_entry_count: int = 0
    roi_exit_count: int = 0
    avg_dwell_time_seconds: float = 0.0


class PerformanceMetrics(BaseModel):
    current_fps: float = 0.0
    average_fps: float = 0.0
    last_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    total_frames: int = 0


class AnalyticsSnapshot(BaseModel):
    session_id: str
    frame_number: int
    timestamp: str
    active_objects: int
    total_unique_objects: int
    total_detections: int
    class_distribution: Dict[str, int]
    class_statistics: Dict[str, ClassStatistics]
    active_tracks: List[TrackStatistics]
    line_crossings: Dict[str, int]
    roi_counts: Dict[str, int]
    roi_dwell_times: Dict[str, float]
    performance: PerformanceMetrics
    recent_events: List[AnalyticsEvent]


# Backward-compatible schemas
class LineCrossingEvent(BaseModel):
    track_id: int
    class_name: str
    direction: str  # "inbound", "outbound", "A_TO_B", "B_TO_A"
    timestamp: str


class AnalyticsReport(BaseModel):
    session_id: str
    total_frames_processed: int
    cumulative_unique_objects: int
    current_active_objects: int
    class_distribution: Dict[str, int]
    inbound_count: int = 0
    outbound_count: int = 0
    recent_events: List[LineCrossingEvent] = []
