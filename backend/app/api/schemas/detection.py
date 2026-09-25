from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    track_id: Optional[int] = None


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    inference_time_ms: float
    device: str
    total_detections: int
    class_counts: Dict[str, int]
    detections: List[DetectionItem]


class ImageDetectionConfig(BaseModel):
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_classes: Optional[List[str]] = None
