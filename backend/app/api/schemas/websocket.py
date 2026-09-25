from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class ClientAction(str, Enum):
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    PING = "ping"


class StreamSourceType(str, Enum):
    WEBCAM = "webcam"
    VIDEO = "video"


class ServerMessageType(str, Enum):
    CONNECTION_ACK = "connection_ack"
    STREAM_STARTED = "stream_started"
    FRAME_RESULT = "frame_result"
    ANALYTICS_UPDATE = "analytics_update"
    STREAM_STOPPED = "stream_stopped"
    ERROR = "error"
    PONG = "pong"


# ==========================================
# Inbound Client Commands
# ==========================================

class ClientCommand(BaseModel):
    """Schema validating inbound client commands over WebSocket."""
    action: ClientAction = Field(..., description="Action to perform: start, stop, pause, resume, ping")
    source: Optional[StreamSourceType] = Field(default=StreamSourceType.WEBCAM, description="Source type: webcam or video")
    camera_index: int = Field(default=0, ge=0, description="Webcam hardware device index")
    path: Optional[str] = Field(default=None, max_length=512, description="Relative path to video file within data directory")
    stride: int = Field(default=1, ge=1, le=60, description="Process every Nth frame")
    max_frames: Optional[int] = Field(default=None, ge=1, le=10000, description="Optional limit of frames to process")
    fps_limit: float = Field(default=15.0, ge=1.0, le=60.0, description="Target streaming frame rate limit")
    conf: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override detection confidence threshold")
    iou: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override NMS IoU threshold")
    target_classes: Optional[List[str]] = Field(default=None, max_length=100, description="Optional class filter (e.g. ['person', 'car'])")
    annotate: bool = Field(default=False, description="Whether to include base64 JPEG annotated frame in message")

    @field_validator("path")
    @classmethod
    def validate_safe_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_clean = v.strip()
            if not v_clean:
                raise ValueError("Path cannot be empty.")
            if "\x00" in v_clean:
                raise ValueError("Null bytes are forbidden in paths.")
            if v_clean.startswith(("\\\\", "//")):
                raise ValueError("UNC network paths are strictly forbidden.")
            if len(v_clean) > 1 and v_clean[1] == ":":
                raise ValueError("Drive letters and absolute paths are strictly forbidden.")
            if v_clean.startswith(("/", "\\")):
                raise ValueError("Root path specifiers are strictly forbidden.")
            if ".." in v_clean.replace("\\", "/").split("/"):
                raise ValueError("Path traversal ('..') is strictly forbidden.")
        return v


# ==========================================
# Outbound Server Message Payloads
# ==========================================

class ConnectionAckData(BaseModel):
    session_id: str
    message: str = "Connected to Real-time CV Streaming Service"
    supported_sources: List[str] = ["webcam", "video"]
    server_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StreamStartedData(BaseModel):
    session_id: str
    source: str
    metadata: Dict[str, Any]
    fps_limit: float


class FrameResultData(BaseModel):
    frame_number: int
    timestamp_ms: float
    inference_time_ms: float
    tracking_time_ms: float
    total_time_ms: float
    fps: float
    active_tracks: int
    unique_tracks: int
    detections: List[Dict[str, Any]]
    tracks: List[Dict[str, Any]]
    analytics: Dict[str, Any]
    annotated_frame: Optional[str] = None


class StreamStoppedData(BaseModel):
    session_id: str
    reason: str
    frames_processed: int
    duration_seconds: float


class StreamErrorData(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ==========================================
# Master WebSocket Message Envelope
# ==========================================

class WebSocketMessage(BaseModel):
    """Standardized envelope for all WebSocket messages sent to the client."""
    type: ServerMessageType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str
    data: Union[
        ConnectionAckData,
        StreamStartedData,
        FrameResultData,
        StreamStoppedData,
        StreamErrorData,
        Dict[str, Any],
    ]
