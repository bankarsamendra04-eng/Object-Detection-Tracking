from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.db.session import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DetectionSession(Base):
    """Represents a live stream, webcam feed, or processed video session."""
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    source_type = Column(String(32), nullable=False)  # "webcam", "video", "image"
    source_name = Column(String(255), nullable=True)
    status = Column(String(32), default="active")  # "active", "completed", "aborted"
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    records = relationship("DetectionRecord", back_populates="session", cascade="all, delete-orphan")
    tracked_objects = relationship("TrackedObject", back_populates="session", cascade="all, delete-orphan")


class DetectionRecord(Base):
    """Stores batch detection snapshots and frame metadata."""
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"), nullable=True)
    frame_number = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=utc_now)
    total_detections = Column(Integer, default=0)
    inference_time_ms = Column(Float, nullable=False)
    class_counts = Column(JSON, default=dict)
    detections_detail = Column(JSON, default=list)

    # Relationships
    session = relationship("DetectionSession", back_populates="records")


class TrackedObject(Base):
    """Stores unique tracked entity lifecycle across a session."""
    __tablename__ = "tracked_objects"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"), nullable=False)
    track_id = Column(Integer, nullable=False, index=True)
    class_name = Column(String(64), nullable=False)
    first_seen = Column(DateTime, default=utc_now)
    last_seen = Column(DateTime, default=utc_now, onupdate=utc_now)
    total_frames = Column(Integer, default=1)
    trajectories = Column(JSON, default=list)

    # Relationships
    session = relationship("DetectionSession", back_populates="tracked_objects")
