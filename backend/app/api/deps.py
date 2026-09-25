import time
from typing import Dict, Generator, Optional
from sqlalchemy.orm import Session

from backend.app.db.session import SessionLocal
from backend.app.services.vision.detector import DetectionEngine, get_detector
from backend.app.services.vision.model_manager import ModelManager, get_model_manager
from backend.app.services.vision.tracker import ByteTrackTracker
from backend.app.services.analytics.counter import AnalyticsEngine

# Server initialization timestamp for uptime calculation
SERVER_START_TIME = time.time()

# Shared session caches
_analytics_engines: Dict[str, AnalyticsEngine] = {}
_trackers: Dict[str, ByteTrackTracker] = {}


def get_db() -> Generator[Session, None, None]:
    """Database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_analytics_engine(session_id: str = "default_session") -> AnalyticsEngine:
    """Returns or creates a stateful AnalyticsEngine for a given session."""
    if session_id not in _analytics_engines:
        _analytics_engines[session_id] = AnalyticsEngine(session_id=session_id)
    return _analytics_engines[session_id]


def get_tracker(session_id: str = "default_session") -> ByteTrackTracker:
    """Returns or creates a stateful ByteTrackTracker for a given session."""
    if session_id not in _trackers:
        _trackers[session_id] = ByteTrackTracker(session_id=session_id)
    return _trackers[session_id]


def get_uptime_seconds() -> float:
    """Returns application uptime in seconds."""
    return round(time.time() - SERVER_START_TIME, 2)
