from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.db.session import get_db
from backend.app.db.models import DetectionRecord
from backend.app.api.deps import get_analytics_engine
from backend.app.services.analytics.counter import (
    AnalyticsEngine,
    InvalidLineError,
    InvalidROIError,
)
from backend.app.api.schemas.analytics import (
    AnalyticsEvent,
    AnalyticsSnapshot,
    LineDefinition,
    ROIDefinition,
)

router = APIRouter()


@router.get(
    "/current",
    response_model=AnalyticsSnapshot,
    summary="Get Current Analytics Snapshot",
    description="Returns real-time analytics snapshot including active counts, unique counts, dwell times, and spatial events.",
)
def get_current_analytics(
    session_id: str = Query("default_session", description="Session identifier"),
) -> AnalyticsSnapshot:
    """Returns the current real-time analytics snapshot."""
    engine = get_analytics_engine(session_id)
    return engine.get_snapshot()


@router.get(
    "/snapshot",
    response_model=AnalyticsSnapshot,
    summary="Get Analytics Snapshot",
    description="Alias for current real-time analytics snapshot.",
)
def get_analytics_snapshot(
    session_id: str = Query("default_session", description="Session identifier"),
) -> AnalyticsSnapshot:
    """Returns the current real-time analytics snapshot."""
    engine = get_analytics_engine(session_id)
    return engine.get_snapshot()


@router.get(
    "/report",
    summary="Get Analytics Report",
    description="Returns comprehensive telemetry report for dashboard KPI cards and graphs.",
)
def get_analytics_report(
    session_id: str = Query("default_session", description="Session identifier"),
) -> Dict[str, Any]:
    """Generates structured analytics report for the dashboard."""
    engine = get_analytics_engine(session_id)
    snapshot = engine.get_snapshot()

    inbound = sum(
        1 for e in snapshot.recent_events
        if e.event_type == "LINE_CROSSED" and e.details.get("direction") in ("inbound", "A_TO_B")
    )
    outbound = sum(
        1 for e in snapshot.recent_events
        if e.event_type == "LINE_CROSSED" and e.details.get("direction") in ("outbound", "B_TO_A")
    )
    total_crossings = sum(snapshot.line_crossings.values()) if snapshot.line_crossings else (inbound + outbound)

    return {
        "status": "success",
        "session_id": session_id,
        "overall_summary": {
            "total_unique_objects": snapshot.total_unique_objects,
            "active_objects": snapshot.active_objects,
            "total_detections": snapshot.total_detections,
            "total_frames_processed": snapshot.frame_number,
        },
        "class_statistics": snapshot.class_statistics,
        "line_crossing": {
            "total_crossings": total_crossings,
            "direction_breakdown": {
                "INBOUND": inbound,
                "OUTBOUND": outbound,
            },
            "line_counts": snapshot.line_crossings,
        },
        "performance": snapshot.performance.model_dump(),
        "recent_events": [e.model_dump() for e in snapshot.recent_events],
    }


@router.get(
    "/summary",
    summary="Get Historical Summary",
    description="Returns aggregated database logs and performance telemetry.",
)
def get_historical_summary(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    session_id: str = Query("default_session"),
) -> Dict[str, Any]:
    """Retrieves historical detection records and combines with engine snapshot totals."""
    total_records = db.query(DetectionRecord).count()
    avg_inference = db.query(func.avg(DetectionRecord.inference_time_ms)).scalar() or 0.0

    recent_records = (
        db.query(DetectionRecord)
        .order_by(DetectionRecord.id.desc())
        .limit(limit)
        .all()
    )

    engine = get_analytics_engine(session_id)
    snapshot = engine.get_snapshot()

    return {
        "status": "success",
        "session_id": session_id,
        "database_records_count": total_records,
        "average_inference_time_ms": round(float(avg_inference), 2),
        "cumulative_unique_objects": snapshot.total_unique_objects,
        "current_active_objects": snapshot.active_objects,
        "class_distribution": snapshot.class_distribution,
        "recent_database_records": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "total_detections": r.total_detections,
                "inference_time_ms": r.inference_time_ms,
                "class_counts": r.class_counts,
            }
            for r in recent_records
        ],
    }


@router.post(
    "/reset",
    summary="Reset Analytics Session",
    description="Resets all track counters, event logs, and dwell timers for the specified session.",
)
def reset_analytics(
    session_id: str = Query("default_session", description="Session to reset"),
) -> Dict[str, str]:
    """Resets the analytics engine state."""
    engine = get_analytics_engine(session_id)
    engine.reset()
    return {"status": "success", "message": f"Analytics session '{session_id}' reset successfully."}


@router.post(
    "/line",
    summary="Register Virtual Line Tripwire",
    description="Defines a new virtual line boundary for crossing detection and direction estimation.",
    responses={
        200: {"description": "Line registered successfully."},
        400: {"description": "Invalid line endpoints provided."},
    },
)
def add_line(
    line: LineDefinition,
    session_id: str = Query("default_session"),
) -> Dict[str, Any]:
    """Adds a virtual tripwire line."""
    engine = get_analytics_engine(session_id)
    try:
        engine.add_line(line)
        return {"status": "success", "message": f"Line '{line.line_id}' registered.", "line": line.model_dump()}
    except InvalidLineError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete(
    "/line/{line_id}",
    summary="Remove Virtual Line",
    description="Removes an existing virtual line tripwire.",
)
def remove_line(
    line_id: str,
    session_id: str = Query("default_session"),
) -> Dict[str, str]:
    """Removes a virtual line."""
    engine = get_analytics_engine(session_id)
    engine.remove_line(line_id)
    return {"status": "success", "message": f"Line '{line_id}' removed."}


@router.post(
    "/roi",
    summary="Register Region of Interest (ROI)",
    description="Registers an arbitrary polygon region for spatial containment, enter/exit events, and dwell time calculation.",
    responses={
        200: {"description": "ROI registered successfully."},
        400: {"description": "Invalid polygon vertices (< 3 points)."},
    },
)
def add_roi(
    roi: ROIDefinition,
    session_id: str = Query("default_session"),
) -> Dict[str, Any]:
    """Adds an ROI polygon."""
    engine = get_analytics_engine(session_id)
    try:
        engine.add_roi(roi)
        return {"status": "success", "message": f"ROI '{roi.roi_id}' registered.", "roi": roi.model_dump()}
    except InvalidROIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete(
    "/roi/{roi_id}",
    summary="Remove Region of Interest",
    description="Removes an existing ROI polygon.",
)
def remove_roi(
    roi_id: str,
    session_id: str = Query("default_session"),
) -> Dict[str, str]:
    """Removes an ROI polygon."""
    engine = get_analytics_engine(session_id)
    engine.remove_roi(roi_id)
    return {"status": "success", "message": f"ROI '{roi_id}' removed."}


@router.get(
    "/events",
    response_model=List[AnalyticsEvent],
    summary="Get Analytics Events",
    description="Returns filtered recent analytics events (OBJECT_ENTERED, OBJECT_EXITED, LINE_CROSSED, ROI_ENTER, ROI_EXIT).",
)
def get_analytics_events(
    session_id: str = Query("default_session"),
    limit: int = Query(20, ge=1, le=100),
    event_type: Optional[str] = Query(None, description="Optional filter by event type"),
) -> List[AnalyticsEvent]:
    """Returns recent analytics events."""
    engine = get_analytics_engine(session_id)
    return engine.get_events(limit=limit, event_type=event_type)
