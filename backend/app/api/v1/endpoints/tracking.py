import io
from pathlib import Path
import shutil
from typing import Any, Dict, Optional
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.vision.detector import DetectionEngine, get_detector
from backend.app.services.vision.tracker import ByteTrackTracker
from backend.app.services.vision.input_sources import VideoInput, MediaSourceError
from backend.app.api.schemas.tracking import TrackingFrameResponse
from backend.app.api.schemas.tracking_api import VideoTrackingResponse
from backend.app.api.deps import get_tracker

logger = get_logger("api.tracking")
router = APIRouter()

MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


@router.post(
    "/video",
    response_model=VideoTrackingResponse,
    summary="Track Objects in Video",
    description="Processes an uploaded video (MP4, AVI, MOV, MKV) using DetectionEngine + ByteTrackTracker with persistent IDs.",
    responses={
        200: {"description": "Successful video tracking output with frame summaries and persistent track IDs."},
        400: {"description": "Invalid video format or corrupt video stream."},
        413: {"description": "Video file exceeds 50 MB limit."},
        422: {"description": "Validation error on stride or max_frames parameters."},
    },
)
async def track_video(
    file: UploadFile = File(..., description="Video file to track (MP4, AVI, MOV, MKV, WEBM)"),
    session_id: Optional[str] = Form(None, description="Optional tracking session ID"),
    confidence: Optional[float] = Form(None, description="Detection confidence filter (0.0 to 1.0)"),
    iou: Optional[float] = Form(None, description="IoU threshold (0.0 to 1.0)"),
    stride: int = Form(1, ge=1, le=10, description="Process every Nth frame"),
    max_frames: int = Form(60, ge=1, le=300, description="Maximum frames to process in REST call"),
    detector: DetectionEngine = Depends(get_detector),
) -> VideoTrackingResponse:
    """Processes uploaded video with ByteTrack tracking."""
    # 1. Parameter Validation
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Confidence threshold must be between 0.0 and 1.0. Received: {confidence}",
        )
    if iou is not None and not (0.0 <= iou <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"IoU threshold must be between 0.0 and 1.0. Received: {iou}",
        )

    # 2. File Format Validation
    filename = file.filename or "uploaded_video.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in VideoInput.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension '{ext}'. Supported formats: {sorted(VideoInput.SUPPORTED_EXTENSIONS)}",
        )

    if file.content_type and not (file.content_type.startswith("video/") or file.content_type == "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME content-type '{file.content_type}'. Must be a video format.",
        )

    active_session_id = session_id or f"vid_{uuid.uuid4().hex[:8]}"
    tracker = ByteTrackTracker(session_id=active_session_id)

    # 3. Safe temporary file write with size limit (pure random UUID name to prevent path traversal)
    upload_dir = settings.resolve_path(settings.UPLOADS_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"track_temp_{uuid.uuid4().hex}{ext}"

    file_size = 0
    try:
        with open(temp_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video file exceeds maximum allowed size of 50 MB.",
                    )
                f_out.write(chunk)

        # 4. Process frames through VideoInput -> DetectionEngine -> ByteTrackTracker
        with VideoInput(temp_path, stride=stride, max_frames=max_frames) as vid_in:
            sample_frames: list = []
            aggregated_classes: Dict[str, int] = {}
            total_tracks_count = 0

            for frame_num, frame, timestamp_ms in vid_in:
                det_result = detector.detect(
                    image=frame,
                    conf=confidence,
                    iou=iou,
                    frame_number=frame_num,
                )
                track_result = tracker.update(det_result)

                total_tracks_count += track_result.active_track_count
                for cls_name, cnt in track_result.class_counts.items():
                    aggregated_classes[cls_name] = aggregated_classes.get(cls_name, 0) + cnt

                if len(sample_frames) < 10:
                    sample_frames.append({
                        "frame_number": frame_num,
                        "timestamp_ms": timestamp_ms,
                        "inference_time_ms": det_result.inference_time_ms,
                        "active_tracks_count": track_result.active_track_count,
                        "tracks": [t.to_dict() for t in track_result.objects],
                    })

            return VideoTrackingResponse(
                status="success",
                session_id=active_session_id,
                video_metadata=vid_in.metadata,
                frames_processed=len(sample_frames),
                total_tracks_observed=total_tracks_count,
                cumulative_unique_tracks=len(tracker.cumulative_unique_ids),
                class_counts=aggregated_classes,
                sample_frames=sample_frames,
            )

    except MediaSourceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.post(
    "/frame",
    response_model=TrackingFrameResponse,
    summary="Track Objects in Single Frame",
    description="Processes an individual image frame through a persistent session tracker.",
)
async def track_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    detector: DetectionEngine = Depends(get_detector),
) -> TrackingFrameResponse:
    """Processes an individual frame through the session tracker."""
    active_session_id = session_id or "default_session"
    tracker = get_tracker(active_session_id)

    try:
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image frame: {str(e)}")

    det_result = detector.detect(pil_image, conf=confidence, iou=iou)
    track_result = tracker.update(det_result)

    from backend.app.api.schemas.detection import BoundingBox
    from backend.app.api.schemas.tracking import TrackedItem

    tracked_items = [
        TrackedItem(
            track_id=o.track_id,
            class_id=o.class_id,
            class_name=o.class_name,
            confidence=o.confidence,
            box=BoundingBox(x1=o.x1, y1=o.y1, x2=o.x2, y2=o.y2),
            trajectory=o.trajectory,
        )
        for o in track_result.objects
    ]

    return TrackingFrameResponse(
        session_id=active_session_id,
        frame_number=track_result.frame_number,
        inference_time_ms=det_result.inference_time_ms,
        device=det_result.device_used,
        active_tracks_count=track_result.active_track_count,
        cumulative_unique_tracks=track_result.cumulative_unique_tracks,
        class_counts=track_result.class_counts,
        tracks=tracked_items,
    )


@router.post(
    "/reset",
    summary="Reset Tracking Session",
    description="Resets the internal tracking state and buffers for a given session ID.",
)
def reset_tracking(session_id: str = Query("default_session")) -> Dict[str, str]:
    """Resets tracking session state."""
    tracker = get_tracker(session_id)
    tracker.reset()
    return {"status": "success", "message": f"Tracking session '{session_id}' reset successfully."}
