from typing import Any, Dict
from fastapi import APIRouter, Query

from backend.app.core.config import settings
from backend.app.api.schemas.system import ConfigurationResponse, SourcesResponse
from backend.app.services.vision.input_sources import (
    ImageInput,
    VideoInput,
    probe_camera_availability,
)

router = APIRouter()


@router.get(
    "/config",
    response_model=ConfigurationResponse,
    summary="Get Runtime Configuration",
    description="Returns public non-sensitive runtime parameters (model, device, confidence, IoU, tracker).",
)
def get_public_configuration() -> ConfigurationResponse:
    """Returns safe, public runtime configuration."""
    return ConfigurationResponse(
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        debug=settings.DEBUG,
        model_name=settings.MODEL_NAME,
        active_model_id=settings.ACTIVE_MODEL_ID,
        model_type=settings.MODEL_TYPE,
        input_size=settings.INFERENCE_IMG_SIZE,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        iou_threshold=settings.IOU_THRESHOLD,
        device=settings.get_effective_device(),
        tracker_type=settings.TRACKER_TYPE,
        track_high_thresh=settings.TRACK_HIGH_THRESH,
        track_low_thresh=settings.TRACK_LOW_THRESH,
        track_match_thresh=settings.TRACK_MATCH_THRESH,
        track_persistence_buffer=settings.TRACK_PERSISTENCE_BUFFER,
        supported_image_formats=sorted(list(ImageInput.SUPPORTED_EXTENSIONS)),
        supported_video_formats=sorted(list(VideoInput.SUPPORTED_EXTENSIONS)),
    )


@router.get(
    "/sources",
    response_model=SourcesResponse,
    summary="Get Supported Input Sources",
    description="Lists supported media ingestion modes and file extensions without acquiring hardware locks.",
)
def get_supported_sources() -> SourcesResponse:
    """Lists available input modes and supported extensions."""
    return SourcesResponse(
        supported_sources=["image", "video", "webcam"],
        supported_image_formats=sorted(list(ImageInput.SUPPORTED_EXTENSIONS)),
        supported_video_formats=sorted(list(VideoInput.SUPPORTED_EXTENSIONS)),
        webcam_config={
            "default_index": 0,
            "probe_endpoint": "/api/v1/sources/webcam/probe",
            "recommended_backend": "DirectShow (Windows) / V4L2 (Linux)",
        },
    )


@router.get(
    "/sources/webcam/probe",
    summary="Probe Camera Hardware",
    description="Safely tests camera device availability without keeping the stream locked.",
)
def probe_webcam(
    camera_index: int = Query(0, ge=0, le=32, description="Camera device index to probe")
) -> Dict[str, Any]:
    """Safely probes camera hardware status."""
    return probe_camera_availability(camera_index=camera_index)
