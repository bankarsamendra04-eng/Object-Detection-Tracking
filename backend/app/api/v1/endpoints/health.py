from datetime import datetime, timezone
import sys
from typing import Any, Dict
import torch
from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.services.vision.detector import get_detector
from backend.app.api.schemas.system import GPUInfo, HealthResponse, ModelInfo, StatusResponse
from backend.app.api.deps import get_uptime_seconds

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns basic application operational status and version.",
)
def health_check() -> HealthResponse:
    """Returns basic operational status."""
    return HealthResponse(
        status="healthy",
        api_status="operational",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Detailed System Status",
    description="Returns runtime telemetry, GPU allocation, detector status, and uptime.",
)
def system_status() -> StatusResponse:
    """Returns detailed hardware, model, and application runtime telemetry."""
    gpu_available = torch.cuda.is_available()
    gpu_info = None
    if gpu_available:
        gpu_info = GPUInfo(
            device_name=torch.cuda.get_device_name(0),
            memory_allocated_mb=round(torch.cuda.memory_allocated(0) / (1024**2), 2),
            memory_reserved_mb=round(torch.cuda.memory_reserved(0) / (1024**2), 2),
        )

    detector = get_detector()
    model_info = ModelInfo(
        name=settings.MODEL_NAME,
        classes_count=len(detector.classes_map),
        loaded=True,
    )

    return StatusResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        version="1.0.0",
        uptime_seconds=get_uptime_seconds(),
        python_version=sys.version.split()[0],
        effective_device=settings.get_effective_device(),
        gpu=gpu_info,
        model=model_info,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
