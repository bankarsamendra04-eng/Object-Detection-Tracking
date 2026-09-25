from contextlib import asynccontextmanager
from datetime import datetime, timezone
import sys
import traceback
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.db.session import init_db
from backend.app.services.vision.detector import DetectionError, get_detector
from backend.app.services.vision.input_sources import MediaSourceError
from backend.app.services.analytics.counter import AnalyticsError
from backend.app.services.vision.model_manager import (
    ModelNotFoundError,
    ModelUnavailableError,
    ModelCorruptError,
)
from backend.app.services.streaming.manager import get_stream_manager
from backend.app.api.v1.router import api_router
from backend.app.api.schemas.system import HealthResponse, StatusResponse, GPUInfo, ModelInfo
from backend.app.api.deps import get_uptime_seconds
from backend.app.api.v1.endpoints.websocket import websocket_stream_endpoint
import torch

# Setup centralized logging
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: database initialization and model warm-up."""
    logger.info("Initializing Object Detection & Tracking application services...")
    
    # 1. Initialize DB tables
    init_db()
    logger.info("Database schemas initialized.")

    # 2. Warm up model
    try:
        detector = get_detector()
        logger.info(f"YOLO detector pre-warmed on device: {detector.device}")
    except Exception as e:
        logger.warning(f"Model pre-warm warning (lazy load will occur on first request): {e}")

    yield

    logger.info("Shutting down application services...")
    try:
        stream_mgr = get_stream_manager()
        await stream_mgr.close_all()
        logger.info("All streaming sessions cleanly terminated.")
    except Exception as e:
        logger.warning(f"Error during stream manager shutdown: {e}")


def create_application() -> FastAPI:
    """Factory function for FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description=(
            "Production-Grade Computer Vision & Multi-Object Tracking REST & WebSocket API.\n\n"
            "Features:\n"
            "- Multi-Input Ingestion: Image, Video, Webcam\n"
            "- High-Throughput Edge AI: YOLOv8 Object Detection with CUDA acceleration\n"
            "- Persistent Tracking: ByteTrack multi-object tracker\n"
            "- Spatial Analytics: Line crossing tripwires, polygon ROIs, dwell times, and class telemetry\n"
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Security Headers Middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # CORS Middleware with configurable origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==========================================
    # Centralized Exception Handlers
    # ==========================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail if isinstance(exc.detail, str) else "HTTP Error",
                "detail": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(MediaSourceError)
    async def media_exception_handler(request: Request, exc: MediaSourceError):
        logger.warning(f"Media error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Media Processing Error",
                "detail": str(exc),
                "status_code": 400,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(DetectionError)
    async def detection_exception_handler(request: Request, exc: DetectionError):
        logger.warning(f"Detection error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Detection Error",
                "detail": str(exc),
                "status_code": 400,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        logger.warning(f"Model not found on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Model Not Found",
                "detail": str(exc),
                "status_code": 404,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
        logger.warning(f"Model unavailable on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Model Unavailable",
                "detail": str(exc),
                "status_code": 400,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(ModelCorruptError)
    async def model_corrupt_handler(request: Request, exc: ModelCorruptError):
        logger.warning(f"Model corrupt on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Model Corrupt",
                "detail": str(exc),
                "status_code": 422,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(AnalyticsError)
    async def analytics_exception_handler(request: Request, exc: AnalyticsError):
        logger.warning(f"Analytics error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Analytics Error",
                "detail": str(exc),
                "status_code": 400,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred while processing the request.",
                "status_code": 500,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Mount API Routers
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Root Level Health, Status, and Welcome
    @app.get("/", tags=["System & Health"], summary="API Root Overview")
    def root():
        return {
            "app": settings.APP_NAME,
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs",
            "redoc": "/redoc",
            "api_v1": settings.API_V1_PREFIX,
        }

    @app.get("/health", response_model=HealthResponse, tags=["System & Health"], summary="Root Health Check")
    def root_health():
        return HealthResponse()

    @app.get("/status", response_model=StatusResponse, tags=["System & Health"], summary="Root System Status")
    def root_status():
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
            name=detector.model_name,
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

    # Root Level WebSocket Endpoint alias
    app.add_api_websocket_route("/ws/stream", websocket_stream_endpoint, name="root_ws_stream")

    return app


app = create_application()
