from fastapi import APIRouter
from backend.app.api.v1.endpoints import (
    health,
    detection,
    tracking,
    analytics,
    config,
    websocket,
    models,
)

api_router = APIRouter()

# Health & Status
api_router.include_router(health.router, tags=["System & Health"])

# Detection Endpoints (both /detect and /detection aliases supported)
api_router.include_router(detection.router, prefix="/detect", tags=["Object Detection"])
api_router.include_router(detection.router, prefix="/detection", tags=["Object Detection"], include_in_schema=False)

# Tracking Endpoints (both /track and /tracking aliases supported)
api_router.include_router(tracking.router, prefix="/track", tags=["Object Tracking"])
api_router.include_router(tracking.router, prefix="/tracking", tags=["Object Tracking"], include_in_schema=False)

# Analytics Endpoints
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics & Reporting"])

# Configuration & Sources Endpoints
api_router.include_router(config.router, tags=["Configuration & Sources"])

# Model Management Endpoints
api_router.include_router(models.router, prefix="/models", tags=["Model Management"])

# WebSocket Endpoints
api_router.include_router(websocket.router, tags=["Real-time Stream"])
