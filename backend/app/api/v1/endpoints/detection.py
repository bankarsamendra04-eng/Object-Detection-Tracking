import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.db.session import get_db
from backend.app.db.models import DetectionRecord
from backend.app.services.vision.detector import (
    DetectionEngine,
    get_detector,
    InvalidImageError,
    InferenceError,
)
from backend.app.services.vision.input_sources import ImageInput
from backend.app.api.schemas.detection import DetectionResponse

logger = get_logger("api.detection")
router = APIRouter()

MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit


@router.post(
    "/image",
    response_model=DetectionResponse,
    summary="Detect Objects in Image",
    description="Processes an uploaded image file (JPG, PNG, WEBP, BMP) using the local YOLO DetectionEngine.",
    responses={
        200: {"description": "Successful detection output with bounding boxes and classes."},
        400: {"description": "Invalid image file format or corrupt image payload."},
        413: {"description": "Uploaded image file exceeds the 15 MB size limit."},
        422: {"description": "Confidence or IoU threshold parameter out of valid [0.0, 1.0] range."},
    },
)
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze (JPG, PNG, WEBP, BMP)"),
    confidence: Optional[float] = Form(None, description="Detection confidence filter (0.0 to 1.0)"),
    iou: Optional[float] = Form(None, description="IoU threshold for NMS suppression (0.0 to 1.0)"),
    detector: DetectionEngine = Depends(get_detector),
    db: Session = Depends(get_db),
) -> DetectionResponse:
    """Detects objects in an uploaded image file."""
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

    # 2. File Format & Extension Validation
    filename = file.filename or "uploaded_image.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in ImageInput.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image extension '{ext}'. Supported formats: {sorted(ImageInput.SUPPORTED_EXTENSIONS)}",
        )

    if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME content-type '{file.content_type}'. Must be an image format.",
        )

    # 3. Read image bytes with size limit enforcement
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image stream: {str(e)}",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file size ({len(image_bytes)/(1024**2):.1f} MB) exceeds maximum allowed size of 15 MB.",
        )

    # 4. Decode Image
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image data: {str(e)}",
        )

    if pil_image.width > settings.MAX_IMAGE_DIMENSION or pil_image.height > settings.MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions ({pil_image.width}x{pil_image.height}) exceed maximum allowed limit of {settings.MAX_IMAGE_DIMENSION}px.",
        )

    # 5. Run Detection
    try:
        result = detector.detect(pil_image, conf=confidence, iou=iou)
    except InvalidImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except InferenceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    response = DetectionResponse.model_validate(result.to_dict())

    # 6. Persist summary in SQLite database
    try:
        record = DetectionRecord(
            frame_number=1,
            total_detections=result.total_detections,
            inference_time_ms=result.inference_time_ms,
            class_counts=result.class_counts,
            detections_detail=[d.to_dict() for d in result.detections],
        )
        db.add(record)
        db.commit()
    except Exception as db_err:
        logger.warning(f"Database persist warning: {db_err}")
        db.rollback()

    return response
