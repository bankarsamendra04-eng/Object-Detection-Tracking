from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Path, status

from backend.app.api.deps import get_model_manager, get_detector
from backend.app.api.schemas.model import (
    ModelInfoResponse,
    ModelListResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
    ModelValidationResponse,
)
from backend.app.core.logging import get_logger
from backend.app.services.vision.detector import DetectionEngine
from backend.app.services.vision.model_manager import (
    ModelCorruptError,
    ModelManager,
    ModelNotFoundError,
    ModelUnavailableError,
)

logger = get_logger("api.models")
router = APIRouter()


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List Registered Models",
    description="Returns all registered models from the model registry along with their real-time physical availability.",
)
def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelListResponse:
    """Lists all models and active detector state."""
    models_data = model_manager.list_models(refresh_status=True)
    models_list = [ModelInfoResponse(**m) for m in models_data]
    return ModelListResponse(
        active_model_id=model_manager.active_model_id,
        total_models=len(models_list),
        models=models_list,
    )


@router.get(
    "/active",
    response_model=ModelInfoResponse,
    summary="Get Active Model Information",
    description="Returns metadata and class details for the currently active detection model.",
)
def get_active_model(
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """Retrieves metadata of the currently active model."""
    active_id = model_manager.active_model_id
    entry = model_manager.get_model(active_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active model '{active_id}' is not found in the model registry.",
        )
    return ModelInfoResponse(**entry.to_dict(is_active=True))


@router.get(
    "/{model_id}",
    response_model=ModelInfoResponse,
    summary="Get Model Details",
    description="Returns detailed metadata and configuration for a specified registered model ID.",
)
def get_model_by_id(
    model_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$", description="Registered model identifier"),
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """Retrieves metadata of a specific model ID."""
    entry = model_manager.get_model(model_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' is not registered in the model registry.",
        )
    is_active = (entry.model_id == model_manager.active_model_id)
    return ModelInfoResponse(**entry.to_dict(is_active=is_active))


@router.post(
    "/{model_id}/validate",
    response_model=ModelValidationResponse,
    summary="Validate Model",
    description="Verifies weights file existence, weight integrity, class metadata, and execution compatibility.",
)
def validate_model(
    model_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$", description="Registered model identifier to validate"),
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelValidationResponse:
    """Executes structured model validation."""
    result = model_manager.validate_model(model_id, run_test_inference=True)
    return ModelValidationResponse(**result)


@router.post(
    "/{model_id}/switch",
    response_model=ModelSwitchResponse,
    summary="Switch Active Detection Model",
    description="Switches the active model for detection and tracking. Restricted strictly to registered models.",
)
def switch_active_model(
    model_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$", description="Target registered model identifier to activate"),
    model_manager: ModelManager = Depends(get_model_manager),
    detector: DetectionEngine = Depends(get_detector),
) -> ModelSwitchResponse:
    """Safely switches the active detection model."""
    try:
        switch_result = model_manager.switch_model(model_id=model_id, detector=detector)
        return ModelSwitchResponse(**switch_result)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ModelUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ModelCorruptError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error switching to model '{model_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {str(e)}",
        )
