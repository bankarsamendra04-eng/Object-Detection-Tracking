import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

from backend.app.core.config import settings, PROJECT_ROOT
from backend.app.services.vision.detector import DetectionEngine, DetectionResult, ModelLoadError, get_detector
from backend.app.services.vision.model_manager import (
    ModelManager,
    ModelEntry,
    ModelManagerError,
    ModelNotFoundError,
    ModelUnavailableError,
    ModelCorruptError,
    get_model_manager,
)
from backend.app.services.vision.tracker import ByteTrackTracker
from backend.app.services.analytics.counter import AnalyticsEngine


@pytest.fixture
def model_manager():
    """Provides a fresh ModelManager instance."""
    return ModelManager()


@pytest.fixture
def synthetic_frame():
    """Provides a synthetic 640x640 BGR image for inference."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


class TestModelRegistryAndLoading:
    """Tests registry loading, metadata extraction, and weight resolution."""

    def test_pretrained_model_remains_loadable(self):
        """1. Pretrained model remains loadable via DetectionEngine and ModelManager."""
        engine = DetectionEngine(model_name_or_path="yolov8n.pt")
        assert engine.model is not None
        assert len(engine.classes_map) > 0
        assert "car" in engine.available_classes or "person" in engine.available_classes

    def test_custom_model_registry_parsing(self, model_manager):
        """2. Custom model registry reads models.yaml and parses metadata accurately."""
        models = model_manager.list_models()
        assert len(models) >= 2

        model_ids = [m["model_id"] for m in models]
        assert "general_pretrained" in model_ids
        assert "custom_visdrone" in model_ids

        visdrone = model_manager.get_model("custom_visdrone")
        assert visdrone is not None
        assert visdrone.model_type == "custom"
        assert visdrone.num_classes == 10
        assert visdrone.classes[0] == "pedestrian"
        assert visdrone.classes[3] == "car"
        assert visdrone.input_size == 1280  # small-object optimized resolution

    def test_unavailable_custom_model_handling(self, model_manager):
        """3. Unavailable custom model is detected and marked NOT_AVAILABLE."""
        assert model_manager.is_model_available("custom_visdrone") is False
        with pytest.raises(ModelUnavailableError) as exc_info:
            model_manager.load_model("custom_visdrone")
        assert "not available" in str(exc_info.value).lower()

    def test_invalid_model_path(self, model_manager):
        """4. Invalid model path resolution returns None or raises appropriately."""
        res = model_manager.resolve_weights_path("non_existent_weights_12345.pt")
        assert res is not None  # returns canonical relative path
        assert not res.is_file()

    def test_missing_model_file_validation(self, model_manager):
        """5. Missing model file validation returns NOT_AVAILABLE status."""
        val = model_manager.validate_model("custom_visdrone", run_test_inference=False)
        assert val["status"] == "NOT_AVAILABLE"
        assert val["available"] is False
        assert val["loadable"] is False
        assert "does not exist" in val["error"].lower()

    def test_unregistered_model_validation(self, model_manager):
        """6. Validation of an unknown model ID returns INVALID status."""
        val = model_manager.validate_model("completely_unknown_id")
        assert val["status"] == "INVALID"
        assert val["available"] is False
        assert "not registered" in val["error"].lower()

    def test_pretrained_model_validation(self, model_manager):
        """7. Successful validation of available pretrained model."""
        val = model_manager.validate_model("general_pretrained", run_test_inference=True)
        assert val["status"] == "VALID"
        assert val["available"] is True
        assert val["loadable"] is True
        assert val["num_classes"] >= 80
        assert val["inference_tested"] is True
        assert val["test_inference_time_ms"] is not None
        assert val["test_inference_time_ms"] > 0

    def test_successful_model_loading_and_caching(self, model_manager):
        """8 & 11. Model loading succeeds and caches instance without reloading."""
        model1, entry1 = model_manager.load_model("general_pretrained")
        assert model1 is not None
        assert entry1.model_id == "general_pretrained"

        # Second load must return identical cached instance
        model2, entry2 = model_manager.load_model("general_pretrained")
        assert model1 is model2


class TestDeviceAndHardwareRouting:
    """Tests hardware device selection, CUDA handling, and CPU fallback."""

    def test_device_selection_and_cpu_fallback(self):
        """9 & 10. Device selection honors configuration and falls back to CPU when needed."""
        cpu_engine = DetectionEngine(model_name_or_path="yolov8n.pt", device="cpu")
        assert cpu_engine.device == "cpu"

        # Check effective device helper
        effective = settings.get_effective_device()
        if torch.cuda.is_available():
            assert "cuda" in effective
        else:
            assert effective == "cpu"


class TestModelSwitchingAndInference:
    """Tests active model switching, inference execution, and downstream compatibility."""

    def test_model_switching_lifecycle(self, model_manager):
        """12. Switching active model updates detector attributes and active ID."""
        detector = DetectionEngine(model_name_or_path="yolov8n.pt")
        res = model_manager.switch_model("general_pretrained", detector=detector)
        assert res["status"] == "success"
        assert res["active_model_id"] == "general_pretrained"
        assert model_manager.active_model_id == "general_pretrained"

    def test_inference_after_model_loading(self, synthetic_frame):
        """13. Inference executes normally after model loading and produces DetectionResult."""
        engine = get_detector()
        result = engine.detect(synthetic_frame, conf=0.25, imgsz=320)
        assert isinstance(result, DetectionResult)
        assert result.image_width == 640
        assert result.image_height == 640
        assert result.inference_time_ms >= 0

    def test_detection_compatibility_with_step5_tracking(self, synthetic_frame):
        """14. DetectionResult integrates seamlessly with ByteTrackTracker."""
        engine = get_detector()
        result = engine.detect(synthetic_frame, imgsz=320)

        tracker = ByteTrackTracker(session_id="test_model_manager_tracking")
        track_result = tracker.update(result)
        assert isinstance(track_result.objects, list)
        assert track_result.active_track_count >= 0

    def test_detection_compatibility_with_step6_analytics(self, synthetic_frame):
        """15. Detection and Tracking integrate seamlessly with AnalyticsEngine."""
        engine = get_detector()
        result = engine.detect(synthetic_frame, imgsz=320)

        tracker = ByteTrackTracker(session_id="test_model_manager_analytics")
        track_result = tracker.update(result)

        analytics = AnalyticsEngine(session_id="test_model_manager_analytics")
        snapshot = analytics.update(track_result.objects)
        assert snapshot.total_unique_objects >= 0
        assert snapshot.active_objects >= 0


class TestSecurityAndEdgeCases:
    """Tests security constraints, corrupt model detection, and path handling."""

    def test_corrupt_model_file_handling(self, model_manager, tmp_path):
        """16. Corrupt or damaged weight file raises ModelLoadError or reports INVALID."""
        fake_corrupt_file = tmp_path / "corrupt_model.pt"
        fake_corrupt_file.write_bytes(b"NOT_A_VALID_PYTORCH_OR_YOLO_WEIGHT_FILE")

        # Test through DetectionEngine
        with pytest.raises(ModelLoadError):
            DetectionEngine(model_name_or_path=str(fake_corrupt_file))

    def test_unsupported_extension_rejection(self, tmp_path):
        """Unsupported model extensions (e.g. .exe, .sh) are rejected."""
        fake_bad_ext = tmp_path / "model.exe"
        fake_bad_ext.write_bytes(b"bad content")
        with pytest.raises(ModelLoadError) as exc_info:
            DetectionEngine(model_name_or_path=str(fake_bad_ext))
        assert "unsupported" in str(exc_info.value).lower()

    def test_security_against_path_traversal(self, model_manager):
        """18. Security constraint prevents directory traversal outside project boundaries."""
        traversal_path = "../../Windows/System32/calc.exe"
        resolved = model_manager.resolve_weights_path(traversal_path)
        assert resolved is None

    def test_small_object_resolution_configuration(self):
        """Small-object detection resolution is configurable and honored in DetectionEngine."""
        engine = DetectionEngine(model_name_or_path="yolov8n.pt", imgsz=1280)
        assert engine.default_imgsz == 1280
