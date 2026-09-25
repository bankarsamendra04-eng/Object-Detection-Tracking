from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image

from backend.app.core.config import settings, PROJECT_ROOT
from backend.app.services.vision.detector import (
    Detection,
    DetectionResult,
    DetectionEngine,
    ModelLoadError,
    InvalidImageError,
    get_detector,
)

SAMPLE_BUS_PATH = PROJECT_ROOT / "data" / "samples" / "bus.jpg"


# ==========================================
# 1. Data Structures Unit Tests
# ==========================================

def test_detection_dataclass():
    det = Detection(
        class_id=0,
        class_name="person",
        confidence=0.8875,
        x1=100.0,
        y1=150.0,
        x2=200.0,
        y2=350.0,
        frame_number=1,
    )

    assert det.class_id == 0
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.8875, rel=1e-3)
    assert det.width == 100.0
    assert det.height == 200.0
    assert det.center == (150.0, 250.0)
    assert det.area == 20000.0
    assert det.box_xyxy == (100.0, 150.0, 200.0, 350.0)

    data = det.to_dict()
    assert data["class_id"] == 0
    assert data["class_name"] == "person"
    assert data["confidence"] == 0.8875
    assert data["box"]["x1"] == 100.0
    assert data["box"]["y1"] == 150.0
    assert data["box"]["x2"] == 200.0
    assert data["box"]["y2"] == 350.0


def test_detection_result_methods():
    det1 = Detection(class_id=0, class_name="person", confidence=0.9, x1=10, y1=10, x2=50, y2=100)
    det2 = Detection(class_id=0, class_name="person", confidence=0.4, x1=60, y1=10, x2=90, y2=100)
    det3 = Detection(class_id=2, class_name="car", confidence=0.85, x1=100, y1=50, x2=200, y2=150)

    result = DetectionResult(
        detections=[det1, det2, det3],
        image_width=640,
        image_height=480,
        inference_time_ms=12.5,
        device_used="cuda:0",
        model_name="yolov8n.pt",
    )

    assert result.total_detections == 3
    assert result.class_counts == {"person": 2, "car": 1}

    # Filter by class
    person_only = result.filter_by_class("person")
    assert len(person_only) == 2
    car_only = result.filter_by_class(["car"])
    assert len(car_only) == 1

    # Filter by confidence
    high_conf = result.filter_by_confidence(0.8)
    assert len(high_conf) == 2
    assert all(d.confidence >= 0.8 for d in high_conf)

    # Dict representation
    res_dict = result.to_dict()
    assert res_dict["total_detections"] == 3
    assert res_dict["class_counts"]["person"] == 2
    assert len(res_dict["detections"]) == 3


# ==========================================
# 2. Engine Error Handling Tests
# ==========================================

def test_missing_model_raises_model_load_error():
    with pytest.raises(ModelLoadError):
        DetectionEngine(model_name_or_path="nonexistent_invalid_path_weights.pt")


def test_invalid_image_inputs_raise_invalid_image_error():
    engine = get_detector()

    # None input
    with pytest.raises(InvalidImageError, match="cannot be None"):
        engine.detect(None)

    # Empty numpy array
    with pytest.raises(InvalidImageError, match="empty or has invalid shape"):
        engine.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    # Invalid dimension array
    with pytest.raises(InvalidImageError, match="empty or has invalid shape"):
        engine.detect(np.array([1, 2, 3]))

    # Non-existent file path
    with pytest.raises(InvalidImageError, match="does not exist"):
        engine.detect("this_file_does_not_exist_xyz.jpg")

    # Unsupported data type
    with pytest.raises(InvalidImageError, match="Unsupported image type"):
        engine.detect(12345)


# ==========================================
# 3. Model Persistence & No-Reload Check
# ==========================================

def test_model_persistence_across_inferences(dummy_frame):
    engine = get_detector()
    model_obj_id = id(engine.model)

    # Run inference twice
    res1 = engine.detect(dummy_frame)
    res2 = engine.detect(dummy_frame)

    assert res1 is not None
    assert res2 is not None
    # Model instance must remain identical in memory
    assert id(engine.model) == model_obj_id


# ==========================================
# 4. CPU & Device Fallback Verification
# ==========================================

def test_cpu_inference(dummy_frame):
    cpu_engine = DetectionEngine(device="cpu")
    assert cpu_engine.device == "cpu"

    result = cpu_engine.detect(dummy_frame)
    assert result.device_used == "cpu"
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.inference_time_ms > 0.0


# ==========================================
# 5. Real Image Inference Verification
# ==========================================

def test_real_image_inference():
    assert SAMPLE_BUS_PATH.is_file(), f"Sample bus image missing at {SAMPLE_BUS_PATH}"
    
    engine = get_detector()
    
    # Run inference using file path input
    result = engine.detect(str(SAMPLE_BUS_PATH), conf=0.25)

    assert result.image_width > 0
    assert result.image_height > 0
    assert result.total_detections > 0
    assert "bus" in result.class_counts or "person" in result.class_counts

    # Verify each detection structure
    for det in result.detections:
        assert isinstance(det.class_name, str)
        assert 0.0 <= det.confidence <= 1.0
        assert 0.0 <= det.x1 < det.x2 <= result.image_width + 5.0
        assert 0.0 <= det.y1 < det.y2 <= result.image_height + 5.0
        assert det.area > 0


def test_confidence_threshold_behavior():
    assert SAMPLE_BUS_PATH.is_file()
    engine = get_detector()

    # Low threshold should yield more or equal detections than high threshold
    low_conf_result = engine.detect(str(SAMPLE_BUS_PATH), conf=0.15)
    high_conf_result = engine.detect(str(SAMPLE_BUS_PATH), conf=0.85)

    assert low_conf_result.total_detections >= high_conf_result.total_detections
    assert all(d.confidence >= 0.85 for d in high_conf_result.detections)


def test_image_annotation_utility():
    assert SAMPLE_BUS_PATH.is_file()
    img_bgr = cv2.imread(str(SAMPLE_BUS_PATH))
    assert img_bgr is not None

    engine = get_detector()
    result = engine.detect(img_bgr, conf=0.35)
    annotated = engine.annotate(img_bgr, result)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img_bgr.shape
