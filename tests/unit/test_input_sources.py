from pathlib import Path
import pytest
import numpy as np
import cv2

from backend.app.core.config import settings, PROJECT_ROOT
from backend.app.services.vision.detector import get_detector, DetectionResult
from backend.app.services.vision.input_sources import (
    ImageInput,
    VideoInput,
    WebcamInput,
    FileNotFoundMediaError,
    UnsupportedFormatError,
    CorruptMediaError,
    DeviceUnavailableError,
    probe_camera_availability,
)

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
BUS_JPG = SAMPLES_DIR / "bus.jpg"
BUS_PNG = SAMPLES_DIR / "bus.png"
BUS_WEBP = SAMPLES_DIR / "bus.webp"
BUS_BMP = SAMPLES_DIR / "bus.bmp"
REAL_VIDEO = SAMPLES_DIR / "sample_real_bus.mp4"
SYNTH_VIDEO = SAMPLES_DIR / "sample_synthetic_shapes.mp4"


# ==========================================
# 1. ImageInput Tests
# ==========================================

@pytest.mark.parametrize("image_path", [BUS_JPG, BUS_PNG, BUS_WEBP, BUS_BMP])
def test_image_input_multi_format_loading(image_path: Path):
    assert image_path.is_file(), f"Sample image {image_path} must exist"
    
    with ImageInput(image_path) as img_in:
        assert img_in.metadata["source_type"] == "image"
        assert img_in.metadata["width"] > 0
        assert img_in.metadata["height"] > 0
        assert img_in.metadata["channels"] == 3

        success, frame, meta = img_in.read()
        assert success is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape[:2] == (meta["height"], meta["width"])

        # Subsequent read returns False (single-frame media consumed)
        success2, frame2, _ = img_in.read()
        assert success2 is False
        assert frame2 is None


def test_image_input_numpy_array():
    arr = np.zeros((120, 160, 3), dtype=np.uint8)
    with ImageInput(arr) as img_in:
        assert img_in.metadata["width"] == 160
        assert img_in.metadata["height"] == 120
        success, frame, meta = img_in.read()
        assert success is True
        assert frame.shape == (120, 160, 3)


def test_image_input_error_handling(tmp_path: Path):
    # 1. Missing file
    with pytest.raises(FileNotFoundMediaError):
        ImageInput(tmp_path / "non_existent_file.jpg")

    # 2. Unsupported extension
    invalid_ext = tmp_path / "test.txt"
    invalid_ext.write_text("dummy text")
    with pytest.raises(UnsupportedFormatError):
        ImageInput(invalid_ext)

    # 3. 0-byte corrupt file
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.touch()
    with pytest.raises(CorruptMediaError):
        ImageInput(corrupt_file)


def test_image_input_detection_integration():
    detector = get_detector()
    with ImageInput(BUS_JPG) as img_in:
        result, annotated = img_in.process(detector=detector, annotate=True, conf=0.3)
        assert isinstance(result, DetectionResult)
        assert result.total_detections > 0
        assert "bus" in result.class_counts or "person" in result.class_counts
        assert annotated is not None
        assert isinstance(annotated, np.ndarray)


# ==========================================
# 2. VideoInput Tests
# ==========================================

def test_video_input_metadata_and_iteration():
    assert REAL_VIDEO.is_file(), "sample_real_bus.mp4 must exist"

    with VideoInput(REAL_VIDEO, stride=2, max_frames=10) as vid_in:
        assert vid_in.metadata["source_type"] == "video"
        assert vid_in.metadata["width"] > 0
        assert vid_in.metadata["height"] > 0
        assert vid_in.metadata["fps"] > 0
        assert vid_in.metadata["total_frames"] > 0

        frames_read = 0
        for frame_num, frame, timestamp_ms in vid_in:
            frames_read += 1
            assert isinstance(frame, np.ndarray)
            assert frame_num > 0
            assert timestamp_ms >= 0.0

        assert frames_read <= 10
        assert frames_read > 0


def test_video_input_error_handling(tmp_path: Path):
    # 1. Non-existent video file
    with pytest.raises(FileNotFoundMediaError):
        VideoInput(tmp_path / "missing_video.mp4")

    # 2. Unsupported video extension
    bad_ext = tmp_path / "video.wav"
    bad_ext.write_bytes(b"dummy")
    with pytest.raises(UnsupportedFormatError):
        VideoInput(bad_ext)

    # 3. Corrupt / 0-byte video file
    corrupt_vid = tmp_path / "corrupt.mp4"
    corrupt_vid.touch()
    with pytest.raises(CorruptMediaError):
        VideoInput(corrupt_vid)


def test_video_input_detection_processing():
    detector = get_detector()
    with VideoInput(REAL_VIDEO, stride=3, max_frames=5) as vid_in:
        results = vid_in.process_all(detector=detector, conf=0.3, max_frames=5)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, DetectionResult)
            assert r.image_width == vid_in.metadata["width"]
            assert r.image_height == vid_in.metadata["height"]


# ==========================================
# 3. WebcamInput Tests
# ==========================================

def test_webcam_unavailable_handling():
    # Camera index 999 should not exist and raise DeviceUnavailableError
    with pytest.raises(DeviceUnavailableError):
        WebcamInput(camera_index=999)


def test_probe_camera_availability():
    # Probe nonexistent camera index
    res_fake = probe_camera_availability(camera_index=999)
    assert res_fake["available"] is False
    assert res_fake["camera_index"] == 999

    # Probe default camera index (0)
    res_default = probe_camera_availability(camera_index=0)
    assert "available" in res_default
    assert "readable" in res_default


def test_real_webcam_bounded_capture_if_available():
    probe = probe_camera_availability(camera_index=0)
    if not probe.get("available") or not probe.get("readable"):
        pytest.skip("Physical webcam is not available on this device; skipping live capture.")

    detector = get_detector()
    with WebcamInput(camera_index=0) as cam:
        assert cam.metadata["is_opened"] is True
        frames = cam.capture_frames(count=3, detector=detector, conf=0.35)
        assert len(frames) == 3
        for frame, res in frames:
            assert isinstance(frame, np.ndarray)
            assert isinstance(res, DetectionResult)
            assert res.image_width == cam.metadata["width"]
