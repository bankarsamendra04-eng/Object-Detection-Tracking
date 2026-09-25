from pathlib import Path
import pytest
import numpy as np
import cv2

from backend.app.core.config import PROJECT_ROOT
from backend.app.services.vision.detector import (
    Detection,
    DetectionResult,
    get_detector,
)
from backend.app.services.vision.tracker import (
    ByteTrackTracker,
    TrackedObject,
    TrackingResult,
    TrackState,
    SingleTrack,
    TrackerConfigError,
    TrackerUpdateError,
    annotate_tracking_frame,
)
from backend.app.services.vision.input_sources import VideoInput, WebcamInput, probe_camera_availability

SAMPLE_VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "sample_real_bus.mp4"
SAMPLE_SYNTH_PATH = PROJECT_ROOT / "data" / "samples" / "sample_synthetic_shapes.mp4"
BUS_IMAGE_PATH = PROJECT_ROOT / "data" / "samples" / "bus.jpg"


# ==========================================
# 1. Tracker Initialization & Configuration
# ==========================================

def test_tracker_initialization():
    tracker = ByteTrackTracker(
        track_high_thresh=0.6,
        track_low_thresh=0.2,
        track_match_thresh=0.7,
        persistence_buffer=25,
    )
    assert tracker.track_high_thresh == 0.6
    assert tracker.track_low_thresh == 0.2
    assert tracker.track_match_thresh == 0.7
    assert tracker.persistence_buffer == 25
    assert tracker.frame_count == 0


def test_tracker_invalid_config():
    # track_low_thresh must not exceed track_high_thresh
    with pytest.raises(TrackerConfigError):
        ByteTrackTracker(track_high_thresh=0.3, track_low_thresh=0.8)


def test_tracker_empty_and_invalid_detections():
    tracker = ByteTrackTracker()

    # Empty detections
    empty_det = DetectionResult(
        detections=[],
        image_width=640,
        image_height=480,
        inference_time_ms=5.0,
        device_used="cpu",
        model_name="yolov8n.pt",
        frame_number=1,
    )
    res = tracker.update(empty_det)
    assert isinstance(res, TrackingResult)
    assert res.active_track_count == 0
    assert len(res.objects) == 0

    # Invalid input (not DetectionResult)
    with pytest.raises(TrackerUpdateError):
        tracker.update("invalid_detection_input")


def test_tracker_reset():
    tracker = ByteTrackTracker()
    det = Detection(class_id=0, class_name="person", confidence=0.9, x1=50, y1=50, x2=100, y2=100)
    det_result = DetectionResult(
        detections=[det],
        image_width=640,
        image_height=480,
        inference_time_ms=5.0,
        device_used="cpu",
        model_name="yolov8n.pt",
        frame_number=1,
    )
    tracker.update(det_result)
    assert tracker.frame_count == 1
    assert len(tracker.tracked_tracks) == 1

    tracker.reset()
    assert tracker.frame_count == 0
    assert len(tracker.tracked_tracks) == 0
    assert len(tracker.cumulative_unique_ids) == 0


# ==========================================
# 2. Persistent ID Association Tests
# ==========================================

def test_persistent_id_same_object_consecutive_frames():
    """Verifies that an object moving slightly across frames maintains its exact track_id."""
    tracker = ByteTrackTracker()
    tracker.reset()

    # Frame 1: Object at [100, 100, 160, 160]
    det1 = Detection(class_id=0, class_name="person", confidence=0.9, x1=100, y1=100, x2=160, y2=160, frame_number=1)
    res1 = tracker.update(
        DetectionResult([det1], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=1)
    )
    assert len(res1.objects) == 1
    orig_id = res1.objects[0].track_id

    # Frame 2: Object moves slightly to [104, 102, 164, 162]
    det2 = Detection(class_id=0, class_name="person", confidence=0.88, x1=104, y1=102, x2=164, y2=162, frame_number=2)
    res2 = tracker.update(
        DetectionResult([det2], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=2)
    )
    assert len(res2.objects) == 1
    assert res2.objects[0].track_id == orig_id, f"Track ID must persist: expected {orig_id}, got {res2.objects[0].track_id}"

    # Frame 3: Object continues to [108, 105, 168, 165]
    det3 = Detection(class_id=0, class_name="person", confidence=0.85, x1=108, y1=105, x2=168, y2=165, frame_number=3)
    res3 = tracker.update(
        DetectionResult([det3], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=3)
    )
    assert len(res3.objects) == 1
    assert res3.objects[0].track_id == orig_id
    assert len(res3.objects[0].trajectory) == 3


def test_persistent_id_multiple_objects_and_lifecycle():
    """Verifies that multiple independent objects maintain distinct, persistent IDs."""
    tracker = ByteTrackTracker()
    tracker.reset()

    # Frame 1: Person and Car
    p1 = Detection(class_id=0, class_name="person", confidence=0.92, x1=50, y1=50, x2=100, y2=150, frame_number=1)
    c1 = Detection(class_id=2, class_name="car", confidence=0.90, x1=300, y1=200, x2=450, y2=300, frame_number=1)
    res1 = tracker.update(DetectionResult([p1, c1], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=1))

    assert res1.active_track_count == 2
    person_obj = res1.filter_by_class("person")[0]
    car_obj = res1.filter_by_class("car")[0]
    p_id, c_id = person_obj.track_id, car_obj.track_id
    assert p_id != c_id

    # Frame 2: Both objects move
    p2 = Detection(class_id=0, class_name="person", confidence=0.91, x1=55, y1=52, x2=105, y2=152, frame_number=2)
    c2 = Detection(class_id=2, class_name="car", confidence=0.89, x1=310, y1=205, x2=460, y2=305, frame_number=2)
    res2 = tracker.update(DetectionResult([p2, c2], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=2))

    assert res2.active_track_count == 2
    assert res2.get_object_by_id(p_id) is not None
    assert res2.get_object_by_id(c_id) is not None
    assert res2.get_object_by_id(p_id).class_name == "person"
    assert res2.get_object_by_id(c_id).class_name == "car"


def test_occlusion_and_low_confidence_recovery():
    """Tests ByteTrack's 2nd stage: recovering a track with lower confidence (occlusion/blur)."""
    tracker = ByteTrackTracker(track_high_thresh=0.6, track_low_thresh=0.2)
    tracker.reset()

    # Frame 1: High confidence detection
    d1 = Detection(class_id=0, class_name="person", confidence=0.85, x1=100, y1=100, x2=150, y2=200, frame_number=1)
    res1 = tracker.update(DetectionResult([d1], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=1))
    orig_id = res1.objects[0].track_id

    # Frame 2: Low confidence detection (e.g. partial occlusion, conf=0.35)
    d2 = Detection(class_id=0, class_name="person", confidence=0.35, x1=102, y1=101, x2=152, y2=201, frame_number=2)
    res2 = tracker.update(DetectionResult([d2], 640, 480, 5.0, "cpu", "yolov8n.pt", frame_number=2))

    assert len(res2.objects) == 1
    # Successfully associated via low-confidence 2nd stage!
    assert res2.objects[0].track_id == orig_id


# ==========================================
# 3. Pipeline Integration Tests
# ==========================================

def test_pipeline_video_input_to_detection_to_tracker():
    """Tests the full decoupled pipeline: VideoInput -> DetectionEngine -> ByteTrackTracker."""
    assert SAMPLE_VIDEO_PATH.is_file(), f"Sample video {SAMPLE_VIDEO_PATH} missing"

    detector = get_detector()
    tracker = ByteTrackTracker()
    tracker.reset()

    tracking_results: List[TrackingResult] = []

    with VideoInput(SAMPLE_VIDEO_PATH, stride=2, max_frames=6) as vid_in:
        for frame_num, frame, timestamp_ms in vid_in:
            # 1. Detection
            det_result = detector.detect(frame, conf=0.25, frame_number=frame_num)
            assert isinstance(det_result, DetectionResult)

            # 2. Tracking
            track_result = tracker.update(det_result)
            assert isinstance(track_result, TrackingResult)
            assert track_result.frame_number == frame_num

            # 3. Annotation
            annotated = annotate_tracking_frame(frame, track_result)
            assert annotated.shape == frame.shape

            tracking_results.append(track_result)

    assert len(tracking_results) > 0
    # Over consecutive frames, tracks should be active
    assert any(r.active_track_count > 0 for r in tracking_results)


def test_pipeline_single_image_behavior():
    """Verifies that a single image produces valid 1-frame track IDs, but documents non-persistence."""
    assert BUS_IMAGE_PATH.is_file()
    img_bgr = cv2.imread(str(BUS_IMAGE_PATH))

    detector = get_detector()
    tracker = ByteTrackTracker()
    tracker.reset()

    det_result = detector.detect(img_bgr, conf=0.35)
    track_result = tracker.update(det_result)

    assert track_result.active_track_count > 0
    for obj in track_result.objects:
        assert obj.track_id > 0
        assert obj.tracking_state == TrackState.ACTIVE.value

    # Annotation check
    annotated = annotate_tracking_frame(img_bgr, track_result)
    assert annotated.shape == img_bgr.shape


def test_pipeline_webcam_bounded_tracking_if_available():
    """Tests bounded webcam tracking if physical hardware is available."""
    probe = probe_camera_availability(camera_index=0)
    if not probe.get("available") or not probe.get("readable"):
        pytest.skip("Physical webcam is not available; skipping live webcam tracking test.")

    detector = get_detector()
    tracker = ByteTrackTracker()
    tracker.reset()

    with WebcamInput(camera_index=0) as cam:
        for _ in range(3):
            success, frame, meta = cam.read()
            if not success or frame is None:
                break
            det_result = detector.detect(frame, conf=0.35, frame_number=meta["frame_number"])
            track_result = tracker.update(det_result)
            assert isinstance(track_result, TrackingResult)
            assert track_result.frame_number == meta["frame_number"]


# ==========================================
# 4. Visualization Annotation Tests
# ==========================================

def test_annotate_tracking_frame():
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    obj = TrackedObject(
        track_id=7,
        class_id=0,
        class_name="person",
        confidence=0.912,
        x1=100.0,
        y1=100.0,
        x2=200.0,
        y2=300.0,
        trajectory=[(150.0, 180.0), (150.0, 200.0)],
    )
    result = TrackingResult(
        frame_number=1,
        timestamp="2026-09-24T12:00:00Z",
        objects=[obj],
        active_track_count=1,
        processing_time_ms=1.5,
    )

    annotated = annotate_tracking_frame(canvas, result, draw_trajectories=True)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == canvas.shape
    # Canvas should not be entirely black anymore (drawn bounding boxes and text)
    assert np.any(annotated > 0)
