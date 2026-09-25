from datetime import datetime, timezone
import time
from typing import List
import pytest
import numpy as np

from backend.app.services.analytics.counter import (
    AnalyticsEngine,
    InvalidLineError,
    InvalidROIError,
    lines_intersect,
    point_in_polygon,
    determine_crossing_direction,
)
from backend.app.api.schemas.analytics import (
    AnalyticsSnapshot,
    LineDefinition,
    ROIDefinition,
)
from backend.app.services.vision.tracker import TrackedObject, TrackingResult


# ==========================================
# 1. Geometry & Vector Tests
# ==========================================

def test_lines_intersect_algorithm():
    # Perpendicular intersecting segments
    assert lines_intersect((0, 10), (20, 10), (10, 0), (10, 20)) is True
    # Non-intersecting parallel segments
    assert lines_intersect((0, 0), (10, 0), (0, 10), (10, 10)) is False
    # Non-intersecting collinear segments
    assert lines_intersect((0, 0), (5, 0), (10, 0), (15, 0)) is False


def test_determine_crossing_direction():
    # Horizontal line from (0, 100) to (200, 100)
    line_start = (0.0, 100.0)
    line_end = (200.0, 100.0)

    # Downward movement (top to bottom across line): y from 90 to 110
    dir1 = determine_crossing_direction((100.0, 90.0), (100.0, 110.0), line_start, line_end)
    assert dir1 in ["A_TO_B", "B_TO_A"]

    # Upward movement (bottom to top across line): y from 110 to 90
    dir2 = determine_crossing_direction((100.0, 110.0), (100.0, 90.0), line_start, line_end)
    assert dir2 != dir1


def test_point_in_polygon():
    # Square ROI from (100, 100) to (300, 300)
    polygon = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]

    # Inside
    assert point_in_polygon((200.0, 200.0), polygon) is True
    # Outside
    assert point_in_polygon((50.0, 50.0), polygon) is False
    assert point_in_polygon((350.0, 200.0), polygon) is False


# ==========================================
# 2. Analytics Engine Counts & Lifecycle
# ==========================================

def test_unique_and_active_counts_across_frames():
    engine = AnalyticsEngine(session_id="test_counts")
    engine.reset()

    # Frame 1: Person #1 and Car #2
    t1 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.9, x1=50, y1=50, x2=100, y2=150)
    t2 = TrackedObject(track_id=2, class_id=2, class_name="car", confidence=0.88, x1=200, y1=200, x2=350, y2=300)
    snap1 = engine.update([t1, t2])

    assert snap1.active_objects == 2
    assert snap1.total_unique_objects == 2
    assert snap1.class_distribution == {"person": 1, "car": 1}

    # Frame 2: Same Person #1 and Car #2, plus New Person #3
    t1_moved = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.9, x1=55, y1=52, x2=105, y2=152)
    t2_moved = TrackedObject(track_id=2, class_id=2, class_name="car", confidence=0.88, x1=205, y1=202, x2=355, y2=302)
    t3 = TrackedObject(track_id=3, class_id=0, class_name="person", confidence=0.85, x1=400, y1=100, x2=450, y2=200)
    snap2 = engine.update([t1_moved, t2_moved, t3])

    # Active count should be 3, Unique count should be 3 (not 5!)
    assert snap2.active_objects == 3
    assert snap2.total_unique_objects == 3
    assert snap2.total_detections == 5
    assert snap2.class_statistics["person"].unique_count == 2
    assert snap2.class_statistics["car"].unique_count == 1


def test_track_lifecycle_and_exit_event():
    engine = AnalyticsEngine(session_id="test_lifecycle", exit_threshold_frames=3)
    engine.reset()

    # Frame 1: Person #1 appears
    p1 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.9, x1=100, y1=100, x2=150, y2=200)
    snap1 = engine.update([p1])
    assert snap1.active_objects == 1
    entered_events = [e for e in engine.events if e.event_type == "OBJECT_ENTERED"]
    assert len(entered_events) == 1
    assert entered_events[0].track_id == 1

    # Frames 2, 3, 4, 5: Person #1 is missing (disappears)
    for _ in range(4):
        snap = engine.update([])
        assert snap.active_objects == 0

    # Person #1 has exceeded exit_threshold_frames (3), so status should be 'exited' and emit OBJECT_EXITED
    exited_events = [e for e in engine.events if e.event_type == "OBJECT_EXITED"]
    assert len(exited_events) == 1
    assert exited_events[0].track_id == 1
    assert engine.track_data[1]["status"] == "exited"


# ==========================================
# 3. Line Crossing Tests
# ==========================================

def test_line_crossing_and_anti_duplicate():
    engine = AnalyticsEngine(session_id="test_line")
    engine.reset()
    engine.clear_lines()

    # Register horizontal tripwire at y = 100
    line = LineDefinition(
        line_id="gate_1",
        name="Main Entrance Gate",
        start_point=(0.0, 100.0),
        end_point=(400.0, 100.0),
        direction="BOTH",
    )
    engine.add_line(line)

    # Frame 1: Object above line (bottom_center at y=80)
    f1 = TrackedObject(track_id=10, class_id=0, class_name="person", confidence=0.9, x1=50, y1=40, x2=70, y2=80)
    engine.update([f1])

    # Frame 2: Object crosses line (bottom_center moves to y=120)
    f2 = TrackedObject(track_id=10, class_id=0, class_name="person", confidence=0.9, x1=50, y1=80, x2=70, y2=120)
    snap2 = engine.update([f2])

    assert snap2.line_crossings["gate_1"] == 1
    crossing_events = [e for e in engine.events if e.event_type == "LINE_CROSSED"]
    assert len(crossing_events) == 1
    assert crossing_events[0].track_id == 10
    assert crossing_events[0].details["line_id"] == "gate_1"

    # Frame 3: Object hovers near line (bottom_center at y=118)
    f3 = TrackedObject(track_id=10, class_id=0, class_name="person", confidence=0.9, x1=50, y1=78, x2=70, y2=118)
    snap3 = engine.update([f3])

    # Anti-duplicate cooldown should prevent double count
    assert snap3.line_crossings["gate_1"] == 1
    assert len([e for e in engine.events if e.event_type == "LINE_CROSSED"]) == 1


# ==========================================
# 4. ROI Analytics & Dwell Time Tests
# ==========================================

def test_roi_enter_exit_and_dwell():
    engine = AnalyticsEngine(session_id="test_roi")
    engine.reset()
    engine.clear_rois()

    # Register ROI polygon: [100, 100] to [200, 200]
    roi = ROIDefinition(
        roi_id="security_zone",
        name="Restricted Zone",
        points=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
        anchor="bottom_center",
    )
    engine.add_roi(roi)

    # Step 1: Object outside ROI (bottom_center at x=50, y=50)
    obj_outside = TrackedObject(track_id=20, class_id=0, class_name="person", confidence=0.9, x1=40, y1=30, x2=60, y2=50)
    snap1 = engine.update([obj_outside])
    assert snap1.roi_counts.get("security_zone", 0) == 0

    # Step 2: Object enters ROI (bottom_center at x=150, y=150)
    obj_inside1 = TrackedObject(track_id=20, class_id=0, class_name="person", confidence=0.9, x1=140, y1=130, x2=160, y2=150)
    snap2 = engine.update([obj_inside1])
    assert snap2.roi_counts["security_zone"] == 1
    roi_enter_events = [e for e in engine.events if e.event_type == "ROI_ENTER"]
    assert len(roi_enter_events) == 1
    assert roi_enter_events[0].track_id == 20

    # Step 3: Object stays inside ROI
    time.sleep(0.01)
    obj_inside2 = TrackedObject(track_id=20, class_id=0, class_name="person", confidence=0.9, x1=145, y1=132, x2=165, y2=152)
    snap3 = engine.update([obj_inside2])
    assert snap3.roi_counts["security_zone"] == 1

    # Step 4: Object exits ROI (bottom_center at x=250, y=250)
    obj_exited = TrackedObject(track_id=20, class_id=0, class_name="person", confidence=0.9, x1=240, y1=230, x2=260, y2=250)
    snap4 = engine.update([obj_exited])
    assert snap4.roi_counts.get("security_zone", 0) == 0

    roi_exit_events = [e for e in engine.events if e.event_type == "ROI_EXIT"]
    assert len(roi_exit_events) == 1
    assert roi_exit_events[0].track_id == 20


# ==========================================
# 5. Deterministic Realistic Sequence Test (Step 16)
# ==========================================

def test_deterministic_walkthrough_sequence():
    """
    Executes the exact 5-frame scenario from Step 16:
    Frame 1: Person #1 outside ROI
    Frame 2: Person #1 approaches ROI
    Frame 3: Person #1 enters ROI
    Frame 4: Person #1 remains inside ROI
    Frame 5: Person #1 exits ROI
    """
    engine = AnalyticsEngine(session_id="deterministic_seq")
    engine.reset()
    engine.clear_rois()

    roi = ROIDefinition(
        roi_id="store_front",
        name="Store Entrance",
        points=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
        anchor="bottom_center",
    )
    engine.add_roi(roi)

    # Frame 1: Person #1 outside ROI (bottom_center: 50, 200)
    f1 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.92, x1=40, y1=150, x2=60, y2=200)
    s1 = engine.update([f1])
    assert s1.total_unique_objects == 1
    assert s1.roi_counts.get("store_front", 0) == 0

    # Frame 2: Person #1 approaches ROI (bottom_center: 80, 200)
    f2 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.91, x1=70, y1=150, x2=90, y2=200)
    s2 = engine.update([f2])
    assert s2.total_unique_objects == 1
    assert s2.roi_counts.get("store_front", 0) == 0

    # Frame 3: Person #1 enters ROI (bottom_center: 150, 200)
    f3 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.90, x1=140, y1=150, x2=160, y2=200)
    s3 = engine.update([f3])
    assert s3.total_unique_objects == 1
    assert s3.roi_counts["store_front"] == 1
    assert any(e.event_type == "ROI_ENTER" and e.track_id == 1 for e in engine.events)

    # Frame 4: Person #1 remains inside ROI (bottom_center: 220, 200)
    f4 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.89, x1=210, y1=150, x2=230, y2=200)
    s4 = engine.update([f4])
    assert s4.total_unique_objects == 1
    assert s4.roi_counts["store_front"] == 1

    # Frame 5: Person #1 exits ROI (bottom_center: 350, 200)
    f5 = TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.88, x1=340, y1=150, x2=360, y2=200)
    s5 = engine.update([f5])
    assert s5.total_unique_objects == 1
    assert s5.roi_counts.get("store_front", 0) == 0
    assert any(e.event_type == "ROI_EXIT" and e.track_id == 1 for e in engine.events)

    # Final Verification
    assert len(engine.trajectories[1]) == 5
    assert s5.class_statistics["person"].unique_count == 1
    assert s5.class_statistics["person"].roi_entry_count == 1
    assert s5.class_statistics["person"].roi_exit_count == 1


# ==========================================
# 6. Edge Cases & Validation Tests
# ==========================================

def test_invalid_roi_and_line_configurations():
    engine = AnalyticsEngine(session_id="test_invalids")

    # Invalid ROI (< 3 points)
    with pytest.raises(InvalidROIError):
        engine.add_roi(ROIDefinition(roi_id="bad_roi", name="Bad", points=[(10.0, 10.0), (20.0, 20.0)]))

    # Invalid Line (identical start & end points)
    with pytest.raises(InvalidLineError):
        engine.add_line(LineDefinition(line_id="bad_line", name="Bad", start_point=(50.0, 50.0), end_point=(50.0, 50.0)))


def test_integration_with_step5_tracking_result():
    """Verifies that AnalyticsEngine directly consumes Step 5 TrackingResult."""
    engine = AnalyticsEngine(session_id="step5_integration")
    engine.reset()

    obj1 = TrackedObject(track_id=5, class_id=0, class_name="person", confidence=0.87, x1=10, y1=20, x2=30, y2=40)
    step5_result = TrackingResult(
        frame_number=10,
        timestamp="2026-09-24T16:00:00Z",
        objects=[obj1],
        active_track_count=1,
        processing_time_ms=14.2,
        session_id="step5_integration",
        cumulative_unique_tracks=1,
    )

    snapshot = engine.update(step5_result)
    assert isinstance(snapshot, AnalyticsSnapshot)
    assert snapshot.frame_number == 10
    assert snapshot.active_objects == 1
    assert snapshot.total_unique_objects == 1
    assert snapshot.performance.last_processing_time_ms >= 14.2
