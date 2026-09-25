from collections import defaultdict, deque
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.api.schemas.analytics import (
    AnalyticsEvent,
    AnalyticsReport,
    AnalyticsSnapshot,
    ClassStatistics,
    LineCrossingEvent,
    LineDefinition,
    PerformanceMetrics,
    ROIDefinition,
    TrackStatistics,
    TrajectoryPoint,
)
from backend.app.services.vision.tracker import TrackedObject, TrackingResult

logger = get_logger("analytics.engine")


# ==========================================
# Custom Exception Hierarchy
# ==========================================

class AnalyticsError(Exception):
    """Base exception for all analytics engine failures."""
    pass


class InvalidROIError(AnalyticsError):
    """Raised when an ROI definition has invalid vertices or dimensions."""
    pass


class InvalidLineError(AnalyticsError):
    """Raised when a virtual tripwire definition has invalid endpoints."""
    pass


# ==========================================
# Geometric & Vector Utilities
# ==========================================

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Returns True if points A, B, C are in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def lines_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """Determines whether segment p1-p2 intersects segment q1-q2."""
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def determine_crossing_direction(
    traj_start: Tuple[float, float],
    traj_end: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> str:
    """
    Computes cross product of line vector and movement vector to determine direction:
    Returns 'A_TO_B' or 'B_TO_A'.
    """
    line_dx = line_end[0] - line_start[0]
    line_dy = line_end[1] - line_start[1]
    move_dx = traj_end[0] - traj_start[0]
    move_dy = traj_end[1] - traj_start[1]

    cross = line_dx * move_dy - line_dy * move_dx
    return "A_TO_B" if cross > 0 else "B_TO_A"


def point_in_polygon(point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:
    """Determines whether a 2D point lies inside a polygon using OpenCV pointPolygonTest."""
    if len(polygon_points) < 3:
        return False
    pts = np.array(polygon_points, dtype=np.float32)
    # pointPolygonTest returns +1 for inside, 0 for on edge, -1 for outside
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def calculate_anchor_point(box_xyxy: Tuple[float, float, float, float], anchor_type: str = "bottom_center") -> Tuple[float, float]:
    """Calculates spatial anchor point for an object's bounding box."""
    x1, y1, x2, y2 = box_xyxy
    if anchor_type == "bottom_center":
        return ((x1 + x2) / 2.0, y2)
    elif anchor_type == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    elif anchor_type == "top_left":
        return (x1, y1)
    return ((x1 + x2) / 2.0, y2)


# ==========================================
# Core Analytics Engine
# ==========================================

class AnalyticsEngine:
    """
    Comprehensive, reusable Analytics Engine.
    Consumes TrackedObject streams from Step 5, tracks lifecycle durations,
    manages multi-line tripwires, ROI dwell time, class statistics, and performance telemetry.
    """

    def __init__(
        self,
        session_id: str,
        line_coords: Optional[List[List[int]]] = None,
        max_trajectory_points: int = 50,
        exit_threshold_frames: int = 5,
        max_event_history: int = 100,
    ):
        self.session_id = session_id
        self.max_trajectory_points = max_trajectory_points
        self.exit_threshold_frames = exit_threshold_frames
        self.max_event_history = max_event_history

        # Configurable Line Tripwires & ROIs
        self.lines: Dict[str, LineDefinition] = {}
        self.rois: Dict[str, ROIDefinition] = {}

        # Default Line from Settings
        default_coords = line_coords or settings.LINE_CROSSING_COORDS
        if default_coords and len(default_coords) == 2:
            self.add_line(
                LineDefinition(
                    line_id="default_line",
                    name="Default Boundary",
                    start_point=(float(default_coords[0][0]), float(default_coords[0][1])),
                    end_point=(float(default_coords[1][0]), float(default_coords[1][1])),
                    direction="BOTH",
                )
            )

        self.reset()

    def reset(self) -> None:
        """Resets all metrics, tracks, events, and performance counters."""
        self.total_frames = 0
        self.total_detections_count = 0
        self.unique_track_ids: Set[int] = set()
        self.active_track_ids: Set[int] = set()

        # Track lifecycle storage: track_id -> metadata
        self.track_data: Dict[int, Dict[str, Any]] = {}
        # Trajectories: track_id -> deque of TrajectoryPoint
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trajectory_points))

        # Event log
        self.events: List[AnalyticsEvent] = []
        # Anti-duplicate line crossing cooldown: (track_id, line_id) -> last_frame_crossed
        self.line_crossing_cooldown: Dict[Tuple[int, str], int] = {}

        # ROI state: track_id -> set of active roi_ids
        self.track_active_rois: Dict[int, Set[str]] = defaultdict(set)
        # ROI dwell accumulator: (track_id, roi_id) -> total_seconds
        self.roi_dwell_accumulators: Dict[Tuple[int, str], float] = defaultdict(float)
        # ROI entry timestamps: (track_id, roi_id) -> entry_iso_time
        self.roi_entry_times: Dict[Tuple[int, str], str] = {}
        # ROI unique visitors count: roi_id -> set of track_ids
        self.roi_unique_visitors: Dict[str, Set[int]] = defaultdict(set)

        # Line crossing counts: line_id -> total crossings
        self.line_crossing_counts: Dict[str, int] = defaultdict(int)

        # Performance monitoring
        self.last_frame_time = time.perf_counter()
        self.processing_times: deque = deque(maxlen=30)
        self.fps_estimates: deque = deque(maxlen=30)

        # Backward-compatibility counters
        self.inbound_count = 0
        self.outbound_count = 0
        self.counted_tracks: Set[int] = set()
        self.recent_events: List[LineCrossingEvent] = []
        self.unique_objects: Set[int] = self.unique_track_ids
        self.class_totals: Dict[str, int] = defaultdict(int)

    # ----------------------------------------------------
    # Configuration Management (Lines & ROIs)
    # ----------------------------------------------------

    def add_line(self, line: LineDefinition) -> None:
        """Registers a virtual tripwire line."""
        p1, p2 = line.start_point, line.end_point
        if p1 == p2:
            raise InvalidLineError(f"Line start and end points cannot be identical: {p1}")
        self.lines[line.line_id] = line
        logger.info(f"Added virtual line tripwire: {line.line_id} ({line.start_point} -> {line.end_point})")

    def remove_line(self, line_id: str) -> None:
        self.lines.pop(line_id, None)

    def clear_lines(self) -> None:
        self.lines.clear()

    def add_roi(self, roi: ROIDefinition) -> None:
        """Registers a region of interest polygon."""
        if len(roi.points) < 3:
            raise InvalidROIError(f"ROI must contain at least 3 vertices. Received {len(roi.points)}: {roi.points}")
        self.rois[roi.roi_id] = roi
        logger.info(f"Added ROI: {roi.roi_id} ('{roi.name}') with {len(roi.points)} vertices")

    def remove_roi(self, roi_id: str) -> None:
        self.rois.pop(roi_id, None)

    def clear_rois(self) -> None:
        self.rois.clear()

    # ----------------------------------------------------
    # Event Logging
    # ----------------------------------------------------

    def _record_event(
        self,
        event_type: str,
        track_id: int,
        class_name: str,
        frame_number: int,
        timestamp: str,
        location: Tuple[float, float],
        details: Optional[Dict[str, Any]] = None,
    ) -> AnalyticsEvent:
        event = AnalyticsEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            track_id=track_id,
            class_name=class_name,
            timestamp=timestamp,
            frame_number=frame_number,
            location=location,
            details=details or {},
        )
        self.events.append(event)
        if len(self.events) > self.max_event_history:
            self.events.pop(0)
        return event

    def get_events(self, limit: Optional[int] = None, event_type: Optional[str] = None) -> List[AnalyticsEvent]:
        """Returns filtered recent analytics events."""
        evs = self.events if not event_type else [e for e in self.events if e.event_type == event_type]
        if limit:
            return evs[-limit:]
        return list(evs)

    def clear_events(self) -> None:
        self.events.clear()

    # ----------------------------------------------------
    # Frame Processing & Analytics Calculation
    # ----------------------------------------------------

    def update(
        self,
        tracking_input: Union[TrackingResult, List[Any]],
        processing_time_ms: Optional[float] = None,
    ) -> AnalyticsSnapshot:
        """
        Processes a TrackingResult (or list of TrackedObjects), updates lifecycle states,
        evaluates line crossings & ROIs, and returns an AnalyticsSnapshot.
        """
        start_time = time.perf_counter()
        now_dt = datetime.now(timezone.utc)
        default_timestamp = now_dt.isoformat()

        # Parse tracking input
        if isinstance(tracking_input, TrackingResult):
            tracks = tracking_input.objects
            frame_number = tracking_input.frame_number
            timestamp = tracking_input.timestamp or default_timestamp
            step_proc_time = tracking_input.processing_time_ms
        elif isinstance(tracking_input, list):
            tracks = tracking_input
            self.total_frames += 1
            frame_number = self.total_frames
            timestamp = default_timestamp
            step_proc_time = processing_time_ms or 0.0
        else:
            raise AnalyticsError(f"Unsupported tracking input type: {type(tracking_input).__name__}")

        self.total_frames = frame_number
        self.total_detections_count += len(tracks)

        current_active_ids: Set[int] = set()

        # Process each currently active tracked object
        for track in tracks:
            # Extract common attributes from TrackedObject or TrackedItem
            track_id = int(getattr(track, "track_id", 0))
            class_name = str(getattr(track, "class_name", "unknown"))
            class_id = int(getattr(track, "class_id", 0))
            confidence = float(getattr(track, "confidence", 1.0))

            # Bounding box extraction
            if hasattr(track, "box_xyxy"):
                box = track.box_xyxy
            elif hasattr(track, "box"):
                b = track.box
                box = (b.x1, b.y1, b.x2, b.y2) if hasattr(b, "x1") else (track.x1, track.y1, track.x2, track.y2)
            else:
                box = (getattr(track, "x1", 0.0), getattr(track, "y1", 0.0), getattr(track, "x2", 0.0), getattr(track, "y2", 0.0))

            anchor_pt = calculate_anchor_point(box, "bottom_center")
            center_pt = calculate_anchor_point(box, "center")

            current_active_ids.add(track_id)
            self.unique_track_ids.add(track_id)
            self.class_totals[class_name] = self.class_totals.get(class_name, 0) + 1

            # 1. Track Lifecycle: Registration & Entry
            if track_id not in self.track_data:
                self.track_data[track_id] = {
                    "track_id": track_id,
                    "class_name": class_name,
                    "class_id": class_id,
                    "first_seen_frame": frame_number,
                    "last_seen_frame": frame_number,
                    "first_seen_time": timestamp,
                    "last_seen_time": timestamp,
                    "status": "active",
                    "total_frames_visible": 1,
                    "last_anchor": anchor_pt,
                }
                # Emit OBJECT_ENTERED event
                self._record_event(
                    event_type="OBJECT_ENTERED",
                    track_id=track_id,
                    class_name=class_name,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    location=center_pt,
                    details={"box": list(box), "confidence": confidence},
                )
            else:
                td = self.track_data[track_id]
                td["last_seen_frame"] = frame_number
                td["last_seen_time"] = timestamp
                td["status"] = "active"
                td["total_frames_visible"] += 1
                td["last_anchor"] = anchor_pt

            # 2. Update Trajectory
            traj_pt = TrajectoryPoint(frame_number=frame_number, timestamp=timestamp, x=anchor_pt[0], y=anchor_pt[1])
            self.trajectories[track_id].append(traj_pt)

            # 3. Virtual Line Crossing Evaluation
            traj = self.trajectories[track_id]
            if len(traj) >= 2:
                prev_pos = (traj[-2].x, traj[-2].y)
                curr_pos = (traj[-1].x, traj[-1].y)

                for line_id, line in self.lines.items():
                    # Check anti-duplicate cooldown
                    last_cross_frame = self.line_crossing_cooldown.get((track_id, line_id), -999)
                    if frame_number - last_cross_frame > 15:
                        if lines_intersect(prev_pos, curr_pos, line.start_point, line.end_point):
                            direction = determine_crossing_direction(prev_pos, curr_pos, line.start_point, line.end_point)

                            if line.direction == "BOTH" or line.direction == direction:
                                self.line_crossing_cooldown[(track_id, line_id)] = frame_number
                                self.line_crossing_counts[line_id] += 1

                                # Backward-compatible counters
                                if direction == "A_TO_B":
                                    self.inbound_count += 1
                                else:
                                    self.outbound_count += 1

                                self._record_event(
                                    event_type="LINE_CROSSED",
                                    track_id=track_id,
                                    class_name=class_name,
                                    frame_number=frame_number,
                                    timestamp=timestamp,
                                    location=anchor_pt,
                                    details={"line_id": line_id, "direction": direction},
                                )
                                self.recent_events.append(
                                    LineCrossingEvent(
                                        track_id=track_id,
                                        class_name=class_name,
                                        direction=direction,
                                        timestamp=timestamp,
                                    )
                                )

            # 4. Region of Interest (ROI) Evaluation
            current_track_rois: Set[str] = set()
            for roi_id, roi in self.rois.items():
                eval_point = calculate_anchor_point(box, roi.anchor)
                is_inside = point_in_polygon(eval_point, roi.points)

                if is_inside:
                    current_track_rois.add(roi_id)
                    self.roi_unique_visitors[roi_id].add(track_id)

                    # Check if newly entered ROI
                    if roi_id not in self.track_active_rois[track_id]:
                        self.roi_entry_times[(track_id, roi_id)] = timestamp
                        self._record_event(
                            event_type="ROI_ENTER",
                            track_id=track_id,
                            class_name=class_name,
                            frame_number=frame_number,
                            timestamp=timestamp,
                            location=eval_point,
                            details={"roi_id": roi_id, "roi_name": roi.name},
                        )

            # Check if exited any previously active ROIs
            for prev_roi in list(self.track_active_rois[track_id]):
                if prev_roi not in current_track_rois:
                    # Exited ROI
                    entry_iso = self.roi_entry_times.pop((track_id, prev_roi), None)
                    dwell_sec = 0.0
                    if entry_iso:
                        try:
                            t_entry = datetime.fromisoformat(entry_iso)
                            dwell_sec = max(0.0, (now_dt - t_entry).total_seconds())
                        except Exception:
                            dwell_sec = 0.0

                    self.roi_dwell_accumulators[(track_id, prev_roi)] += dwell_sec
                    self._record_event(
                        event_type="ROI_EXIT",
                        track_id=track_id,
                        class_name=class_name,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        location=anchor_pt,
                        details={"roi_id": prev_roi, "dwell_seconds": round(dwell_sec, 2)},
                    )

            self.track_active_rois[track_id] = current_track_rois

        # 5. Handle Disappeared / Exited Objects
        for tid in list(self.track_data.keys()):
            if tid not in current_active_ids:
                td = self.track_data[tid]
                frames_missing = frame_number - td["last_seen_frame"]

                if frames_missing > self.exit_threshold_frames and td["status"] != "exited":
                    td["status"] = "exited"
                    last_loc = td.get("last_anchor", (0.0, 0.0))

                    # Emit OBJECT_EXITED event
                    self._record_event(
                        event_type="OBJECT_EXITED",
                        track_id=tid,
                        class_name=td["class_name"],
                        frame_number=frame_number,
                        timestamp=timestamp,
                        location=last_loc,
                        details={"last_seen_frame": td["last_seen_frame"]},
                    )

                    # Also trigger ROI_EXIT for any ROIs the object was inside
                    for active_roi in list(self.track_active_rois[tid]):
                        self._record_event(
                            event_type="ROI_EXIT",
                            track_id=tid,
                            class_name=td["class_name"],
                            frame_number=frame_number,
                            timestamp=timestamp,
                            location=last_loc,
                            details={"roi_id": active_roi, "forced_exit": True},
                        )
                    self.track_active_rois[tid].clear()
                elif td["status"] == "active":
                    td["status"] = "disappeared"

        self.active_track_ids = current_active_ids

        # 6. Performance Telemetry
        now_time = time.perf_counter()
        elapsed_since_last = now_time - self.last_frame_time
        self.last_frame_time = now_time
        if elapsed_since_last > 0:
            self.fps_estimates.append(1.0 / elapsed_since_last)

        analytics_time_ms = (now_time - start_time) * 1000.0
        total_pipeline_ms = (step_proc_time or 0.0) + analytics_time_ms
        self.processing_times.append(total_pipeline_ms)

        return self.get_snapshot()

    # ----------------------------------------------------
    # Analytics Snapshot Generation
    # ----------------------------------------------------

    def get_snapshot(self) -> AnalyticsSnapshot:
        """Builds a complete, point-in-time structured AnalyticsSnapshot."""
        now_iso = datetime.now(timezone.utc).isoformat()
        current_dt = datetime.now(timezone.utc)

        # Active tracks statistics
        active_track_stats: List[TrackStatistics] = []
        class_active_counts: Dict[str, int] = defaultdict(int)
        class_unique_counts: Dict[str, Set[int]] = defaultdict(set)
        class_dwell_durations: Dict[str, List[float]] = defaultdict(list)

        for tid, td in self.track_data.items():
            class_unique_counts[td["class_name"]].add(tid)

            # Calculate total duration
            try:
                t_first = datetime.fromisoformat(td["first_seen_time"])
                t_last = datetime.fromisoformat(td["last_seen_time"])
                duration_sec = max(0.0, (t_last - t_first).total_seconds())
            except Exception:
                duration_sec = 0.0

            class_dwell_durations[td["class_name"]].append(duration_sec)

            if tid in self.active_track_ids:
                class_active_counts[td["class_name"]] += 1
                rois_for_track = list(self.track_active_rois[tid])
                dwell_dict = {
                    roi_id: round(self.roi_dwell_accumulators[(tid, roi_id)], 2)
                    for roi_id in self.rois
                    if (tid, roi_id) in self.roi_dwell_accumulators
                }

                stat = TrackStatistics(
                    track_id=tid,
                    class_name=td["class_name"],
                    first_seen_frame=td["first_seen_frame"],
                    last_seen_frame=td["last_seen_frame"],
                    first_seen_time=td["first_seen_time"],
                    last_seen_time=td["last_seen_time"],
                    duration_seconds=round(duration_sec, 2),
                    status=td["status"],
                    total_frames_visible=td["total_frames_visible"],
                    current_rois=rois_for_track,
                    roi_dwell_times=dwell_dict,
                )
                active_track_stats.append(stat)

        # Class statistics
        class_stats_map: Dict[str, ClassStatistics] = {}
        all_classes = set(class_unique_counts.keys()) | set(class_active_counts.keys())
        for cname in all_classes:
            durations = class_dwell_durations.get(cname, [])
            avg_dwell = round(float(np.mean(durations)), 2) if durations else 0.0

            entered_count = sum(1 for e in self.events if e.event_type == "OBJECT_ENTERED" and e.class_name == cname)
            exited_count = sum(1 for e in self.events if e.event_type == "OBJECT_EXITED" and e.class_name == cname)
            crossing_count = sum(1 for e in self.events if e.event_type == "LINE_CROSSED" and e.class_name == cname)
            roi_in_count = sum(1 for e in self.events if e.event_type == "ROI_ENTER" and e.class_name == cname)
            roi_out_count = sum(1 for e in self.events if e.event_type == "ROI_EXIT" and e.class_name == cname)

            class_stats_map[cname] = ClassStatistics(
                class_name=cname,
                active_count=class_active_counts[cname],
                unique_count=len(class_unique_counts[cname]),
                entered_count=entered_count,
                exited_count=exited_count,
                line_crossing_count=crossing_count,
                roi_entry_count=roi_in_count,
                roi_exit_count=roi_out_count,
                avg_dwell_time_seconds=avg_dwell,
            )

        # ROI counts & dwell times
        roi_current_counts: Dict[str, int] = defaultdict(int)
        for tid in self.active_track_ids:
            for rid in self.track_active_rois[tid]:
                roi_current_counts[rid] += 1

        roi_avg_dwell: Dict[str, float] = {}
        for rid in self.rois:
            times = [self.roi_dwell_accumulators[(tid, rid)] for (tid, r) in self.roi_dwell_accumulators if r == rid]
            roi_avg_dwell[rid] = round(float(np.mean(times)), 2) if times else 0.0

        # Performance metrics
        curr_fps = round(self.fps_estimates[-1], 1) if self.fps_estimates else 0.0
        avg_fps = round(float(np.mean(self.fps_estimates)), 1) if self.fps_estimates else 0.0
        last_ms = round(self.processing_times[-1], 2) if self.processing_times else 0.0
        avg_ms = round(float(np.mean(self.processing_times)), 2) if self.processing_times else 0.0

        perf = PerformanceMetrics(
            current_fps=curr_fps,
            average_fps=avg_fps,
            last_processing_time_ms=last_ms,
            average_processing_time_ms=avg_ms,
            total_frames=self.total_frames,
        )

        return AnalyticsSnapshot(
            session_id=self.session_id,
            frame_number=self.total_frames,
            timestamp=now_iso,
            active_objects=len(self.active_track_ids),
            total_unique_objects=len(self.unique_track_ids),
            total_detections=self.total_detections_count,
            class_distribution=dict(class_active_counts),
            class_statistics=class_stats_map,
            active_tracks=active_track_stats,
            line_crossings=dict(self.line_crossing_counts),
            roi_counts=dict(roi_current_counts),
            roi_dwell_times=roi_avg_dwell,
            performance=perf,
            recent_events=list(self.events[-15:]),
        )
