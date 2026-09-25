from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.vision.detector import (
    Detection,
    DetectionEngine,
    DetectionError,
    DetectionResult,
    get_detector,
)

logger = get_logger("vision.tracker")


# ==========================================
# Tracking Exceptions Hierarchy
# ==========================================

class TrackingError(DetectionError):
    """Base exception for all tracking errors."""
    pass


class TrackerConfigError(TrackingError):
    """Raised when tracker configuration parameters are invalid."""
    pass


class TrackerUpdateError(TrackingError):
    """Raised when tracker update fails during association or state estimation."""
    pass


# ==========================================
# Track State Enum
# ==========================================

class TrackState(str, Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


# ==========================================
# Tracked Object & Tracking Result
# ==========================================

@dataclass
class TrackedObject:
    """Represents an individual tracked entity with a persistent ID across frames."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_number: Optional[int] = None
    timestamp: Optional[str] = None
    tracking_state: str = TrackState.ACTIVE.value
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "box": {
                "x1": round(float(self.x1), 2),
                "y1": round(float(self.y1), 2),
                "x2": round(float(self.x2), 2),
                "y2": round(float(self.y2), 2),
            },
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "tracking_state": self.tracking_state,
            "trajectory": [
                (round(float(pt[0]), 1), round(float(pt[1]), 1))
                for pt in self.trajectory
            ],
        }


@dataclass
class TrackingResult:
    """Encapsulates tracking outputs for a single frame."""
    frame_number: int
    timestamp: str
    objects: List[TrackedObject]
    active_track_count: int
    processing_time_ms: float
    session_id: Optional[str] = None
    cumulative_unique_tracks: int = 0

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self.objects:
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return counts

    def filter_by_class(self, class_names: Union[str, List[str], Set[str]]) -> List[TrackedObject]:
        targets = {class_names} if isinstance(class_names, str) else set(class_names)
        return [o for o in self.objects if o.class_name in targets]

    def filter_by_confidence(self, min_conf: float) -> List[TrackedObject]:
        return [o for o in self.objects if o.confidence >= min_conf]

    def get_object_by_id(self, track_id: int) -> Optional[TrackedObject]:
        for o in self.objects:
            if o.track_id == track_id:
                return o
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "active_track_count": self.active_track_count,
            "cumulative_unique_tracks": self.cumulative_unique_tracks,
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "class_counts": self.class_counts,
            "objects": [o.to_dict() for o in self.objects],
            "tracks": [o.to_dict() for o in self.objects],
        }


# ==========================================
# Kalman Filter & Track Representation
# ==========================================

class SingleTrack:
    """Maintains state, trajectory, and position estimation for an individual tracked target."""
    _next_id = 1

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 1

    def __init__(
        self,
        detection: Detection,
        frame_number: int,
        timestamp: Optional[str] = None,
        max_trajectory: int = 30,
    ):
        self.track_id = SingleTrack._next_id
        SingleTrack._next_id += 1

        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.box = (detection.x1, detection.y1, detection.x2, detection.y2)
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.state = TrackState.ACTIVE
        self.time_since_update = 0
        self.hits = 1
        self.max_trajectory = max_trajectory

        cx = (detection.x1 + detection.x2) / 2.0
        cy = (detection.y1 + detection.y2) / 2.0
        self.trajectory: deque[Tuple[float, float]] = deque([(cx, cy)], maxlen=max_trajectory)

        w = max(1.0, detection.x2 - detection.x1)
        h = max(1.0, detection.y2 - detection.y1)
        self.state_vector = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float32)

    def predict(self) -> Tuple[float, float, float, float]:
        """Predicts the next bounding box position based on smoothed velocity."""
        self.state_vector[0] += self.state_vector[4]  # cx += vx
        self.state_vector[1] += self.state_vector[5]  # cy += vy

        # Dampen velocity slightly to avoid runaway predictions
        self.state_vector[4] *= 0.90
        self.state_vector[5] *= 0.90

        cx, cy, w, h = self.state_vector[0], self.state_vector[1], self.state_vector[2], self.state_vector[3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        self.box = (x1, y1, x2, y2)
        self.time_since_update += 1
        return self.box

    def update(self, detection: Detection, frame_number: int, timestamp: Optional[str] = None) -> None:
        """Updates track with a new matched detection."""
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name

        new_cx = (detection.x1 + detection.x2) / 2.0
        new_cy = (detection.y1 + detection.y2) / 2.0
        new_w = max(1.0, detection.x2 - detection.x1)
        new_h = max(1.0, detection.y2 - detection.y1)

        # Update velocity with smoothing (alpha = 0.5)
        if self.time_since_update > 0:
            vx = (new_cx - self.state_vector[0]) / self.time_since_update
            vy = (new_cy - self.state_vector[1]) / self.time_since_update
            self.state_vector[4] = 0.5 * vx + 0.5 * self.state_vector[4]
            self.state_vector[5] = 0.5 * vy + 0.5 * self.state_vector[5]

        self.state_vector[0] = new_cx
        self.state_vector[1] = new_cy
        self.state_vector[2] = new_w
        self.state_vector[3] = new_h

        self.box = (detection.x1, detection.y1, detection.x2, detection.y2)
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.ACTIVE

        # O(1) bounded ring buffer append without list shifting
        self.trajectory.append((new_cx, new_cy))

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def to_tracked_object(self) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=float(self.confidence),
            x1=float(self.box[0]),
            y1=float(self.box[1]),
            x2=float(self.box[2]),
            y2=float(self.box[3]),
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            tracking_state=self.state.value,
            trajectory=list(self.trajectory),
        )


# ==========================================
# Bounding Box IoU & Association Utilities
# ==========================================

def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def associate_detections_to_tracks(
    tracks: List[SingleTrack],
    detections: List[Detection],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associates tracks with detections based on IoU score using greedy maximum matching.
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            if track.class_id == det.class_id:
                iou_matrix[t_idx, d_idx] = compute_iou(track.box, det.box_xyxy)
            else:
                iou_matrix[t_idx, d_idx] = 0.0

    matches: List[Tuple[int, int]] = []
    matched_tracks: Set[int] = set()
    matched_dets: Set[int] = set()

    candidates: List[Tuple[float, int, int]] = []
    for t_idx in range(len(tracks)):
        for d_idx in range(len(detections)):
            score = iou_matrix[t_idx, d_idx]
            if score >= iou_threshold:
                candidates.append((score, t_idx, d_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for score, t_idx, d_idx in candidates:
        if t_idx not in matched_tracks and d_idx not in matched_dets:
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            matches.append((t_idx, d_idx))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_detections = [j for j in range(len(detections)) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_detections


# ==========================================
# Abstract Tracker Base Class
# ==========================================

class Tracker(ABC):
    """Abstract base class for multi-object trackers consuming DetectionResult."""

    @abstractmethod
    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        """Updates tracking state using detections from DetectionResult."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets all internal tracks and state counters."""
        pass


# ==========================================
# ByteTrack Implementation
# ==========================================

class ByteTrackTracker(Tracker):
    """
    ByteTrack baseline implementation.
    Associates high-confidence detections first, then low-confidence detections,
    maintaining persistent IDs and handling occlusion and track recovery.
    """

    def __init__(
        self,
        track_high_thresh: Optional[float] = None,
        track_low_thresh: Optional[float] = None,
        track_match_thresh: Optional[float] = None,
        persistence_buffer: Optional[int] = None,
        session_id: Optional[str] = None,
    ):
        self.track_high_thresh = track_high_thresh if track_high_thresh is not None else settings.TRACK_HIGH_THRESH
        self.track_low_thresh = track_low_thresh if track_low_thresh is not None else settings.TRACK_LOW_THRESH
        self.track_match_thresh = track_match_thresh if track_match_thresh is not None else 0.5
        self.persistence_buffer = persistence_buffer if persistence_buffer is not None else settings.TRACK_PERSISTENCE_BUFFER
        self.session_id = session_id

        if not (0.0 <= self.track_low_thresh <= self.track_high_thresh <= 1.0):
            raise TrackerConfigError(
                f"Invalid threshold ordering: track_low_thresh ({self.track_low_thresh}) "
                f"must be <= track_high_thresh ({self.track_high_thresh})"
            )

        self.tracked_tracks: List[SingleTrack] = []
        self.lost_tracks: List[SingleTrack] = []
        self.frame_count = 0
        self.cumulative_unique_ids: Set[int] = set()

        logger.info(
            f"Initialized ByteTrackTracker(high={self.track_high_thresh}, "
            f"low={self.track_low_thresh}, match={self.track_match_thresh}, buffer={self.persistence_buffer})"
        )

    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        """Executes the two-stage ByteTrack association algorithm on detection results."""
        if not isinstance(detection_result, DetectionResult):
            raise TrackerUpdateError(
                f"Expected DetectionResult instance, received: {type(detection_result).__name__}"
            )

        start_time = time.perf_counter()
        self.frame_count += 1
        frame_number = detection_result.frame_number or self.frame_count
        timestamp = detection_result.timestamp or datetime.now(timezone.utc).isoformat()

        # 1. Predict locations of existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # 2. Partition detections into high and low confidence sets
        detections_high: List[Detection] = []
        detections_low: List[Detection] = []

        for det in detection_result.detections:
            if det.confidence >= self.track_high_thresh:
                detections_high.append(det)
            elif det.confidence >= self.track_low_thresh:
                detections_low.append(det)

        # Candidate tracks for first association: active + lost tracks
        candidate_tracks = self.tracked_tracks + self.lost_tracks

        # 3. First association: candidate tracks with high-confidence detections
        matches_a, unmatched_tracks_a, unmatched_dets_a = associate_detections_to_tracks(
            candidate_tracks,
            detections_high,
            self.track_match_thresh,
        )

        for t_idx, d_idx in matches_a:
            track = candidate_tracks[t_idx]
            det = detections_high[d_idx]
            track.update(det, frame_number, timestamp)
            self.cumulative_unique_ids.add(track.track_id)

        # 4. Second association: remaining candidate tracks with low-confidence detections
        remaining_tracks = [candidate_tracks[i] for i in unmatched_tracks_a]
        matches_b, unmatched_tracks_b, _ = associate_detections_to_tracks(
            remaining_tracks,
            detections_low,
            iou_threshold=0.4,  # Lower threshold for occluded/recovery matches
        )

        for t_idx, d_idx in matches_b:
            track = remaining_tracks[t_idx]
            det = detections_low[d_idx]
            track.update(det, frame_number, timestamp)
            self.cumulative_unique_ids.add(track.track_id)

        # 5. Handle remaining unmatched tracks: mark lost or removed
        still_unmatched_tracks = [remaining_tracks[i] for i in unmatched_tracks_b]
        new_lost_tracks: List[SingleTrack] = []
        new_tracked_tracks: List[SingleTrack] = []

        for track in candidate_tracks:
            if track.time_since_update == 0:
                new_tracked_tracks.append(track)
            elif track.time_since_update <= self.persistence_buffer:
                track.mark_lost()
                new_lost_tracks.append(track)
            else:
                track.mark_removed()

        # 6. Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_dets_a:
            det = detections_high[d_idx]
            new_track = SingleTrack(det, frame_number, timestamp)
            new_tracked_tracks.append(new_track)
            self.cumulative_unique_ids.add(new_track.track_id)

        self.tracked_tracks = new_tracked_tracks
        self.lost_tracks = new_lost_tracks

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0

        active_objects = [t.to_tracked_object() for t in self.tracked_tracks]

        return TrackingResult(
            frame_number=frame_number,
            timestamp=timestamp,
            objects=active_objects,
            active_track_count=len(active_objects),
            processing_time_ms=processing_time_ms,
            session_id=self.session_id,
            cumulative_unique_tracks=len(self.cumulative_unique_ids),
        )

    def reset(self) -> None:
        """Resets all track buffers and resets track IDs."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.frame_count = 0
        self.cumulative_unique_ids.clear()
        SingleTrack.reset_id_counter()
        logger.info("ByteTrackTracker reset successfully.")


# ==========================================
# Tracking Visualization Utility
# ==========================================

TRACK_COLORS = [
    (56, 189, 248),   # Sky blue
    (52, 211, 153),   # Emerald green
    (251, 191, 36),   # Amber
    (244, 114, 182),  # Pink
    (167, 139, 250),  # Purple
    (251, 146, 60),   # Orange
    (74, 222, 128),   # Lime
    (96, 165, 250),   # Blue
    (248, 113, 113),  # Red
    (45, 212, 191),   # Teal
]


def annotate_tracking_frame(
    frame: np.ndarray,
    tracking_result: TrackingResult,
    draw_trajectories: bool = True,
) -> np.ndarray:
    """
    Renders bounding boxes, persistent track IDs, class names, and confidence scores onto an image.
    Decoupled from tracker logic.
    Example label: 'Person #7 0.91'
    """
    canvas = frame.copy()

    for obj in tracking_result.objects:
        color = TRACK_COLORS[obj.track_id % len(TRACK_COLORS)]
        p1 = (int(obj.x1), int(obj.y1))
        p2 = (int(obj.x2), int(obj.y2))

        # Draw bounding box
        cv2.rectangle(canvas, p1, p2, color, 2)

        # Label: Class Name #ID Confidence
        label = f"{obj.class_name.capitalize()} #{obj.track_id} {obj.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_bg_top = max(0, p1[1] - text_h - 6)
        cv2.rectangle(canvas, (p1[0], label_bg_top), (p1[0] + text_w + 4, p1[1]), color, -1)
        cv2.putText(
            canvas,
            label,
            (p1[0] + 2, p1[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if draw_trajectories and obj.trajectory and len(obj.trajectory) > 1:
            pts = np.array(obj.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

    return canvas


# ==========================================
# ObjectTracker Pipeline Adapter
# ==========================================

class ObjectTracker:
    """
    Pipeline adapter providing end-to-end Tracking by composing
    the existing DetectionEngine with ByteTrackTracker.
    Maintains backward compatibility with earlier schemas while separating detection and tracking.
    """

    def __init__(
        self,
        session_id: str,
        tracker_type: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_trajectory_points: int = 30,
        detector: Optional[DetectionEngine] = None,
    ):
        self.session_id = session_id
        self.detector = detector or get_detector()
        self.tracker = ByteTrackTracker(session_id=session_id)
        self.max_trajectory_points = max_trajectory_points
        self.tracker_type = tracker_type or settings.TRACKER_TYPE

    def update(
        self,
        frame: Union[np.ndarray, Any],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
    ) -> Any:
        """
        Runs DetectionEngine on the frame, feeds DetectionResult to ByteTrackTracker,
        and returns the tracking result.
        """
        det_result = self.detector.detect(
            image=frame,
            conf=conf,
            iou=iou,
            target_classes=target_classes,
        )

        track_result = self.tracker.update(det_result)
        track_result.session_id = self.session_id

        from backend.app.api.schemas.detection import BoundingBox
        from backend.app.api.schemas.tracking import TrackedItem, TrackingFrameResponse

        tracked_items = [
            TrackedItem(
                track_id=o.track_id,
                class_id=o.class_id,
                class_name=o.class_name,
                confidence=o.confidence,
                box=BoundingBox(x1=o.x1, y1=o.y1, x2=o.x2, y2=o.y2),
                trajectory=o.trajectory,
            )
            for o in track_result.objects
        ]

        return TrackingFrameResponse(
            session_id=self.session_id,
            frame_number=track_result.frame_number,
            inference_time_ms=det_result.inference_time_ms,
            device=det_result.device_used,
            active_tracks_count=track_result.active_track_count,
            cumulative_unique_tracks=track_result.cumulative_unique_tracks,
            class_counts=track_result.class_counts,
            tracks=tracked_items,
        )

    def reset(self) -> None:
        self.tracker.reset()
