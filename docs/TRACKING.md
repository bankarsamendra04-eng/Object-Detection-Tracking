# Multi-Object Tracking Engine Documentation (ByteTrack)

This document describes the multi-object tracking implementation, persistent ID lifecycle, two-stage data association, and trajectory management in the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Tracking Pipeline Architecture

The tracking engine (`backend/app/services/vision/tracker.py`) consumes structured `DetectionResult` objects from the detection engine without duplicating YOLO inference:

```
Frame Ingestion (Video / Webcam)
       ↓
DetectionEngine (YOLOv8)
       ↓  Structured Detections: [bbox, conf, class_id]
ByteTrackTracker
  ├── Stage 1: High-Confidence Association (IoU matching)
  ├── Stage 2: Low-Confidence Association (Occlusion recovery)
  ├── Track State Machine (NEW -> CONFIRMED -> LOST -> REMOVED)
  └── Trajectory Ring Buffer (O(1) Centroid History)
       ↓
TrackingResult (TrackedObject with Persistent IDs)
```

---

## 2. Two-Stage Data Association Algorithm

Traditional trackers discard low-confidence detections below the detection threshold, causing track breaks when objects are partially occluded or blurred. ByteTrack retains all detections and performs two-stage association:

1. **Stage 1 (High-Confidence Association):**
   - Evaluates detections with $\text{confidence} \ge \text{track\_high\_thresh}$ (default: `0.5`).
   - Computes Intersection-over-Union (IoU) distance matrix between confirmed tracks and high-score detections.
   - Solves the linear assignment problem using the Hungarian algorithm.

2. **Stage 2 (Low-Confidence Association):**
   - Matches remaining unmatched confirmed tracks against low-confidence detections ($\text{track\_low\_thresh} \le \text{confidence} < \text{track\_high\_thresh}$, default: `0.1` to `0.5`).
   - Recovers targets undergoing temporary occlusion or motion blur without creating false-positive new tracks.

---

## 3. Persistent Track Lifecycle & State Machine

Tracks transition through four explicit states:

```
  [Detection]
       ↓
     (NEW) ── (Matches in consecutive frames) ──> (CONFIRMED)
                                                      │
                                           (Missed detection)
                                                      ↓
                                                    (LOST)
                                                      │
                                           (> persistence_buffer)
                                                      ↓
                                                  (REMOVED)
```

- **Persistence Buffer (`persistence_buffer: 30`):** When an object is temporarily obscured (e.g. passing behind a pillar or tree), the track enters the `LOST` state. It is retained in memory for up to 30 frames. If re-detected within 30 frames, its persistent ID is preserved.
- **Track Deletion:** Tracks exceeding the persistence buffer are cleanly evicted, releasing internal state.

---

## 4. $O(1)$ Trajectory Ring Buffers

To visualize object paths and feed spatial analytics, each active track retains centroid coordinates:

- **Data Structure:** `collections.deque(maxlen=self.max_trajectory)` (default: 30 centroids).
- **Constant Time Eviction:** Appending a new centroid automatically evicts the oldest point in $O(1)$ constant time, eliminating the $O(N)$ memory copies inherent in Python list slicing.

---

## 5. Tracker Configuration Parameters

| Parameter | Type | Default | Operational Impact |
| :--- | :--- | :--- | :--- |
| `track_high_thresh` | float | `0.5` | Threshold for primary association stage. Higher values decrease false positive associations. |
| `track_low_thresh` | float | `0.1` | Floor threshold for second-stage occlusion recovery. |
| `track_match_thresh`| float | `0.8` | Minimum IoU overlap required to validate a track-detection association. |
| `persistence_buffer`| int | `30` | Number of consecutive missed frames before a lost track is removed. |
| `max_trajectory` | int | `30` | Number of historical centroid points stored in the trajectory ring buffer. |
