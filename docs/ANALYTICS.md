# Spatial Analytics Engine Documentation

This document describes the spatial analytics engine, line-crossing tripwires, cumulative counting, and telemetry computation in the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Analytics Pipeline Architecture

The Analytics Engine (`backend/app/services/analytics/counter.py`) sits downstream of the tracking engine:

```
TrackingResult (Tracked Objects with Persistent IDs)
       ↓
AnalyticsEngine
  ├── 1. Class Distribution Aggregator (Active & Cumulative)
  ├── 2. Unique Entity Registry (Deduplication across time)
  ├── 3. Spatial Line Tripwire Evaluator (2D Vector Cross-Products)
  └── 4. Pipeline Telemetry Profiler (FPS, Latency meters)
       ↓
AnalyticsReport (Exposed via REST & WebSocket)
```

---

## 2. Virtual Line-Crossing Tripwires

The platform supports directional virtual line tripwires (e.g. traffic counters, doorway ingress/egress).

### Mathematical Intersection Check
Line crossing is computed using counter-clockwise (`CCW`) orientation tests on 2D vectors without external geometry libraries:

$$\text{CCW}(A, B, C) = (C_y - A_y)(B_x - A_x) > (B_y - A_y)(C_x - A_x)$$

Two line segments $\overline{AB}$ (the virtual tripwire) and $\overline{CD}$ (the object's centroid motion vector between $t-1$ and $t$) intersect if and only if:
$$\text{CCW}(A, C, D) \ne \text{CCW}(B, C, D) \quad \text{AND} \quad \text{CCW}(A, B, C) \ne \text{CCW}(A, B, D)$$

### Directional Resolution
The sign of the 2D cross-product determines whether the object crossed in the **Forward** or **Backward** direction:
$$\vec{V}_{\text{tripwire}} \times \vec{V}_{\text{motion}} = (B_x - A_x)(D_y - C_y) - (B_y - A_y)(D_x - C_x)$$

- Cross-product $> 0$: **Forward Crossing** (e.g. Entry / Inbound)
- Cross-product $< 0$: **Backward Crossing** (e.g. Exit / Outbound)

---

## 3. Stateful Entity Counting & Deduplication

- **Unique Cumulative Entities:** A hash set (`Set[int]`) stores all seen track IDs across the session. An object that stays in frame for 500 frames is counted exactly once.
- **Active Entity Count:** The number of currently tracked objects actively visible in the current frame.
- **Class Breakdown:** Real-time histogram tallying unique objects per COCO/VisDrone class category.

---

## 4. Pipeline Telemetry Profiling

For every streaming frame, the analytics engine records and calculates:
- **`inference_ms`:** GPU/CPU YOLO model evaluation latency.
- **`tracking_ms`:** ByteTrack association and Kalman/trajectory update latency.
- **`render_ms`:** OpenCV visual annotation and JPEG base64 encoding latency.
- **`effective_fps`:** Moving window frame rate based on real frame processing intervals.

---

## 5. Analytical Limitations & Edge Cases

1. **Occlusion Re-identification:** If an object is lost for longer than `persistence_buffer` (30 frames), it will receive a new track ID upon reappearing, incrementing unique counts.
2. **Tripwire Position:** Centroids must physically cross the line segment between two consecutive sampled frames. Very fast objects sampled at low frame rates may jump across lines if the sampling stride is too large.
