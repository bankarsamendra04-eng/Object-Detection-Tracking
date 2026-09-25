import asyncio
import base64
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
from fastapi import WebSocket
import numpy as np

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.core.logging import get_logger
from backend.app.services.vision.detector import DetectionEngine, get_detector
from backend.app.services.vision.tracker import ByteTrackTracker, TrackingResult
from backend.app.services.analytics.counter import AnalyticsEngine
from backend.app.services.vision.input_sources import (
    InputSource,
    VideoInput,
    WebcamInput,
    MediaSourceError,
    DeviceUnavailableError,
    FileNotFoundMediaError,
    UnsupportedFormatError,
    CorruptMediaError,
)
from backend.app.api.schemas.websocket import (
    ClientAction,
    ClientCommand,
    ConnectionAckData,
    FrameResultData,
    ServerMessageType,
    StreamErrorData,
    StreamSourceType,
    StreamStartedData,
    StreamStoppedData,
    WebSocketMessage,
)

logger = get_logger("streaming.session")


class StreamSession:
    """
    Manages an active WebSocket streaming session.
    Isolates per-session CV state (ByteTrackTracker and AnalyticsEngine) while
    reusing the shared DetectionEngine YOLO singleton.
    Provides backpressure protection via bounded async queues and frame-rate control.
    """

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket

        # Shared singleton detector (loaded once, pre-warmed)
        self.detector: DetectionEngine = get_detector()

        # Session-isolated tracker and analytics engine
        self.tracker = ByteTrackTracker(session_id=session_id)
        self.analytics = AnalyticsEngine(session_id=session_id)

        # Media input
        self.input_source: Optional[InputSource] = None
        self.source_type: Optional[str] = None

        # State controls
        self.is_running = False
        self.is_paused = False
        self.fps_limit: float = 15.0
        self.stride: int = 1
        self.conf: Optional[float] = None
        self.iou: Optional[float] = None
        self.target_classes: Optional[List[str]] = None
        self.annotate: bool = False

        # Metrics
        self.frames_processed = 0
        self.start_timestamp: float = 0.0

        # Async tasks & bounded queue for backpressure protection
        self.processing_task: Optional[asyncio.Task] = None
        self.sender_task: Optional[asyncio.Task] = None
        self.send_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=10)

    # ----------------------------------------------------
    # Safe Message Transmission & Queue Management
    # ----------------------------------------------------

    async def enqueue_message(self, message: WebSocketMessage) -> None:
        """
        Enqueues a message into the bounded queue.
        If the queue is full (slow client / backpressure), drops the oldest frame
        message to avoid unbounded memory growth and server stalls.
        """
        raw_json = message.model_dump_json()
        try:
            self.send_queue.put_nowait(raw_json)
        except asyncio.QueueFull:
            # Backpressure: Client cannot keep up with transmission rate
            if message.type == ServerMessageType.FRAME_RESULT:
                try:
                    # Drop oldest unconsumed message to make room for newest state
                    _ = self.send_queue.get_nowait()
                    self.send_queue.put_nowait(raw_json)
                    logger.debug(f"Backpressure: Dropped oldest frame message for session={self.session_id}")
                except Exception:
                    pass
            else:
                # Critical messages (errors, stopped, ack) await queue availability
                await self.send_queue.put(raw_json)

    async def _sender_loop(self) -> None:
        """Dedicated sender worker dequeuing messages and sending them over WebSocket."""
        try:
            while True:
                msg = await self.send_queue.get()
                if msg is None:  # Poison pill to shut down
                    break
                await self.websocket.send_text(msg)
                self.send_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Sender loop terminated for session={self.session_id}: {e}")

    # ----------------------------------------------------
    # Command Handling
    # ----------------------------------------------------

    async def handle_command(self, cmd: ClientCommand) -> None:
        """Processes an incoming client command."""
        logger.info(f"Handling command '{cmd.action}' for session={self.session_id}")

        if cmd.action == ClientAction.PING:
            msg = WebSocketMessage(
                type=ServerMessageType.PONG,
                session_id=self.session_id,
                data={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
            await self.enqueue_message(msg)

        elif cmd.action == ClientAction.PAUSE:
            self.is_paused = True
            logger.info(f"Session {self.session_id} paused.")

        elif cmd.action == ClientAction.RESUME:
            self.is_paused = False
            logger.info(f"Session {self.session_id} resumed.")

        elif cmd.action == ClientAction.STOP:
            await self.stop_stream(reason="User stopped stream")

        elif cmd.action == ClientAction.START:
            await self.start_stream(cmd)

    # ----------------------------------------------------
    # Stream Initialization & Control
    # ----------------------------------------------------

    async def start_stream(self, cmd: ClientCommand) -> None:
        """Initializes input source, configures processing parameters, and starts tasks."""
        if self.is_running:
            logger.info(f"Stream already running on session={self.session_id}. Restarting with new config.")
            await self.stop_stream(reason="Restarting with new source")

        self.fps_limit = cmd.fps_limit
        self.stride = cmd.stride
        self.conf = cmd.conf
        self.iou = cmd.iou
        self.target_classes = cmd.target_classes
        self.annotate = cmd.annotate

        # Validate and open media source
        try:
            if cmd.source == StreamSourceType.WEBCAM:
                self.source_type = "webcam"
                self.input_source = await asyncio.to_thread(
                    WebcamInput,
                    camera_index=cmd.camera_index,
                )
            elif cmd.source == StreamSourceType.VIDEO:
                self.source_type = "video"
                resolved_path = self._resolve_video_path(cmd.path)
                self.input_source = await asyncio.to_thread(
                    VideoInput,
                    source=resolved_path,
                    stride=cmd.stride,
                    max_frames=cmd.max_frames,
                )
            else:
                raise ValueError(f"Unsupported source type: {cmd.source}")

        except (DeviceUnavailableError, FileNotFoundMediaError, UnsupportedFormatError, CorruptMediaError, ValueError) as err:
            logger.warning(f"Failed to open source for session={self.session_id}: {err}")
            error_msg = WebSocketMessage(
                type=ServerMessageType.ERROR,
                session_id=self.session_id,
                data=StreamErrorData(
                    code="SOURCE_UNAVAILABLE",
                    message=str(err),
                ),
            )
            await self.enqueue_message(error_msg)
            return
        except Exception as e:
            logger.error(f"Unexpected error opening source for session={self.session_id}: {e}")
            error_msg = WebSocketMessage(
                type=ServerMessageType.ERROR,
                session_id=self.session_id,
                data=StreamErrorData(
                    code="INTERNAL_ERROR",
                    message="Failed to open requested media source.",
                ),
            )
            await self.enqueue_message(error_msg)
            return

        # Start sender loop if not active
        if self.sender_task is None or self.sender_task.done():
            self.sender_task = asyncio.create_task(self._sender_loop())

        # Notify client of stream start
        self.is_running = True
        self.is_paused = False
        self.frames_processed = 0
        self.start_timestamp = time.perf_counter()

        start_ack = WebSocketMessage(
            type=ServerMessageType.STREAM_STARTED,
            session_id=self.session_id,
            data=StreamStartedData(
                session_id=self.session_id,
                source=self.source_type,
                metadata=self.input_source.metadata,
                fps_limit=self.fps_limit,
            ),
        )
        await self.enqueue_message(start_ack)

        # Launch background frame-processing loop
        self.processing_task = asyncio.create_task(self._processing_loop())

    def _resolve_video_path(self, raw_path: Optional[str]) -> Path:
        """
        Validates and safely resolves video file path within project directory boundaries.
        Prevents arbitrary filesystem traversal, UNC injection, and restricts access to allowed media directories.
        """
        if not raw_path:
            raise ValueError("Video source requires 'path' parameter.")

        try:
            candidate = settings.resolve_safe_path(raw_path)
        except ValueError as e:
            raise ValueError(f"Invalid media path: {e}")

        # Enforce that video files must reside within allowed media directories (data or outputs)
        allowed_subdirs = [
            (PROJECT_ROOT / "data").resolve(),
            (PROJECT_ROOT / "outputs").resolve(),
        ]
        is_allowed_dir = False
        for sdir in allowed_subdirs:
            try:
                candidate.relative_to(sdir)
                is_allowed_dir = True
                break
            except ValueError:
                continue

        if not is_allowed_dir:
            raise ValueError(
                f"Access denied: Media path '{raw_path}' must reside within 'data/' or 'outputs/' directories."
            )

        ext = candidate.suffix.lower()
        if ext not in VideoInput.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported video extension '{ext}'. Allowed: {sorted(VideoInput.SUPPORTED_EXTENSIONS)}"
            )

        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundMediaError(f"Video file not found at path: {raw_path}")

        return candidate

    # ----------------------------------------------------
    # Real-Time Frame Processing Loop
    # ----------------------------------------------------

    async def _processing_loop(self) -> None:
        """
        Processes frames sequentially:
        Read -> YOLO Detect -> ByteTrack -> Analytics -> Send Result.
        Respects target FPS limits and thread safety.
        """
        logger.info(f"Processing loop started for session={self.session_id}")
        target_interval = 1.0 / max(1.0, self.fps_limit)

        try:
            while self.is_running:
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue

                loop_start = time.perf_counter()

                # 1. Read frame offloaded to thread to keep asyncio event loop responsive
                success, frame, meta = await asyncio.to_thread(self.input_source.read)

                if not success or frame is None:
                    # End of stream (video EOF or camera stream closed)
                    logger.info(f"End of stream reached for session={self.session_id}")
                    await self.stop_stream(reason="End of media stream reached")
                    break

                self.frames_processed += 1
                frame_number = meta.get("frame_number", self.frames_processed)
                timestamp_ms = meta.get("timestamp_ms", round(self.frames_processed * target_interval * 1000.0, 2))

                # 2. YOLO Object Detection
                t_det_start = time.perf_counter()
                det_result = await asyncio.to_thread(
                    self.detector.detect,
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    target_classes=self.target_classes,
                    frame_number=frame_number,
                )
                t_det_end = time.perf_counter()

                # 3. ByteTrack Multi-Object Tracking
                t_track_start = time.perf_counter()
                tracking_result: TrackingResult = await asyncio.to_thread(
                    self.tracker.update,
                    det_result,
                    frame,
                )
                t_track_end = time.perf_counter()

                # 4. Analytics Engine Update
                analytics_snapshot = self.analytics.update(
                    tracking_result,
                    processing_time_ms=(t_track_end - t_det_start) * 1000.0,
                )

                # 5. Optional Annotation Base64 Rendering
                annotated_base64: Optional[str] = None
                if self.annotate:
                    def _render_annotated(f_mat, d_res):
                        ann = self.detector.annotate(f_mat, d_res)
                        _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        return base64.b64encode(buf).decode("utf-8")

                    annotated_base64 = await asyncio.to_thread(_render_annotated, frame, det_result)

                # 6. Assemble Frame Result Payload
                total_step_ms = round((time.perf_counter() - loop_start) * 1000.0, 2)
                effective_fps = round(1000.0 / total_step_ms, 1) if total_step_ms > 0 else self.fps_limit

                result_data = FrameResultData(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    inference_time_ms=round((t_det_end - t_det_start) * 1000.0, 2),
                    tracking_time_ms=round((t_track_end - t_track_start) * 1000.0, 2),
                    total_time_ms=total_step_ms,
                    fps=effective_fps,
                    active_tracks=tracking_result.active_track_count,
                    unique_tracks=tracking_result.cumulative_unique_tracks,
                    detections=[d.to_dict() for d in det_result.detections],
                    tracks=[t.to_dict() for t in tracking_result.objects],
                    analytics=analytics_snapshot.model_dump(),
                    annotated_frame=annotated_base64,
                )

                msg = WebSocketMessage(
                    type=ServerMessageType.FRAME_RESULT,
                    session_id=self.session_id,
                    data=result_data,
                )
                await self.enqueue_message(msg)

                # 7. Frame rate throttling
                elapsed = time.perf_counter() - loop_start
                sleep_duration = max(0.001, target_interval - elapsed)
                await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for session={self.session_id}")
        except Exception as e:
            logger.error(f"Processing loop error on session={self.session_id}: {e}", exc_info=True)
            err_msg = WebSocketMessage(
                type=ServerMessageType.ERROR,
                session_id=self.session_id,
                data=StreamErrorData(
                    code="PROCESSING_ERROR",
                    message="Unexpected error during frame processing.",
                ),
            )
            await self.enqueue_message(err_msg)
        finally:
            self._release_source()

    # ----------------------------------------------------
    # Teardown & Resource Cleanup
    # ----------------------------------------------------

    async def stop_stream(self, reason: str = "Stream stopped") -> None:
        """Stops video/camera streaming and cleans up processing task."""
        if not self.is_running:
            return

        self.is_running = False
        duration = round(time.perf_counter() - self.start_timestamp, 2) if self.start_timestamp > 0 else 0.0

        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        self._release_source()

        # Send stopped notification
        stopped_msg = WebSocketMessage(
            type=ServerMessageType.STREAM_STOPPED,
            session_id=self.session_id,
            data=StreamStoppedData(
                session_id=self.session_id,
                reason=reason,
                frames_processed=self.frames_processed,
                duration_seconds=duration,
            ),
        )
        await self.enqueue_message(stopped_msg)
        logger.info(f"Stream stopped for session={self.session_id}. Reason: {reason}")

    def _release_source(self) -> None:
        """Safely releases VideoCapture / Camera hardware."""
        if self.input_source is not None:
            try:
                self.input_source.release()
            except Exception as e:
                logger.warning(f"Error releasing input source: {e}")
            finally:
                self.input_source = None

    async def close(self) -> None:
        """Completely terminates session, cancels all tasks, and flushes queues."""
        self.is_running = False
        self._release_source()

        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()

        # Put poison pill into sender queue and wait for sender task
        if self.sender_task and not self.sender_task.done():
            try:
                self.send_queue.put_nowait(None)
                await asyncio.wait_for(self.sender_task, timeout=1.0)
            except Exception:
                self.sender_task.cancel()

        # Reset tracker and analytics states
        self.tracker.reset()
        self.analytics.reset()
        logger.info(f"StreamSession closed and cleaned up for session={self.session_id}")
