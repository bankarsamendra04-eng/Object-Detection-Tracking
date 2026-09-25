from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.core.logging import get_logger
from backend.app.services.vision.detector import (
    DetectionEngine,
    DetectionError,
    DetectionResult,
    get_detector,
)

logger = get_logger("vision.input_sources")


# ==========================================
# Media Exceptions Hierarchy
# ==========================================

class MediaSourceError(DetectionError):
    """Base exception for all media source and input pipeline errors."""
    pass


class FileNotFoundMediaError(MediaSourceError):
    """Raised when an input image or video file cannot be found."""
    pass


class UnsupportedFormatError(MediaSourceError):
    """Raised when the input file format/extension is not supported."""
    pass


class CorruptMediaError(MediaSourceError):
    """Raised when media data is corrupted, empty, or unreadable by OpenCV."""
    pass


class DeviceUnavailableError(MediaSourceError):
    """Raised when a camera/webcam hardware index cannot be opened or is busy."""
    pass


class StreamReadError(MediaSourceError):
    """Raised when an open media stream fails during a frame read."""
    pass


# ==========================================
# Abstract InputSource Base Class
# ==========================================

class InputSource(ABC):
    """
    Abstract base class establishing the contract for all frame/media sources
    (ImageInput, VideoInput, WebcamInput).
    """

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Reads the next available frame.
        Returns:
            Tuple: (success_or_has_more: bool, frame: Optional[np.ndarray], metadata: Dict[str, Any])
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases all underlying OpenCV handles, memory buffers, or device streams."""
        pass

    def __enter__(self) -> "InputSource":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# ==========================================
# 1. ImageInput
# ==========================================

class ImageInput(InputSource):
    """
    Handles static image inputs across standard image formats:
    JPG/JPEG, PNG, WEBP, BMP.
    Validates file integrity, parses via OpenCV, and interfaces with DetectionEngine.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, source: Union[str, Path, np.ndarray]):
        self._source = source
        self._consumed = False
        self._frame: Optional[np.ndarray] = None
        self.source_path: Optional[Path] = None
        self.metadata: Dict[str, Any] = {}

        if isinstance(source, (str, Path)):
            self.source_path = Path(source)
            self._validate_and_load_file()
        elif isinstance(source, np.ndarray):
            self._validate_numpy_array(source)
            self._frame = source.copy()
            h, w = self._frame.shape[:2]
            self.metadata = {
                "source_type": "image",
                "width": int(w),
                "height": int(h),
                "channels": int(self._frame.shape[2]) if len(self._frame.shape) > 2 else 1,
                "file_name": "in_memory_array",
                "file_size_bytes": int(self._frame.nbytes),
            }
        else:
            raise UnsupportedFormatError(
                f"Unsupported source type '{type(source).__name__}'. Expected file path (str, Path) or np.ndarray."
            )

    def _validate_and_load_file(self) -> None:
        assert self.source_path is not None
        if not self.source_path.exists():
            raise FileNotFoundMediaError(f"Image file does not exist: {self.source_path}")

        if self.source_path.is_dir():
            raise FileNotFoundMediaError(f"Target path is a directory, not an image file: {self.source_path}")

        ext = self.source_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported image extension '{ext}'. Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        # Check for zero-byte files
        file_size = self.source_path.stat().st_size
        if file_size == 0:
            raise CorruptMediaError(f"Image file is empty (0 bytes): {self.source_path}")

        # Load using OpenCV
        img = cv2.imread(str(self.source_path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise CorruptMediaError(
                f"Failed to decode image from '{self.source_path}'. The file may be corrupt or not a valid image."
            )

        self._frame = img
        h, w = img.shape[:2]
        self.metadata = {
            "source_type": "image",
            "file_name": self.source_path.name,
            "file_path": str(self.source_path.resolve()),
            "file_size_bytes": file_size,
            "width": int(w),
            "height": int(h),
            "channels": int(img.shape[2]) if len(img.shape) > 2 else 1,
        }
        logger.debug(f"Loaded ImageInput: {self.source_path.name} ({w}x{h})")

    def _validate_numpy_array(self, arr: np.ndarray) -> None:
        if arr.size == 0 or len(arr.shape) < 2:
            raise CorruptMediaError(f"NumPy image array is empty or has invalid shape: {arr.shape}")
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            raise CorruptMediaError(f"Invalid NumPy image dimensions: h={h}, w={w}")

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Yields the loaded image once, then marks consumed."""
        if not self._consumed and self._frame is not None:
            self._consumed = True
            return True, self._frame.copy(), self.metadata
        return False, None, {}

    def process(
        self,
        detector: Optional[DetectionEngine] = None,
        annotate: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
    ) -> Tuple[DetectionResult, Optional[np.ndarray]]:
        """
        Executes detection using the provided or default DetectionEngine.
        Optionally generates an annotated image copy without mutating the original.
        """
        if self._frame is None:
            raise CorruptMediaError("No valid image data available to process.")

        engine = detector or get_detector()
        result = engine.detect(
            image=self._frame,
            conf=conf,
            iou=iou,
            target_classes=target_classes,
            frame_number=1,
        )

        annotated_image: Optional[np.ndarray] = None
        if annotate:
            annotated_image = engine.annotate(self._frame, result)

        return result, annotated_image

    def release(self) -> None:
        self._frame = None
        self._consumed = True


# ==========================================
# 2. VideoInput
# ==========================================

class VideoInput(InputSource):
    """
    Handles sequential frame-by-frame video processing across common formats:
    MP4, AVI, MOV, MKV, WEBM.
    Exposes frame numbers, timestamps, FPS, and video metadata.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        source: Union[str, Path],
        stride: int = 1,
        max_frames: Optional[int] = None,
    ):
        self.source_path = Path(source)
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.current_frame_number = 0
        self.frames_yielded = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata: Dict[str, Any] = {}

        self._validate_and_open()

    def _validate_and_open(self) -> None:
        if not self.source_path.exists():
            raise FileNotFoundMediaError(f"Video file does not exist: {self.source_path}")

        if self.source_path.is_dir():
            raise FileNotFoundMediaError(f"Target path is a directory, not a video file: {self.source_path}")

        ext = self.source_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported video extension '{ext}'. Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if self.source_path.stat().st_size == 0:
            raise CorruptMediaError(f"Video file is empty (0 bytes): {self.source_path}")

        self.cap = cv2.VideoCapture(str(self.source_path))
        if not self.cap.isOpened():
            self.release()
            raise CorruptMediaError(f"OpenCV could not open video stream: {self.source_path}")

        # Extract metadata
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle corrupt video headers where dimensions or frame counts are 0
        if w <= 0 or h <= 0:
            self.release()
            raise CorruptMediaError(f"Video has invalid dimensions: {w}x{h}")

        duration_sec = round(total_frames / fps, 2) if fps > 0 and total_frames > 0 else 0.0

        self.metadata = {
            "source_type": "video",
            "file_name": self.source_path.name,
            "file_path": str(self.source_path.resolve()),
            "width": w,
            "height": h,
            "fps": fps if fps > 0 else 30.0,
            "total_frames": total_frames,
            "duration_seconds": duration_sec,
            "stride": self.stride,
        }
        logger.info(
            f"Opened VideoInput: {self.source_path.name} ({w}x{h} @ {fps:.1f} fps, {total_frames} frames)"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Reads the next strided frame from the video stream."""
        if self.cap is None or not self.cap.isOpened():
            return False, None, {}

        if self.max_frames is not None and self.frames_yielded >= self.max_frames:
            return False, None, {}

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # End of stream or corrupt frame encountered
                return False, None, {}

            self.current_frame_number += 1

            # Check stride
            if (self.current_frame_number - 1) % self.stride != 0:
                continue

            self.frames_yielded += 1
            fps = self.metadata.get("fps", 30.0)
            timestamp_ms = round((self.current_frame_number / fps) * 1000.0, 2)

            frame_meta = {
                **self.metadata,
                "frame_number": self.current_frame_number,
                "timestamp_ms": timestamp_ms,
            }
            return True, frame, frame_meta

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, float]]:
        """Yields (frame_number, frame, timestamp_ms) until end of video."""
        while True:
            has_frame, frame, meta = self.read()
            if not has_frame or frame is None:
                break
            yield meta["frame_number"], frame, meta["timestamp_ms"]

    def process_all(
        self,
        detector: Optional[DetectionEngine] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
        max_frames: Optional[int] = None,
    ) -> List[DetectionResult]:
        """
        Processes video frames sequentially through the detection engine.
        Reuses the existing model without reloading.
        """
        engine = detector or get_detector()
        results: List[DetectionResult] = []
        limit = max_frames or self.max_frames

        for frame_num, frame, timestamp_ms in self:
            res = engine.detect(
                image=frame,
                conf=conf,
                iou=iou,
                target_classes=target_classes,
                frame_number=frame_num,
            )
            results.append(res)
            if limit is not None and len(results) >= limit:
                break

        return results

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing VideoCapture: {e}")
            finally:
                self.cap = None


# ==========================================
# 3. WebcamInput
# ==========================================

class WebcamInput(InputSource):
    """
    Handles live camera streams safely with configurable camera index (default 0),
    device availability probing, bounded frame capture, and clean shutdown.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ):
        self.camera_index = camera_index
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_counter = 0
        self.metadata: Dict[str, Any] = {}

        self._validate_and_open()

    def _validate_and_open(self) -> None:
        # On Windows, DirectShow (CAP_DSHOW) provides fast and non-blocking camera discovery
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.release()
            raise DeviceUnavailableError(
                f"Camera index {self.camera_index} is unavailable, disconnected, or currently in use by another application."
            )

        # Apply desired settings if requested
        if self.desired_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        if self.desired_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
        if self.desired_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.metadata = {
            "source_type": "webcam",
            "camera_index": self.camera_index,
            "width": actual_w,
            "height": actual_h,
            "fps": actual_fps if actual_fps > 0 else 30.0,
            "is_opened": True,
        }
        logger.info(
            f"Opened WebcamInput: index={self.camera_index} ({actual_w}x{actual_h} @ {actual_fps:.1f} fps)"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Reads a single frame from the camera stream."""
        if self.cap is None or not self.cap.isOpened():
            return False, None, {}

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None, {}

        self.frame_counter += 1
        frame_meta = {
            **self.metadata,
            "frame_number": self.frame_counter,
        }
        return True, frame, frame_meta

    def capture_frames(
        self,
        count: int,
        detector: Optional[DetectionEngine] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Tuple[np.ndarray, Optional[DetectionResult]]]:
        """
        Safely captures up to `count` frames from the camera, optionally running detection.
        Ensures execution is strictly bounded.
        """
        frames: List[Tuple[np.ndarray, Optional[DetectionResult]]] = []
        engine = detector or (get_detector() if detector is not False else None)

        for _ in range(count):
            success, frame, meta = self.read()
            if not success or frame is None:
                break
            
            res = None
            if engine is not None:
                res = engine.detect(
                    frame,
                    conf=conf,
                    iou=iou,
                    frame_number=meta["frame_number"],
                )
            frames.append((frame, res))

        return frames

    def release(self) -> None:
        """Safely and idempotently closes the camera handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera index {self.camera_index}: {e}")
            finally:
                self.cap = None
        self.metadata["is_opened"] = False


# ==========================================
# Device Probing Utilities
# ==========================================

def probe_camera_availability(camera_index: int = 0) -> Dict[str, Any]:
    """
    Safely probes whether a camera device is available without holding the device open.
    """
    try:
        with WebcamInput(camera_index=camera_index) as cam:
            ret, frame, meta = cam.read()
            return {
                "available": True,
                "camera_index": camera_index,
                "readable": bool(ret and frame is not None),
                "resolution": f"{meta.get('width', 0)}x{meta.get('height', 0)}",
                "fps": meta.get("fps", 0),
            }
    except DeviceUnavailableError as e:
        return {
            "available": False,
            "camera_index": camera_index,
            "readable": False,
            "error": str(e),
        }
    except Exception as e:
        return {
            "available": False,
            "camera_index": camera_index,
            "readable": False,
            "error": f"Unexpected error probing camera: {str(e)}",
        }
