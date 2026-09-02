"""Video stream handler supporting multiple input sources."""

import time
import logging
import threading
from typing import Optional, Callable, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Thread-safe video stream handler.

    Supports:
        - USB Webcam (device index)
        - RTSP streams (rtsp://...)
        - RTMP streams (rtmp://...)
        - Local video files (.mp4, .avi, etc.)
        - HTTP video streams
    """

    def __init__(
        self,
        source: str = "0",
        buffer_size: int = 3,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
    ):
        self.source = source
        self.buffer_size = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = time.time()
        self._source_type = self._detect_source_type(source)
        self._native_fps = None

    @staticmethod
    def _detect_source_type(source: str) -> str:
        """Detect the type of video source."""
        source_lower = source.lower().strip()
        if source_lower.startswith("rtsp://") or source_lower.startswith("rtsp\\:"):
            return "rtsp"
        elif source_lower.startswith("rtmp://") or source_lower.startswith("rtmp\\:"):
            return "rtmp"
        elif source_lower.startswith("http://") or source_lower.startswith("https://"):
            return "http"
        elif source_lower.isdigit():
            return "webcam"
        elif any(source_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]):
            return "file"
        else:
            return "file"

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    def _open_source(self) -> bool:
        """Open the video source."""
        if self._cap is not None:
            self._cap.release()

        if self._source_type == "webcam":
            idx = int(self.source)
            self._cap = cv2.VideoCapture(idx)
        elif self._source_type in ("rtsp", "rtmp", "http"):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or None
        logger.info(f"Video source opened: {self.source} (type={self._source_type})")
        return True

    def _read_loop(self):
        """Background thread for reading frames."""
        frame_interval = (1.0 / self._native_fps) if (
            self._source_type == "file" and self._native_fps
        ) else 0.0
        last_read = time.time()
        while self._running and self._cap is not None:
            if frame_interval:
                wait = frame_interval - (time.time() - last_read)
                if wait > 0:
                    time.sleep(wait)
            last_read = time.time()

            ret, frame = self._cap.read()
            if not ret:
                if self._source_type in ("rtsp", "rtmp", "http"):
                    logger.warning("Stream lost, attempting reconnect...")
                    time.sleep(self.reconnect_delay)
                    for attempt in range(self.reconnect_attempts):
                        if self._open_source():
                            logger.info(f"Reconnected on attempt {attempt+1}")
                            break
                        time.sleep(self.reconnect_delay)
                    else:
                        logger.error("Max reconnect attempts reached")
                        self._running = False
                    continue
                elif self._source_type == "file":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_read = time.time()
                    continue
                else:
                    logger.error("Failed to read frame")
                    continue

            with self._lock:
                self._frame = frame

            self._frame_count += 1
            current_time = time.time()
            elapsed = current_time - self._last_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_time = current_time

    def start(self) -> bool:
        """Start the video stream."""
        if self._running:
            return True

        if not self._open_source():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("Video stream started")
        return True

    def stop(self):
        """Stop the video stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Video stream stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_frame_dimensions(self) -> Optional[Tuple[int, int]]:
        """Get (width, height) of the video source."""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return None

    def __del__(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
