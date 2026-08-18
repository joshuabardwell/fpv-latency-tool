"""
Video I/O module.

Wraps cv2.VideoCapture to provide frame-accurate seeking and metadata
access. Frame-index-based seeking is used throughout instead of
time-based seeking, since time-based seeks in OpenCV/ffmpeg are not
reliably frame-accurate, especially on high-fps footage.

IMPORTANT: container-reported FPS (cv2.CAP_PROP_FPS) can be wrong,
especially for slow-motion footage from phones/GoPros where the
container metadata does not reflect the true capture rate. Always
let the user confirm/override this in the UI before trusting any
latency numbers computed from it.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: Path
    frame_count: int
    fps_reported: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps_reported <= 0:
            return 0.0
        return self.frame_count / self.fps_reported


class VideoReader:
    """Frame-accurate random-access reader over a video file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"OpenCV could not open video file: {self.path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps_reported = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # User-confirmed fps overrides the (possibly wrong) container value.
        # Defaults to the reported value until the user changes it in the UI.
        self.fps_effective = self._fps_reported

        self._next_index = -1  # frame the capture will decode next; -1 = unknown

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            path=self.path,
            frame_count=self._frame_count,
            fps_reported=self._fps_reported,
            width=self._width,
            height=self._height,
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read_frame(self, index: int) -> np.ndarray:
        """
        Return the BGR frame at the given index (0-based).

        Seeking via CAP_PROP_POS_FRAMES and calling grab()/retrieve()
        once, rather than reading sequentially from zero every time,
        so scrubbing stays responsive on large files. This is still
        an approximate seek on some codecs (esp. long-GOP H.264) --
        if exact-frame correctness ever becomes suspect, fall back to
        sequential decoding from the nearest preceding keyframe.
        """
        if index < 0 or index >= self._frame_count:
            raise IndexError(
                f"Frame index {index} out of range (0..{self._frame_count - 1})"
            )

        if index != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame = self._cap.read()
        if not ok:
            raise IOError(f"Failed to decode frame {index} from {self.path}")

        self._next_index = index + 1  # cap auto-advances after read()
        return frame

    def frame_to_timestamp(self, index: int) -> float:
        """Seconds from start of file, using the user-confirmed fps."""
        if self.fps_effective <= 0:
            return 0.0
        return index / self.fps_effective

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
