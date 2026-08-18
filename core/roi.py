from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ROI:
    """Rectangular region of interest in original-frame pixel coordinates.

    Note the validity boundary: is_valid() requires width/height > 1 (gates
    accepting a user-drawn rectangle), while clipped() floors dimensions at 1
    so a valid ROI hanging off the frame edge still yields a usable region.
    A clipped result may therefore be a 1-px sliver even though the unclipped
    ROI passed is_valid()."""

    x: int
    y: int
    width: int
    height: int

    def clipped(self, frame_w: int, frame_h: int) -> "ROI":
        x = max(0, min(self.x, frame_w - 1))
        y = max(0, min(self.y, frame_h - 1))
        w = max(1, min(self.width, frame_w - x))
        h = max(1, min(self.height, frame_h - y))
        return ROI(x, y, w, h)

    def mean_brightness(self, frame: np.ndarray) -> float:
        """Mean grayscale brightness (0–255) inside this ROI.

        Crops first, then converts only the region — converting the whole
        frame per call is wasted work when this runs on every scrubbed frame."""
        region = frame[self.y : self.y + self.height, self.x : self.x + self.width]
        if region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_valid(self) -> bool:
        return self.width > 1 and self.height > 1
