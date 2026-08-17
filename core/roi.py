from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ROI:
    """Rectangular region of interest in original-frame pixel coordinates."""

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
        """Mean grayscale brightness (0–255) inside this ROI."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        region = gray[self.y : self.y + self.height, self.x : self.x + self.width]
        return float(np.mean(region)) if region.size > 0 else 0.0

    def is_valid(self) -> bool:
        return self.width > 1 and self.height > 1
