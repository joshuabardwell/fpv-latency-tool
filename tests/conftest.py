"""
Shared fixtures.

Qt runs on the "offscreen" platform plugin so the suite works headless
(CI runners, no X/Wayland). The synthetic video fixture produces a small
mp4 with a known bright/dark pattern so extractor and video-io tests can
assert exact frame indices and brightness levels.
"""

import os
import sys

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frame layout of the synthetic clip: 40 frames, 64x48.
# Left half of the frame is the "original" screen, right half the "display".
# Original goes dark->bright at frame 10 and bright->dark at frame 25.
# Display mirrors it 3 frames later (13 and 28) — a constant 3-frame latency.
SYNTH_FRAME_COUNT = 40
SYNTH_W, SYNTH_H = 64, 48
SYNTH_ORIG_RISE, SYNTH_ORIG_FALL = 10, 25
SYNTH_LATENCY = 3
SYNTH_DARK, SYNTH_BRIGHT = 20, 220


def _synth_frame(i: int) -> np.ndarray:
    frame = np.zeros((SYNTH_H, SYNTH_W, 3), dtype=np.uint8)
    orig_bright = SYNTH_ORIG_RISE <= i < SYNTH_ORIG_FALL
    disp_bright = (
        SYNTH_ORIG_RISE + SYNTH_LATENCY <= i < SYNTH_ORIG_FALL + SYNTH_LATENCY
    )
    frame[:, : SYNTH_W // 2] = SYNTH_BRIGHT if orig_bright else SYNTH_DARK
    frame[:, SYNTH_W // 2 :] = SYNTH_BRIGHT if disp_bright else SYNTH_DARK
    return frame


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory) -> str:
    """Path to a 40-frame mp4 with known transitions (see module docstring)."""
    path = str(tmp_path_factory.mktemp("video") / "synth.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (SYNTH_W, SYNTH_H)
    )
    assert writer.isOpened()
    for i in range(SYNTH_FRAME_COUNT):
        writer.write(_synth_frame(i))
    writer.release()
    return path
