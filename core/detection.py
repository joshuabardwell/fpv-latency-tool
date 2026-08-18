"""
Derivative-based transition detection on a brightness trace.

Pure NumPy, no Qt — usable from the GUI and from tests alike.

A transition exists where a single frame-to-frame brightness step exceeds
the delta threshold. Runs of consecutive over-threshold steps are collapsed
to the frame of steepest change. Note the inherent limit: a slow fade spread
over many frames, none of which individually crosses delta, is not detected.
"""

import numpy as np


def collapse_peaks(candidates: np.ndarray, diff: np.ndarray, sign: int) -> list[int]:
    """Group consecutive candidate diff-indices; return frame of steepest change per group.
    diff[i] = data[i+1]-data[i], so transition frame = diff_index + 1."""
    if len(candidates) == 0:
        return []
    result = []
    group = [candidates[0]]
    for i in range(1, len(candidates)):
        if candidates[i] == candidates[i - 1] + 1:
            group.append(candidates[i])
        else:
            best = group[int(np.argmax(sign * diff[group]))]
            result.append(int(best) + 1)
            group = [candidates[i]]
    best = group[int(np.argmax(sign * diff[group]))]
    result.append(int(best) + 1)
    return result


def find_rising(data: np.ndarray, delta: float) -> list[int]:
    """Indices of dark→light transitions (frame where the signal is first bright)."""
    if len(data) < 2:
        return []
    diff = np.diff(data.astype(np.float64))
    candidates = np.where(diff >= delta)[0]
    return collapse_peaks(candidates, diff, sign=1)


def find_falling(data: np.ndarray, delta: float) -> list[int]:
    """Indices of light→dark transitions (frame where the signal is first dark)."""
    if len(data) < 2:
        return []
    diff = np.diff(data.astype(np.float64))
    candidates = np.where(diff <= -delta)[0]
    return collapse_peaks(candidates, diff, sign=-1)


def apply_min_spacing(frames: list[int], min_spacing: int) -> list[int]:
    """Drop transitions closer than min_spacing frames to the previously kept one."""
    if min_spacing <= 1 or not frames:
        return frames
    result = [frames[0]]
    for f in frames[1:]:
        if f - result[-1] >= min_spacing:
            result.append(f)
    return result
