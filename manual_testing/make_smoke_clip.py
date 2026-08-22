"""
Generates a small synthetic clip for manually exercising the app without
needing real FPV footage. Produces a 200-frame, 64x48 video: the left half
(Original) and right half (Display) both flash bright/dark on a repeating
period, with Display delayed by a fixed number of frames relative to
Original -- every transition pair has the same known latency, which makes it
easy to sanity-check detection/pairing/latency numbers by eye.

Usage:
    python manual_testing/make_smoke_clip.py [output_path]

Default output: manual_testing/smoke_test.mp4 (gitignored -- regenerate
locally rather than committing the binary).

Then, in the app: Open Video -> set ROIs per the printed suggestion below ->
Analyze.
"""

import sys

import cv2
import numpy as np

FRAME_COUNT = 200
W, H = 64, 48
LATENCY = 3    # frames of Display delay relative to Original
DARK, BRIGHT = 20, 220
PERIOD = 20    # frames per full bright/dark cycle
FPS = 30.0


def make_frame(i: int) -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    orig_bright = (i % PERIOD) < PERIOD // 2
    disp_bright = ((i - LATENCY) % PERIOD) < PERIOD // 2
    frame[:, : W // 2] = BRIGHT if orig_bright else DARK
    frame[:, W // 2 :] = BRIGHT if disp_bright else DARK
    return frame


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "manual_testing/smoke_test.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FRAME_COUNT):
        writer.write(make_frame(i))
    writer.release()

    orig_roi = (2, 2, W // 2 - 4, H - 4)
    disp_roi = (W // 2 + 2, 2, W // 2 - 4, H - 4)
    print(f"Wrote {FRAME_COUNT} frames to {out_path}")
    print(f"Suggested ROIs -- Original: {','.join(map(str, orig_roi))}   "
          f"Display: {','.join(map(str, disp_roi))}")
    print(f"Known per-transition latency: {LATENCY} frames = "
          f"{LATENCY / FPS * 1000:.1f} ms at {FPS} fps")


if __name__ == "__main__":
    main()
