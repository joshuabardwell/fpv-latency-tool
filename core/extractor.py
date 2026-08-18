"""
BrightnessExtractor — QThread worker that sequentially reads every frame in an
in/out range, computes mean grayscale brightness for two ROIs, and stores the
results as a pair of float32 NumPy arrays.

Sequential reads are faster than repeated random seeks, so this is done in one
forward pass rather than re-seeking for each frame.

The single CAP_PROP_POS_FRAMES seek to the in point has the same caveat as
VideoReader (see core/video_io.py): on long-GOP codecs it can land off by a
few frames. Latency deltas are unaffected (both ROIs are sampled from the
same frames) but absolute frame labels may shift.
"""

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from core.roi import ROI


class BrightnessExtractor(QThread):
    # (frames_done, frames_total) — emitted every 30 frames and at completion
    progress = pyqtSignal(int, int)

    # (orig_array, disp_array, first_frame) — float32 arrays, one value per frame.
    # May hold fewer frames than requested if the file ends early (container
    # frame counts are unreliable); the receiver compares against what it asked
    # for. Deliberately NOT named "finished": that would shadow the built-in
    # QThread.finished, which the GUI relies on to know the thread has exited.
    extraction_done = pyqtSignal(object, object, int)

    # human-readable error message
    error = pyqtSignal(str)

    def __init__(
        self,
        path: str,
        in_point: int,
        out_point: int,
        roi_original: ROI,
        roi_display: ROI,
        frame_w: int,
        frame_h: int,
    ) -> None:
        super().__init__()
        self._path = path
        self._in_point = in_point
        self._out_point = out_point
        self._roi_orig = roi_original.clipped(frame_w, frame_h)
        self._roi_disp = roi_display.clipped(frame_w, frame_h)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {self._path}")
            return

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._in_point)
            count = self._out_point - self._in_point + 1
            orig = np.empty(count, dtype=np.float32)
            disp = np.empty(count, dtype=np.float32)

            ro = self._roi_orig
            rd = self._roi_disp

            for i in range(count):
                if self._cancelled:
                    return

                ok, frame = cap.read()
                if not ok:
                    # Container frame counts routinely overestimate; salvage
                    # what was extracted instead of discarding the whole run.
                    if i > 0:
                        self.progress.emit(i, count)
                        self.extraction_done.emit(
                            orig[:i].copy(), disp[:i].copy(), self._in_point
                        )
                    else:
                        self.error.emit(
                            f"Frame read failed at frame {self._in_point + i}"
                        )
                    return

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                orig[i] = float(np.mean(gray[ro.y : ro.y + ro.height, ro.x : ro.x + ro.width]))
                disp[i] = float(np.mean(gray[rd.y : rd.y + rd.height, rd.x : rd.x + rd.width]))

                if (i + 1) % 30 == 0 or i == count - 1:
                    self.progress.emit(i + 1, count)

            if not self._cancelled:
                self.extraction_done.emit(orig, disp, self._in_point)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            cap.release()
