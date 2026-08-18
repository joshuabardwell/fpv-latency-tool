"""
BrightnessExtractor tests.

run() is called directly (same thread) instead of start(): the logic under
test is the extraction loop, not Qt's thread machinery, and direct-connected
signals deliver synchronously without an event loop.
"""

import numpy as np

from core.detection import find_falling, find_rising
from core.extractor import BrightnessExtractor
from core.roi import ROI
from tests.conftest import (
    SYNTH_FRAME_COUNT,
    SYNTH_H,
    SYNTH_LATENCY,
    SYNTH_ORIG_FALL,
    SYNTH_ORIG_RISE,
    SYNTH_W,
)

ROI_ORIG = ROI(2, 2, SYNTH_W // 2 - 4, SYNTH_H - 4)
ROI_DISP = ROI(SYNTH_W // 2 + 2, 2, SYNTH_W // 2 - 4, SYNTH_H - 4)


def run_extractor(path, in_point, out_point, qapp):
    ex = BrightnessExtractor(
        path=str(path),
        in_point=in_point,
        out_point=out_point,
        roi_original=ROI_ORIG,
        roi_display=ROI_DISP,
        frame_w=SYNTH_W,
        frame_h=SYNTH_H,
    )
    results, errors, progress = [], [], []
    ex.extraction_done.connect(lambda o, d, f: results.append((o, d, f)))
    ex.error.connect(errors.append)
    ex.progress.connect(lambda done, total: progress.append((done, total)))
    ex.run()
    return results, errors, progress


class TestExtraction:
    def test_full_range(self, synth_video, qapp):
        results, errors, progress = run_extractor(
            synth_video, 0, SYNTH_FRAME_COUNT - 1, qapp
        )
        assert errors == []
        assert len(results) == 1
        orig, disp, first = results[0]
        assert first == 0
        assert len(orig) == len(disp) == SYNTH_FRAME_COUNT
        assert orig.dtype == disp.dtype == np.float32
        # Final progress signal covers the full range.
        assert progress[-1] == (SYNTH_FRAME_COUNT, SYNTH_FRAME_COUNT)

    def test_known_latency_recovered(self, synth_video, qapp):
        results, _, _ = run_extractor(synth_video, 0, SYNTH_FRAME_COUNT - 1, qapp)
        orig, disp, _ = results[0]
        assert find_rising(orig, 50) == [SYNTH_ORIG_RISE]
        assert find_falling(orig, 50) == [SYNTH_ORIG_FALL]
        assert find_rising(disp, 50) == [SYNTH_ORIG_RISE + SYNTH_LATENCY]
        assert find_falling(disp, 50) == [SYNTH_ORIG_FALL + SYNTH_LATENCY]

    def test_subrange_offsets_first_frame(self, synth_video, qapp):
        results, errors, _ = run_extractor(synth_video, 5, 20, qapp)
        assert errors == []
        orig, disp, first = results[0]
        assert first == 5
        assert len(orig) == 16
        # Rise at absolute frame 10 = index 5 within the extracted window.
        assert find_rising(orig, 50) == [SYNTH_ORIG_RISE - 5]

    def test_truncated_file_emits_partial(self, synth_video, qapp):
        """Out point beyond the real file end: partial data, no error."""
        results, errors, _ = run_extractor(
            synth_video, 0, SYNTH_FRAME_COUNT + 19, qapp
        )
        assert errors == []
        assert len(results) == 1
        orig, disp, first = results[0]
        assert first == 0
        assert len(orig) == len(disp) == SYNTH_FRAME_COUNT

    def test_unreadable_path_emits_error(self, tmp_path, qapp):
        results, errors, _ = run_extractor(tmp_path / "missing.mp4", 0, 10, qapp)
        assert results == []
        assert len(errors) == 1
        assert "Cannot open" in errors[0]

    def test_cancel_before_run_emits_nothing(self, synth_video, qapp):
        ex = BrightnessExtractor(
            path=str(synth_video),
            in_point=0,
            out_point=SYNTH_FRAME_COUNT - 1,
            roi_original=ROI_ORIG,
            roi_display=ROI_DISP,
            frame_w=SYNTH_W,
            frame_h=SYNTH_H,
        )
        results, errors = [], []
        ex.extraction_done.connect(lambda *a: results.append(a))
        ex.error.connect(errors.append)
        ex.cancel()
        ex.run()
        assert results == [] and errors == []
