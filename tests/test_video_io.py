import numpy as np
import pytest

from core.video_io import VideoReader
from tests.conftest import (
    SYNTH_FRAME_COUNT,
    SYNTH_H,
    SYNTH_ORIG_RISE,
    SYNTH_W,
)


class TestVideoReader:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nope.mp4")

    def test_metadata(self, synth_video):
        with VideoReader(synth_video) as r:
            meta = r.metadata
            assert meta.frame_count == SYNTH_FRAME_COUNT
            assert meta.width == SYNTH_W
            assert meta.height == SYNTH_H
            assert meta.fps_reported == pytest.approx(30.0)
            assert meta.duration_seconds == pytest.approx(
                SYNTH_FRAME_COUNT / 30.0
            )

    def test_read_frame_shape(self, synth_video):
        with VideoReader(synth_video) as r:
            frame = r.read_frame(0)
            assert frame.shape == (SYNTH_H, SYNTH_W, 3)

    def test_out_of_range_raises(self, synth_video):
        with VideoReader(synth_video) as r:
            with pytest.raises(IndexError):
                r.read_frame(-1)
            with pytest.raises(IndexError):
                r.read_frame(SYNTH_FRAME_COUNT)

    def test_random_access_content(self, synth_video):
        """Frame just before the rise is dark on the left, just after bright."""
        with VideoReader(synth_video) as r:
            left = slice(0, SYNTH_W // 2)
            before = r.read_frame(SYNTH_ORIG_RISE - 1)
            after = r.read_frame(SYNTH_ORIG_RISE)
            assert float(np.mean(before[:, left])) < 80
            assert float(np.mean(after[:, left])) > 150

    def test_sequential_then_repeat(self, synth_video):
        """Sequential reads and a repeated read return consistent content."""
        with VideoReader(synth_video) as r:
            a = r.read_frame(5)
            b = r.read_frame(6)
            a_again = r.read_frame(5)
            assert np.array_equal(a, a_again)
            assert b.shape == a.shape

    def test_frame_to_timestamp_uses_effective_fps(self, synth_video):
        with VideoReader(synth_video) as r:
            assert r.frame_to_timestamp(30) == pytest.approx(1.0)
            r.fps_effective = 60.0
            assert r.frame_to_timestamp(30) == pytest.approx(0.5)
            r.fps_effective = 0.0
            assert r.frame_to_timestamp(30) == 0.0
