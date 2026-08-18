import numpy as np

from core.roi import ROI


def gradient_frame():
    """100x200 BGR frame: left half black, right half white."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, 100:] = 255
    return frame


class TestClipped:
    def test_inside_frame_unchanged(self):
        roi = ROI(10, 10, 50, 40)
        assert roi.clipped(200, 100) == roi

    def test_negative_origin_clamped(self):
        c = ROI(-5, -5, 50, 40).clipped(200, 100)
        assert (c.x, c.y) == (0, 0)

    def test_overhanging_size_clamped(self):
        c = ROI(180, 90, 50, 40).clipped(200, 100)
        assert c.x + c.width <= 200
        assert c.y + c.height <= 100

    def test_far_outside_yields_one_px_sliver(self):
        c = ROI(500, 500, 10, 10).clipped(200, 100)
        assert (c.width, c.height) == (1, 1)
        assert 0 <= c.x < 200 and 0 <= c.y < 100


class TestIsValid:
    def test_valid(self):
        assert ROI(0, 0, 10, 10).is_valid()

    def test_degenerate(self):
        assert not ROI(0, 0, 1, 10).is_valid()
        assert not ROI(0, 0, 10, 1).is_valid()
        assert not ROI(0, 0, 0, 0).is_valid()


class TestMeanBrightness:
    def test_black_region(self):
        assert ROI(0, 0, 100, 100).mean_brightness(gradient_frame()) == 0.0

    def test_white_region(self):
        assert ROI(100, 0, 100, 100).mean_brightness(gradient_frame()) == 255.0

    def test_half_and_half(self):
        b = ROI(50, 0, 100, 100).mean_brightness(gradient_frame())
        assert abs(b - 127.5) < 1.0

    def test_empty_region_returns_zero(self):
        assert ROI(0, 0, 0, 0).mean_brightness(gradient_frame()) == 0.0
