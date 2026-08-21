import numpy as np

from ui.brightness_graph import BrightnessGraphWidget


def make_graph(qtbot):
    g = BrightnessGraphWidget()
    qtbot.addWidget(g)
    return g


class TestOrigPeriod:
    def test_none_with_fewer_than_two_transitions(self, qtbot):
        g = make_graph(qtbot)
        orig = np.array([20.0, 20.0, 220.0, 220.0], dtype=np.float64)
        disp = np.array([20.0, 20.0, 20.0, 220.0], dtype=np.float64)
        g.set_data(orig, disp, in_point=0)
        assert g.get_orig_period_frames("both") is None

    def test_mean_period_across_multiple_rising_transitions(self, qtbot):
        g = make_graph(qtbot)
        # Two full cycles: rising edges land 10 frames apart (frames 5, 15).
        orig = np.array([20.0] * 5 + [220.0] * 5 + [20.0] * 5 + [220.0] * 5, dtype=np.float64)
        disp = orig.copy()
        g.set_data(orig, disp, in_point=0)
        assert g.get_orig_period_frames("both") == 10.0

    def test_falling_only_when_polarity_falling(self, qtbot):
        g = make_graph(qtbot)
        # Irregular timing so rising-period != falling-period, proving the
        # "falling" branch reads the fall list rather than falling back to
        # the (different-valued) rise list.
        # Rising transitions at frames 5, 15, 30 (mean gap 12.5).
        # Falling transitions at frames 10, 20 (mean gap 10.0).
        orig = np.array(
            [20.0] * 5 + [220.0] * 5 + [20.0] * 5 + [220.0] * 5 + [20.0] * 10 + [220.0] * 5,
            dtype=np.float64,
        )
        disp = orig.copy()
        g.set_data(orig, disp, in_point=0)
        assert g.get_orig_period_frames("both") == 12.5
        assert g.get_orig_period_frames("falling") == 10.0
