import numpy as np

from ui.brightness_graph import BrightnessGraphWidget


def make_graph(qtbot):
    g = BrightnessGraphWidget()
    qtbot.addWidget(g)
    return g


class TestBrightnessGraphPlayhead:
    def test_hidden_before_data_loaded(self, qtbot):
        g = make_graph(qtbot)
        assert not g._playhead_marker.isVisible()
        assert not g._playhead_stalk.isVisible()

    def test_visible_and_positioned_after_set_data(self, qtbot):
        g = make_graph(qtbot)
        orig = np.array([20.0, 20.0, 220.0, 220.0], dtype=np.float64)
        disp = np.array([20.0, 20.0, 20.0, 220.0], dtype=np.float64)
        g.set_data(orig, disp, in_point=100)

        assert g._playhead_marker.isVisible()
        assert g._playhead_stalk.isVisible()

        g.set_frame(102)
        marker_x, marker_y = g._playhead_marker.getData()
        stalk_x, stalk_y = g._playhead_stalk.getData()
        assert list(marker_x) == [102]
        assert list(stalk_x) == [102, 102]
        assert marker_y[0] == g._playhead_y0
        assert stalk_y[0] == g._playhead_y0
        assert stalk_y[1] == g._playhead_y1
        assert g._playhead_y1 > g._playhead_y0

    def test_hidden_again_after_clear_data(self, qtbot):
        g = make_graph(qtbot)
        orig = np.array([20.0, 220.0], dtype=np.float64)
        disp = np.array([20.0, 220.0], dtype=np.float64)
        g.set_data(orig, disp, in_point=0)
        g.clear_data()
        assert not g._playhead_marker.isVisible()
        assert not g._playhead_stalk.isVisible()
