import numpy as np

from ui.brightness_graph import BrightnessGraphWidget


def make_graph(qtbot):
    g = BrightnessGraphWidget()
    qtbot.addWidget(g)
    return g


class TestBrightnessGraphPlayhead:
    def test_playhead_symbol_points_up_and_matches_timeline_size(self):
        """Locks in bug #1 (was pointing down) and #2 (was 16x16, timeline
        is 10x7) directly against the symbol geometry, independent of any
        widget/ViewBox state."""
        from ui.brightness_graph import (
            _PLAYHEAD_SYMBOL, _PLAYHEAD_SYMBOL_SIZE,
            _PLAYHEAD_TRI_BASE_PX, _PLAYHEAD_TRI_H_PX,
        )
        rect = _PLAYHEAD_SYMBOL.boundingRect()
        # Apex (min local y) is above the base (max local y) -- points up.
        assert rect.top() < 0.0 <= rect.bottom()
        # Rendered pixel footprint matches TimelineWidget's playhead triangle.
        assert round(rect.width() * _PLAYHEAD_SYMBOL_SIZE) == _PLAYHEAD_TRI_BASE_PX
        assert round(rect.height() * _PLAYHEAD_SYMBOL_SIZE) == _PLAYHEAD_TRI_H_PX

    def test_hidden_before_data_loaded(self, qtbot):
        g = make_graph(qtbot)
        assert not g._playhead_marker.isVisible()
        assert not g._playhead_stalk.isVisible()

    def test_visible_and_positioned_after_set_data(self, qtbot):
        g = make_graph(qtbot)
        # A real ViewBox pixel geometry is required for the playhead's
        # pixel->data-unit math (ViewBox.viewPixelSize()) to be meaningful --
        # an unshown widget reports a stale/default ViewBox rect.
        g.resize(400, g.height())
        g.show()
        qtbot.waitExposed(g)

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

        # Marker's plotted point is still the triangle's BASE.
        assert marker_y[0] == g._playhead_y0

        # Stalk now starts at the triangle's APEX, strictly above the base --
        # it must not pass back down through the triangle body (bug #3).
        assert stalk_y[0] == g._playhead_apex_y
        assert g._playhead_y1 > stalk_y[0] > g._playhead_y0
        assert stalk_y[1] == g._playhead_y1

        # The triangle's base sits inside the ViewBox's rendered range, with
        # a strict safety margin above the lower edge -- not clipped (bug #4).
        view_ymin, view_ymax = g.getPlotItem().getViewBox().viewRange()[1]
        assert view_ymin < g._playhead_y0 < view_ymax

    def test_hidden_again_after_clear_data(self, qtbot):
        g = make_graph(qtbot)
        orig = np.array([20.0, 220.0], dtype=np.float64)
        disp = np.array([20.0, 220.0], dtype=np.float64)
        g.set_data(orig, disp, in_point=0)
        g.clear_data()
        assert not g._playhead_marker.isVisible()
        assert not g._playhead_stalk.isVisible()
