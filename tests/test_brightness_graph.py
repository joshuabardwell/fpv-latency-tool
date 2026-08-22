import numpy as np
import pytest
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols

from core.view_range import MIN_ZOOM_FRAMES
from ui.brightness_graph import BrightnessGraphWidget


def make_graph(qtbot):
    g = BrightnessGraphWidget()
    qtbot.addWidget(g)
    return g


def load_graph(qtbot, n=200, in_point=1000):
    """A graph with real data loaded and shown, so ViewBox pixel geometry
    (needed by wheel-zoom and click-drag pan) is meaningful -- mirrors the
    setup in TestBrightnessGraphPlayhead.test_visible_and_positioned_after_set_data."""
    g = make_graph(qtbot)
    g.resize(400, g.height())
    g.show()
    qtbot.waitExposed(g)
    orig = np.linspace(20, 220, n)
    disp = orig.copy()
    g.set_data(orig, disp, in_point=in_point)
    return g


def _apex_is_up(symbol_key):
    """True if the named pyqtgraph triangle symbol's lone vertex is at the
    top (min y) -- i.e. it renders pointing up."""
    path = Symbols[symbol_key]
    ys = [path.elementAt(i).y for i in range(path.elementCount() - 1)]
    counts = {y: ys.count(y) for y in set(ys)}
    apex_y = min(counts, key=counts.get)
    return apex_y == min(ys)


class TestBrightnessGraphTransitionSymbols:
    def test_rise_markers_point_up_and_fall_markers_point_down(self, qtbot):
        """Locks in the rise/fall symbol swap (was rise="t" rendering down,
        fall="t2" rendering right) against the module docstring's
        triangle-up = rising / triangle-down = falling contract."""
        g = make_graph(qtbot)
        rise_items = (g._sc_rise_orig, g._sc_rise_disp, g._sc_unmatched_rise)
        fall_items = (g._sc_fall_orig, g._sc_fall_disp, g._sc_unmatched_fall)
        for item in rise_items:
            assert _apex_is_up(item.opts["symbol"])
        for item in fall_items:
            assert not _apex_is_up(item.opts["symbol"])


class TestBrightnessGraphMarkerContrast:
    def test_signal_lines_paint_above_transition_markers(self, qtbot):
        g = make_graph(qtbot)
        assert g._line_orig.zValue() > g._sc_rise_orig.zValue()
        assert g._line_orig.zValue() > g._sc_fall_orig.zValue()
        assert g._line_disp.zValue() > g._sc_rise_disp.zValue()
        assert g._line_disp.zValue() > g._sc_fall_disp.zValue()

    def test_marker_tints_are_darker_than_line(self):
        from ui.brightness_graph import _GREEN, _GREEN_MARKER, _AMBER, _AMBER_MARKER
        assert sum(_GREEN_MARKER) < sum(_GREEN)
        assert sum(_AMBER_MARKER) < sum(_AMBER)


class TestUnmatchedNavigation:
    def test_apply_polarity_populates_unmatched_frames_from_real_detection(self, qtbot):
        """orig has a second rising/falling cycle that disp never mirrors, so
        that cycle's orig transitions are unmatched; the first cycle, which
        disp does mirror, is matched and must not appear."""
        orig = np.array(
            [20.0] * 5 + [220.0] * 5 + [20.0] * 5 + [220.0] * 5 + [20.0] * 5,
            dtype=np.float64,
        )  # rising@5, falling@10, rising@15, falling@20
        disp = np.array(
            [20.0] * 5 + [220.0] * 5 + [20.0] * 15,
            dtype=np.float64,
        )  # rising@5, falling@10, then flat -- never mirrors the 2nd cycle
        g = make_graph(qtbot)
        g.set_data(orig, disp, in_point=0)
        assert g._unmatched_frames == [15, 20]
        # The matched pair (frame 5) is a transition but not "unmatched".
        assert 5 not in g._unmatched_frames
        assert g._transition_frames == [5, 10, 15, 20]

    def test_next_unmatched_skips_matched_and_returns_none_past_last(self, qtbot):
        g = make_graph(qtbot)
        g._unmatched_frames = [15, 20]
        assert g.next_unmatched(0) == 15
        assert g.next_unmatched(15) == 20
        assert g.next_unmatched(20) is None

    def test_prev_unmatched_returns_none_before_first(self, qtbot):
        g = make_graph(qtbot)
        g._unmatched_frames = [15, 20]
        assert g.prev_unmatched(25) == 20
        assert g.prev_unmatched(20) == 15
        assert g.prev_unmatched(15) is None

    def test_cleared_by_clear_data(self, qtbot):
        g = make_graph(qtbot)
        g._unmatched_frames = [15, 20]
        g.clear_data()
        assert g._unmatched_frames == []


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


class TestBrightnessGraphDomain:
    def test_domain_set_from_set_data(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        assert (g._range_lo, g._range_hi) == (1000.0, 1199.0)
        assert (g._visible_start, g._visible_end) == (1000.0, 1199.0)

    def test_domain_changed_emitted_on_set_data(self, qtbot):
        g = make_graph(qtbot)
        seen = []
        g.domain_changed.connect(lambda lo, hi: seen.append((lo, hi)))
        orig = np.array([20.0, 220.0] * 5, dtype=np.float64)
        g.set_data(orig, orig.copy(), in_point=50)
        assert seen == [(50.0, 59.0)]

    def test_domain_changed_emitted_on_clear_data(self, qtbot):
        g = load_graph(qtbot)
        seen = []
        g.domain_changed.connect(lambda lo, hi: seen.append((lo, hi)))
        g.clear_data()
        assert seen == [(0.0, -1.0)]


class TestBrightnessGraphSetVisibleRange:
    def test_clamps_to_domain(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(0, 5000)
        assert (g._visible_start, g._visible_end) == (1000.0, 1199.0)

    def test_idempotent_no_signal_on_repeat(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        seen = []
        g.visible_range_changed.connect(lambda s, e: seen.append((s, e)))
        g.set_visible_range(1050, 1150)
        g.set_visible_range(1050, 1150)
        assert seen == [(1050.0, 1150.0)]

    def test_noop_before_data_loaded(self, qtbot):
        g = make_graph(qtbot)
        seen = []
        g.visible_range_changed.connect(lambda s, e: seen.append((s, e)))
        g.set_visible_range(0, 100)
        assert seen == []


class TestBrightnessGraphRecenter:
    def test_set_frame_recenters_preserving_width(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(1050, 1150)  # width 100, zoomed in
        g.set_frame(1180)  # off-center: would-be window [1130, 1230] overhangs the domain
        assert g._visible_end - g._visible_start == 100
        assert (g._visible_start, g._visible_end) == (1099.0, 1199.0)

    def test_recenter_pins_to_domain_edge_without_dead_space(self, qtbot):
        """No dead space at the data boundary: the window's width is
        preserved and its edge pins to the last analyzed frame instead of
        overhanging past it."""
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(1000, 1050)  # width 50
        g.set_frame(1199)  # last frame in the domain
        assert g._visible_end == 1199.0
        assert g._visible_end - g._visible_start == 50

    def test_set_frame_before_data_does_not_touch_range(self, qtbot):
        g = make_graph(qtbot)
        g.set_frame(5)
        assert (g._visible_start, g._visible_end) == (0.0, 0.0)


class TestBrightnessGraphWheelZoom:
    def test_apply_zoom_in_keeps_anchor_fraction_fixed(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)  # domain width 199
        frac_before = (1100 - g._visible_start) / (g._visible_end - g._visible_start)
        g._apply_zoom(0.5, anchor=1100)
        frac_after = (1100 - g._visible_start) / (g._visible_end - g._visible_start)
        assert frac_after == pytest.approx(frac_before, abs=1e-6)
        assert (g._visible_end - g._visible_start) == pytest.approx(99.5, abs=0.01)

    def test_apply_zoom_emits_visible_range_changed(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        seen = []
        g.visible_range_changed.connect(lambda s, e: seen.append((s, e)))
        g._apply_zoom(0.5, anchor=1100)
        assert len(seen) == 1

    def test_apply_zoom_floors_at_min_zoom_width(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g._apply_zoom(0.0001, anchor=1100)
        assert g._visible_end - g._visible_start == MIN_ZOOM_FRAMES

    def test_apply_zoom_noop_before_data_loaded(self, qtbot):
        g = make_graph(qtbot)
        g._apply_zoom(0.5, anchor=0)  # must not raise despite no domain
        assert (g._visible_start, g._visible_end) == (0.0, 0.0)


class TestBrightnessGraphClickDragPan:
    def test_pans_without_changing_width(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(1050, 1150)
        g._pan_drag_start_range = (1050.0, 1150.0)
        g._apply_pan_drag(-40)  # drag left -> reveals later frames
        assert g._visible_end - g._visible_start == pytest.approx(100.0, abs=0.5)
        assert g._visible_start > 1050.0

    def test_clamps_at_domain_edge_without_changing_width(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(1000, 1100)  # already pinned at the domain's left edge
        g._pan_drag_start_range = (1000.0, 1100.0)
        g._apply_pan_drag(500)  # drag hard right, past the left edge
        assert g._visible_start == 1000.0
        assert g._visible_end - g._visible_start == 100.0

    def test_emits_visible_range_changed(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(1050, 1150)
        g._pan_drag_start_range = (1050.0, 1150.0)
        seen = []
        g.visible_range_changed.connect(lambda s, e: seen.append((s, e)))
        g._apply_pan_drag(-30)
        assert len(seen) == 1
