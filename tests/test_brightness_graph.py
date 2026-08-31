import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols

from core.manual import EditTarget, ManualEdit
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


def load_graph_with_pairs(qtbot):
    """Two matched pairs (rising orig@5/disp@7, falling orig@10/disp@12) and
    two unmatched orig transitions (rising@15, falling@20) -- disp never
    mirrors the second cycle. Shown+resized so ViewBox pixel geometry is
    meaningful for hover hit-testing."""
    g = make_graph(qtbot)
    g.resize(400, g.height())
    g.show()
    qtbot.waitExposed(g)
    orig = np.array(
        [20.0] * 5 + [220.0] * 5 + [20.0] * 5 + [220.0] * 5 + [20.0] * 5,
        dtype=np.float64,
    )
    disp = np.array(
        [20.0] * 7 + [220.0] * 5 + [20.0] * 13,
        dtype=np.float64,
    )
    g.set_data(orig, disp, in_point=0)
    return g


def _marker_widget_pos(g, frame, side):
    from PyQt6.QtCore import QPointF
    data = g._orig_data if side == "orig" else g._disp_data
    y = float(data[frame - g._in_point])
    return g.mapFromScene(g._vb.mapViewToScene(QPointF(frame, y)))


class TestMarkerHighlight:
    def test_pairs_and_unmatched_detected_as_expected(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        assert set(g._frame_to_pair.keys()) == {5, 7, 10, 12}
        assert g._frame_to_pair[5].disp_frame == 7
        assert g._frame_to_pair[10].disp_frame == 12
        assert g._unmatched_frames == [15, 20]

    def test_resolve_highlight_pair_hover_wins_over_playhead(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g._current_frame = 10          # matched falling pair
        g._hover_matched_frame = 5     # different matched rising pair
        assert g._resolve_highlight_pair().orig_frame == 5

    def test_resolve_highlight_pair_falls_back_to_playhead(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g._current_frame = 10
        g._hover_matched_frame = None
        assert g._resolve_highlight_pair().orig_frame == 10

    def test_resolve_highlight_pair_none_when_neither_matched(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g._current_frame = 15  # unmatched
        g._hover_matched_frame = None
        assert g._resolve_highlight_pair() is None

    def test_update_marker_highlight_sets_both_items(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g._current_frame = 5
        g._update_marker_highlight()
        xs, ys = g._sc_highlight.getData()
        assert sorted(xs) == [5, 7]
        cxs, cys = g._connector_highlight.getData()
        assert sorted(cxs) == [5, 7]
        assert cys[0] == cys[1] == g._connector_y_level

    def test_update_marker_highlight_clears_when_no_pair(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g._current_frame = 5
        g._update_marker_highlight()
        g._current_frame = 15  # unmatched
        g._update_marker_highlight()
        xs, _ = g._sc_highlight.getData()
        assert len(xs) == 0
        cxs, _ = g._connector_highlight.getData()
        assert cxs is None

    def test_set_frame_on_matched_frame_highlights_it(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        xs, _ = g._sc_highlight.getData()
        assert sorted(xs) == [5, 7]

    def test_set_frame_on_unmatched_frame_clears_highlight(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        g.set_frame(15)
        xs, _ = g._sc_highlight.getData()
        assert len(xs) == 0

    def test_hover_hit_test_finds_matched_marker(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 5, "orig")
        assert g._hover_hit_test(pos) == 5

    def test_hover_hit_test_is_x_only(self, qtbot):
        """Regression: the real cursor is hidden while the hover line shows
        (see _update_cursor_and_line), so the user can't see or aim by
        vertical position -- hit-testing must not require it."""
        g = load_graph_with_pairs(qtbot)
        marker_pos = _marker_widget_pos(g, 5, "orig")
        far_y_pos = QPoint(marker_pos.x(), marker_pos.y() + 1000)
        assert g._hover_hit_test(far_y_pos) == 5

    def test_hover_hit_test_ignores_unmatched_marker(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 15, "orig")  # unmatched -- not in _frame_to_pair
        assert g._hover_hit_test(pos) is None

    def test_hover_hit_test_none_far_from_anything(self, qtbot):
        from PyQt6.QtCore import QPointF
        g = load_graph_with_pairs(qtbot)
        assert g._hover_hit_test(QPointF(-1000, -1000)) is None

    def test_update_hover_sets_hover_state_and_highlight(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 5, "orig")
        g._update_hover(pos)
        assert g._hover_matched_frame == 5
        xs, _ = g._sc_highlight.getData()
        assert sorted(xs) == [5, 7]

    def test_leave_event_clears_hover(self, qtbot):
        from PyQt6.QtCore import QEvent
        g = load_graph_with_pairs(qtbot)
        g._hover_matched_frame = 5
        g._update_marker_highlight()
        g.leaveEvent(QEvent(QEvent.Type.Leave))
        assert g._hover_matched_frame is None
        xs, _ = g._sc_highlight.getData()
        assert len(xs) == 0

    def test_polarity_change_clears_stale_highlight(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)  # highlights the rising pair
        xs, _ = g._sc_highlight.getData()
        assert len(xs) == 2
        g.set_polarity("falling")  # rising pair no longer active
        xs, _ = g._sc_highlight.getData()
        assert len(xs) == 0

    def test_clear_data_resets_highlight_state(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        g.clear_data()
        assert g._frame_to_pair == {}
        assert g._current_frame is None
        assert g._hover_matched_frame is None
        xs, _ = g._sc_highlight.getData()
        assert len(xs) == 0


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


class TestExcludedPairStyling:
    def test_set_excluded_pairs_mutes_marker_color(self, qtbot):
        from ui.brightness_graph import _MUTED_MARKER

        g = load_graph_with_pairs(qtbot)
        g.set_excluded_pairs({0}, set())  # exclude the single rising pair (orig=5, disp=7)

        assert g._sc_rise_orig.data["brush"][0].color().getRgb()[:3] == _MUTED_MARKER
        assert g._sc_rise_disp.data["brush"][0].color().getRgb()[:3] == _MUTED_MARKER
        # Falling pair wasn't excluded -- its markers keep the construction
        # default (no per-point brush override).
        assert g._sc_fall_orig.data["brush"][0] is None
        assert g._sc_fall_disp.data["brush"][0] is None

    def test_set_excluded_pairs_excluded_connector_uses_muted_pen(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_excluded_pairs({0}, set())

        exc_x, _ = g._connector_excluded.getData()
        normal_x, _ = g._pair_connectors.getData()
        assert 5.0 in exc_x and 7.0 in exc_x
        assert 5.0 not in normal_x and 7.0 not in normal_x
        assert 10.0 in normal_x and 12.0 in normal_x  # falling pair stays on the normal connector

    def test_set_excluded_pairs_does_not_emit_pairs_updated(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        calls = []
        g.pairs_updated.connect(lambda: calls.append(1))
        g.set_excluded_pairs({0}, set())
        assert calls == []

    def test_redetect_resets_marker_styling(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_excluded_pairs({0}, set())
        assert g._sc_rise_orig.data["brush"][0] is not None

        g.set_delta(g._delta)  # forces a redetect even with an unchanged value
        assert g._sc_rise_orig.data["brush"][0] is None
        exc_x, _ = g._connector_excluded.getData()
        assert not exc_x  # None or empty, depending on pyqtgraph's internal state


class TestPlayheadPairSignal:
    def test_set_frame_on_matched_frame_emits_the_pair(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        seen = []
        g.playhead_pair_changed.connect(lambda p: seen.append(p))
        g.set_frame(5)  # rising pair, orig side
        assert len(seen) == 1
        assert seen[0].orig_frame == 5 and seen[0].disp_frame == 7

    def test_set_frame_off_a_match_emits_none(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        seen = []
        g.playhead_pair_changed.connect(lambda p: seen.append(p))
        g.set_frame(15)  # unmatched
        assert seen == [None]

    def test_set_frame_does_not_re_emit_for_the_same_pair(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        seen = []
        g.playhead_pair_changed.connect(lambda p: seen.append(p))
        g.set_frame(7)  # disp side of the same rising pair
        assert seen == []

    def test_hover_alone_does_not_emit_playhead_pair_changed(self, qtbot):
        """Regression: the table highlight is playhead-only by design --
        hovering a matched marker must not touch it, even though the
        existing graph ring highlight (_resolve_highlight_pair) does react
        to hover."""
        g = load_graph_with_pairs(qtbot)
        seen = []
        g.playhead_pair_changed.connect(lambda p: seen.append(p))
        g._hover_matched_frame = 5
        g._update_marker_highlight()
        assert seen == []

    def test_redetect_reemits_when_current_frame_match_changes(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        seen = []
        g.playhead_pair_changed.connect(lambda p: seen.append(p))
        g.set_delta(250)  # far above the 200-unit brightness swing -> 0 pairs
        assert seen == [None]

    def test_clear_data_emits_none_if_something_was_highlighted(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_frame(5)
        seen = []
        g.playhead_pair_changed.connect(lambda p: seen.append(p))
        g.clear_data()
        assert seen == [None]


class TestHitTestAnyMarker:
    def test_finds_matched_marker(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 5, "orig")
        assert g._hit_test_any_marker(pos) == 5

    def test_finds_unmatched_marker(self, qtbot):
        """Regression: _hover_hit_test (matched-only) would return None here
        -- _hit_test_any_marker must find unmatched markers too."""
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 15, "orig")
        assert g._hover_hit_test(pos) is None
        assert g._hit_test_any_marker(pos) == 15

    def test_matched_marker_is_x_only(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        marker_pos = _marker_widget_pos(g, 5, "orig")
        far_y_pos = QPoint(marker_pos.x(), marker_pos.y() + 1000)
        assert g._hit_test_any_marker(far_y_pos) == 5

    def test_unmatched_marker_is_x_only(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        marker_pos = _marker_widget_pos(g, 15, "orig")
        far_y_pos = QPoint(marker_pos.x(), marker_pos.y() + 1000)
        assert g._hit_test_any_marker(far_y_pos) == 15

    def test_respects_live_polarity(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        g.set_polarity("falling")  # hides the rising marker @5
        pos = _marker_widget_pos(g, 5, "orig")
        assert g._hit_test_any_marker(pos) is None

    def test_none_far_from_anything(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        assert g._hit_test_any_marker(QPoint(-1000, -1000)) is None


def _mouse_event(event_type, pos, button=Qt.MouseButton.LeftButton, buttons=None):
    """Directly-constructed QMouseEvent for calling BrightnessGraphWidget's
    mousePressEvent/mouseMoveEvent/mouseReleaseEvent overrides as plain
    method calls -- QGraphicsView routes real/qtbot-synthesized mouse events
    through its viewport child widget, not the view itself, so simulating
    via qtbot.mousePress/mouseMove/mouseRelease(g, ...) never reaches these
    overrides. Every other event-adjacent test in this file (e.g.
    test_leave_event_clears_hover) already calls the handler directly for
    the same reason."""
    if buttons is None:
        buttons = button if event_type == QEvent.Type.MouseButtonPress else Qt.MouseButton.NoButton
    return QMouseEvent(event_type, QPointF(pos), button, buttons, Qt.KeyboardModifier.NoModifier)


def _press(g, pos):
    g.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, pos))


def _move(g, pos):
    g.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, pos, buttons=Qt.MouseButton.NoButton))


def _release(g, pos):
    g.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, pos))


class TestClickToSeek:
    def test_plain_click_emits_the_raw_clicked_frame(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 3, "orig")  # no marker near frame 3
        seen = []
        g.frame_clicked.connect(lambda f: seen.append(f))
        _press(g, pos)
        _release(g, pos)
        assert seen == [3]

    def test_click_snaps_to_hovered_marker_over_raw_click_position(self, qtbot):
        """The click lands at frame 3's pixel, but a marker was hovered
        (captured at press time) -- the snap must win over the raw position."""
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 3, "orig")
        g._hover_any_marker_frame = 15  # simulate having hovered marker 15 just before pressing
        seen = []
        g.frame_clicked.connect(lambda f: seen.append(f))
        _press(g, pos)
        _release(g, pos)
        assert seen == [15]

    def test_hovering_then_clicking_a_marker_snaps_to_it(self, qtbot):
        """End-to-end (real hover, not injected state): move onto a matched
        marker, then click without moving further."""
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 5, "orig")
        seen = []
        g.frame_clicked.connect(lambda f: seen.append(f))
        _move(g, pos)
        _press(g, pos)
        _release(g, pos)
        assert seen == [5]

    def test_drag_beyond_threshold_pans_and_does_not_emit(self, qtbot):
        g = load_graph(qtbot, n=200, in_point=1000)
        g.set_visible_range(1050, 1150)
        before = (g._visible_start, g._visible_end)
        seen = []
        g.frame_clicked.connect(lambda f: seen.append(f))
        start = QPoint(50, 50)
        end = QPoint(80, 50)  # 30px net displacement, above the 4px threshold
        _press(g, start)
        _move(g, end)
        _release(g, end)
        assert seen == []
        assert (g._visible_start, g._visible_end) != before

    def test_movement_below_threshold_still_counts_as_a_click(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 3, "orig")
        nearby = QPoint(pos.x() + 2, pos.y())  # 2px, below the 4px threshold
        seen = []
        g.frame_clicked.connect(lambda f: seen.append(f))
        _press(g, pos)
        _move(g, nearby)
        _release(g, nearby)
        assert len(seen) == 1

    def test_no_click_without_data_loaded(self, qtbot):
        g = make_graph(qtbot)
        seen = []
        g.frame_clicked.connect(lambda f: seen.append(f))
        pos = QPoint(50, 50)
        _press(g, pos)
        _release(g, pos)
        assert seen == []

    def test_cursor_is_pointing_hand_over_any_marker(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 15, "orig")  # unmatched marker
        _move(g, pos)
        assert g.cursor().shape() == Qt.CursorShape.PointingHandCursor
    # Cursor/line behavior away from markers is covered by TestHoverCursorAndLine.


class TestHoverCursorAndLine:
    def test_hovering_empty_area_shows_line_and_blank_cursor(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 3, "orig")  # no marker near frame 3
        _move(g, pos)
        assert g._hover_line.isVisible()
        assert g._hover_line.value() == 3
        assert g.cursor().shape() == Qt.CursorShape.BlankCursor

    def test_hovering_matched_marker_shows_line_snapped_to_it(self, qtbot):
        """The line stays visible and snaps to the marker's frame (not the
        raw mouse position) -- this is what disambiguates two markers whose
        X-only hover radii overlap."""
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 5, "orig")  # matched rising pair
        _move(g, pos)
        assert g._hover_line.isVisible()
        assert g._hover_line.value() == 5
        assert g.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_hovering_unmatched_marker_shows_line_snapped_to_it(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 15, "orig")  # unmatched
        _move(g, pos)
        assert g._hover_line.isVisible()
        assert g._hover_line.value() == 15
        assert g.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_press_hides_line_and_shows_closed_hand_immediately(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 3, "orig")
        _move(g, pos)  # establish the line first
        assert g._hover_line.isVisible()
        _press(g, pos)
        assert not g._hover_line.isVisible()
        assert g.cursor().shape() == Qt.CursorShape.ClosedHandCursor

    def test_release_as_click_restores_line_for_release_position(self, qtbot):
        """Regression: hover state must resync to the release position, not
        stay frozen at whatever was hovered before the drag started (moves
        are skipped while _pan_drag_active, so nothing updates it mid-drag)."""
        g = load_graph_with_pairs(qtbot)
        press_pos = _marker_widget_pos(g, 5, "orig")  # a matched marker
        _move(g, press_pos)
        _press(g, press_pos)
        release_pos = _marker_widget_pos(g, 3, "orig")  # empty area, no marker
        _release(g, release_pos)
        assert g._hover_line.isVisible()
        assert g._hover_line.value() == 3
        assert g.cursor().shape() == Qt.CursorShape.BlankCursor

    def test_release_as_click_shows_pointing_hand_if_released_on_a_marker(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 15, "orig")  # unmatched marker, press == release
        _move(g, pos)
        _press(g, pos)
        _release(g, pos)
        assert g._hover_line.isVisible()
        assert g._hover_line.value() == 15
        assert g.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_leave_event_hides_line_and_unsets_cursor(self, qtbot):
        from PyQt6.QtCore import QEvent as _QEvent
        g = load_graph_with_pairs(qtbot)
        pos = _marker_widget_pos(g, 3, "orig")
        _move(g, pos)
        assert g._hover_line.isVisible()
        g.leaveEvent(_QEvent(_QEvent.Type.Leave))
        assert not g._hover_line.isVisible()
        assert g.cursor().shape() != Qt.CursorShape.BlankCursor

    def test_no_data_loaded_no_line_no_crash(self, qtbot):
        g = make_graph(qtbot)
        pos = QPoint(50, 50)
        _move(g, pos)  # must not raise
        assert not g._hover_line.isVisible()


def load_graph_with_ramps(qtbot):
    """Both signals ramp rather than switching instantly, so first-light and
    fully-lit land on different frames and the new markers are distinguishable.

    orig: dark 0-4, ramp 5-7, bright 8-14, ramp 15-17, dark 18-29
          -> rising first=5 full=8, falling first=15 full=18
    disp: the same shape delayed by 4 frames.
    """
    g = make_graph(qtbot)
    g.resize(400, g.height())
    g.show()
    qtbot.waitExposed(g)

    def shape(delay):
        d = np.full(30, 20.0)
        d[5 + delay : 8 + delay] = [70.0, 120.0, 170.0]
        d[8 + delay : 15 + delay] = 220.0
        d[15 + delay : 18 + delay] = [170.0, 120.0, 70.0]
        d[18 + delay :] = 20.0
        return d

    g.set_data(shape(0), shape(4), in_point=0)
    return g


class TestEdgeMarkers:
    def test_both_first_light_and_fully_lit_are_drawn(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        assert sorted(g._sc_rise_orig.data["x"]) == [5.0, 8.0]

    def test_instantaneous_transition_draws_one_marker_not_two(self, qtbot):
        """first == full on a square wave, so the two triangles would land on
        the same frame; drawing both would just darken that pixel. This signal
        has two rising transitions, so two markers total -- not four."""
        g = load_graph_with_pairs(qtbot)
        assert list(g._sc_rise_orig.data["x"]) == [5.0, 15.0]

    def test_anchor_is_no_longer_drawn_on_its_own(self, qtbot):
        """The steepest step stays internal — it drives pairing and
        min-spacing, but it is not one of the three reported frames."""
        g = load_graph_with_ramps(qtbot)
        # Anchor for the orig rising ramp is inside 5..8 but is only drawn if
        # it happens to coincide with first or full.
        assert set(g._sc_rise_orig.data["x"]) == {5.0, 8.0}


class TestMidpointMarker:
    def test_midpoint_sits_at_the_half_frame_with_interpolated_y(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        xs = list(g._sc_mid_rise_orig.data["x"])
        assert xs == [6.5]
        # Between samples 6 (120.0) and 7 (170.0) -> 145.0.
        assert g._sc_mid_rise_orig.data["y"][0] == pytest.approx(145.0)

    def test_no_midpoint_for_an_instantaneous_transition(self, qtbot):
        g = load_graph_with_pairs(qtbot)
        assert len(g._sc_mid_rise_orig.data["x"]) == 0

    def test_midpoint_reads_as_secondary_to_the_real_markers(self):
        from ui.brightness_graph import _MID_MARKER, _MID_MARKER_SIZE
        # It is a derived position, not an observation, so it must not compete
        # with the two measured frames: smaller than their 9px triangles, and
        # achromatic so it never reads as a signal color.
        assert _MID_MARKER_SIZE < 9
        assert len(set(_MID_MARKER)) == 1


class TestNavigationLandsOnFirstLight:
    def test_one_stop_per_transition_not_three(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        # 4 transitions on orig (rise, fall) x (orig, disp) = 4 first-light
        # frames: orig 5, 15; disp 9, 19.
        assert g._transition_frames == [5, 9, 15, 19]

    def test_next_transition_skips_the_fully_lit_frame(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        # From just before the first ramp, the next stop is first-light (5),
        # then the display's first-light (9) -- never the fully-lit 8.
        assert g.next_transition(4) == 5
        assert g.next_transition(5) == 9

    def test_prev_transition_also_lands_on_first_light(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        assert g.prev_transition(15) == 9


class TestFrameToPairOverEdges:
    def test_every_marker_frame_of_a_pair_resolves_to_it(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        rise = g._rise_pairs[0]
        for frame in (5, 8, 9, 12):  # orig first/full, disp first/full
            assert g._frame_to_pair[frame] is rise

    def test_orig_marker_frames_disambiguates_which_signal(self, qtbot):
        """A pair owns four frames now, so which signal a frame belongs to can
        no longer be inferred by comparing it against the pair's anchors."""
        g = load_graph_with_ramps(qtbot)
        assert 5 in g._orig_marker_frames and 8 in g._orig_marker_frames
        assert 9 not in g._orig_marker_frames and 12 not in g._orig_marker_frames


class TestClickSnapping:
    def test_snaps_to_an_edge_marker(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        assert g._hit_test_any_marker(_marker_widget_pos(g, 8, "orig")) == 8

    def test_never_snaps_to_a_midpoint(self, qtbot):
        """Seeking to a half-frame no sample exists at would be a lie, so the
        midpoint dot is drawn but is not a hit-test candidate."""
        g = load_graph_with_ramps(qtbot)
        snapped = [
            g._hit_test_any_marker(_marker_widget_pos(g, f, "orig"))
            for f in (5, 8, 15, 18)
        ]
        assert all(s in (5, 8, 15, 18) for s in snapped if s is not None)
        assert 6 not in snapped and 7 not in snapped


class TestWarningOutlines:
    def test_flagged_transition_markers_carry_the_warning_pen(self, qtbot):
        from ui.brightness_graph import _WARN_OUTLINE

        g = make_graph(qtbot)
        # The baseline tilts steadily before the step, so the "flat" region it
        # is measured from is not flat: unsteady-level. The tilt's per-frame
        # steps stay well under delta, so detection still finds exactly one
        # transition and this stays a test of edges.py's verdict, not of
        # detection's sensitivity.
        orig = np.full(40, 220.0)
        orig[:10] = np.linspace(20.0, 120.0, 10)
        orig[25:] = 20.0
        g.set_data(orig, orig.copy(), in_point=0)
        g.set_delta(50)

        pens = g._sc_rise_orig.data["pen"]
        assert any(p.color().getRgb()[:3] == _WARN_OUTLINE for p in pens)

    def test_clean_signal_carries_no_warning_pen(self, qtbot):
        # A clean signal skips the pen= override entirely, leaving per-point
        # pens as None; a flagged one materializes real QPens. The contract is
        # "no marker wears the warning color", which covers both.
        from ui.brightness_graph import _WARN_OUTLINE

        g = load_graph_with_ramps(qtbot)
        pens = g._sc_rise_orig.data["pen"]
        assert all(
            p is None or p.color().getRgb()[:3] != _WARN_OUTLINE for p in pens
        )


class TestSignalQualityAccessor:
    def test_clean_signal_reports_clean(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        orig_q, disp_q = g.get_signal_quality()
        assert orig_q.is_clean and disp_q.is_clean

    def test_no_data_reports_clean(self, qtbot):
        g = make_graph(qtbot)
        orig_q, disp_q = g.get_signal_quality()
        assert orig_q.is_clean and disp_q.is_clean


class TestSigmaK:
    def test_setting_sigma_k_moves_the_measured_frames(self, qtbot):
        """Needs real noise: on noiseless data sigma is exactly 0, so the band
        falls back to its amplitude/absolute floors and k has no effect at all
        -- correct behavior, but it would make this test vacuous."""
        g = make_graph(qtbot)
        rng = np.random.default_rng(11)
        base = np.full(40, 20.0)
        base[10:14] = [70.0, 120.0, 170.0, 210.0]
        base[14:30] = 220.0
        base[30:] = 20.0
        data = base + rng.normal(0.0, 3.0, 40)
        g.set_data(data, data.copy(), in_point=0)
        g.set_delta(40)

        g.set_sigma_k(1.0)
        loose = list(g._sc_rise_orig.data["x"])
        g.set_sigma_k(12.0)
        tight = list(g._sc_rise_orig.data["x"])
        assert tight != loose

    def test_sigma_k_has_no_effect_without_noise(self, qtbot):
        """Corollary, worth pinning: a clean signal measures the same at any
        sensitivity, so the knob can never make good footage worse."""
        g = load_graph_with_ramps(qtbot)
        before = list(g._sc_rise_orig.data["x"])
        g.set_sigma_k(40.0)
        assert list(g._sc_rise_orig.data["x"]) == before

    def test_sigma_k_is_never_negative(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        g.set_sigma_k(-5.0)
        assert g._sigma_k == 0.0


class TestExcludedCoversEdgeMarkers:
    def test_muting_covers_both_markers_of_an_excluded_transition(self, qtbot):
        from ui.brightness_graph import _MUTED_MARKER

        g = load_graph_with_ramps(qtbot)
        g.set_excluded_pairs({0}, set())
        brushes = g._sc_rise_orig.data["brush"]
        # Both first-light and fully-lit go gray, not just one of them.
        assert len(brushes) == 2
        assert all(b.color().getRgb()[:3] == _MUTED_MARKER for b in brushes)

    def test_falling_markers_unaffected_by_a_rising_exclusion(self, qtbot):
        """Regression: unioning the two excluded sets would let a rising
        exclusion mute a falling marker that shares a frame."""
        g = load_graph_with_ramps(qtbot)
        g.set_excluded_pairs({0}, set())
        assert g._sc_fall_orig.data["brush"][0] is None


class TestManualEditingPipeline:
    """core.manual applied through the graph's own detect->characterize->pair
    chain. The pure functions are covered in test_manual.py; these check the
    wiring and the two invariants the pipeline order exists to protect."""

    def test_an_override_moves_the_marker_and_the_metric(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        pair = g.get_pairs_for("rising")[0]
        before = pair.first_delta_frames()
        g.set_manual_edits([
            ManualEdit("display", "rising", 9, first_frame=8, full_frame=12)
        ])
        assert 8 in list(g._sc_rise_disp.data["x"])
        assert g.get_pairs_for("rising")[0].first_delta_frames() == before - 1

    def test_nudging_never_repairs(self, qtbot):
        """The reason overrides are applied downstream of pairing's inputs:
        pairing keys on anchors, which an override never touches."""
        g = load_graph_with_ramps(qtbot)
        before = [(p.orig_frame, p.disp_frame) for p in g.get_pairs()]
        g.set_manual_edits([
            ManualEdit("display", "rising", 9, first_frame=8, full_frame=20)
        ])
        assert [(p.orig_frame, p.disp_frame) for p in g.get_pairs()] == before

    def test_deleting_a_transition_removes_its_pair(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        before = len(g.get_pairs())
        g.set_manual_edits([ManualEdit("display", "rising", 9, deleted=True)])
        assert len(g.get_pairs()) == before - 1
        assert 9 not in g._rise_disp_frames

    def test_deleting_a_false_read_lets_the_real_one_pair(self, qtbot):
        """The case the feature exists for: a spurious display transition
        detected before the real one steals the greedy match, and deleting it
        hands the match back."""
        orig = np.concatenate([np.full(10, 20.0), np.full(30, 220.0)])
        disp = np.concatenate([
            np.full(12, 20.0), [120.0], np.full(5, 20.0), np.full(22, 220.0),
        ])
        g = make_graph(qtbot)
        g.set_data(orig, disp, in_point=0)
        g.set_delta(60.0)
        blip, real = g._rise_disp_frames[0], g._rise_disp_frames[1]
        assert g.get_pairs_for("rising")[0].disp_frame == blip

        g.set_manual_edits([ManualEdit("display", "rising", blip, deleted=True)])
        assert g.get_pairs_for("rising")[0].disp_frame == real

    def test_edits_survive_an_edge_sensitivity_change(self, qtbot):
        """Anchors do not move when sigma changes, so an edit bound to one
        stays bound - this is what "the user's decision is authoritative"
        means in practice."""
        g = load_graph_with_ramps(qtbot)
        g.set_manual_edits([
            ManualEdit("display", "rising", 9, first_frame=7, full_frame=13)
        ])
        g.set_sigma_k(12.0)
        assert g._disp_edges[9].first_frame == 7
        assert g._disp_edges[9].full_frame == 13
        assert g._disp_edges[9].manual is True

    def test_no_edits_leaves_the_graph_byte_identical(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        before = list(g._sc_rise_disp.data["x"]), list(g._sc_rise_disp.data["y"])
        g.set_manual_edits([])
        assert (list(g._sc_rise_disp.data["x"]), list(g._sc_rise_disp.data["y"])) == before


class TestManualEditingRendering:
    def test_a_manual_marker_gets_the_manual_outline(self, qtbot):
        from ui.brightness_graph import _MANUAL_OUTLINE

        g = load_graph_with_ramps(qtbot)
        g.set_manual_edits([
            ManualEdit("display", "rising", 9, first_frame=8, full_frame=12)
        ])
        pens = g._sc_rise_disp.data["pen"]
        assert any(p.color().getRgb()[:3] == _MANUAL_OUTLINE for p in pens)

    def test_a_deleted_transition_draws_a_cross_and_no_triangle(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        g.set_manual_edits([ManualEdit("display", "rising", 9, deleted=True)])
        assert list(g._sc_deleted.data["x"]) == [9.0]
        assert 9 not in list(g._sc_rise_disp.data["x"])

    def test_a_deleted_transition_is_skipped_by_navigation(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        assert 9 in g._transition_frames
        g.set_manual_edits([ManualEdit("display", "rising", 9, deleted=True)])
        assert 9 not in g._transition_frames

    def test_a_dormant_deletion_draws_nothing(self, qtbot):
        """An edit whose anchor detection no longer finds must not claim a
        transition was removed that was never there this pass."""
        g = load_graph_with_ramps(qtbot)
        g.set_manual_edits([ManualEdit("display", "rising", 999, deleted=True)])
        assert list(g._sc_deleted.data["x"]) == []


class TestManualEditingSelection:
    def test_select_frame_picks_the_marker_drawn_there(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        assert g.select_frame(9)
        assert g.selection() == EditTarget("display", "rising", 9, "first")

    def test_select_frame_returns_false_where_there_is_no_marker(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        assert not g.select_frame(2)
        assert g.selection() is None

    def test_selection_emits_once_per_change(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        seen = []
        g.selection_changed.connect(seen.append)
        g.select_frame(9)
        g.select_frame(9)          # same marker: no second emission
        g.clear_selection()
        assert seen == [EditTarget("display", "rising", 9, "first"), None]

    def test_a_coincident_transition_still_exposes_both_ends(self, qtbot):
        """first == full is the normal shape for an instantaneous transition
        and the graph draws one triangle there, so the fully-lit end has to be
        reachable through the selection rather than by clicking."""
        data = np.concatenate([np.full(10, 20.0), np.full(20, 220.0)])
        g = make_graph(qtbot)
        g.set_data(data, data.copy(), in_point=0)
        anchor = g._rise_orig_frames[0]
        assert g._orig_edges[anchor].first_frame == g._orig_edges[anchor].full_frame
        target = EditTarget("original", "rising", anchor, "full")
        assert g.marker_frame(target) == anchor

    def test_the_selection_ring_follows_a_nudged_marker(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        g.select_frame(9)
        g.set_manual_edits([
            ManualEdit("display", "rising", 9, first_frame=7, full_frame=12)
        ])
        assert list(g._sc_selection.data["x"]) == [7.0]

    def test_a_selection_whose_transition_vanishes_is_dropped(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        g.select_frame(9)
        seen = []
        g.selection_changed.connect(seen.append)
        g.set_delta(500.0)  # nothing survives this threshold
        assert g.selection() is None
        assert seen == [None]

    def test_a_deleted_transition_stays_selectable(self, qtbot):
        """A deletion the user cannot select is one they cannot undo."""
        g = load_graph_with_ramps(qtbot)
        g.set_manual_edits([ManualEdit("display", "rising", 9, deleted=True)])
        assert g.select_frame(9)
        assert g.is_deleted(g.selection())

    def test_transition_position_counts_the_navigation_stops(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        g.select_frame(5)  # the first transition in the clip
        assert g.transition_position(g.selection()) == (1, 4)

    def test_clear_data_drops_the_selection(self, qtbot):
        g = load_graph_with_ramps(qtbot)
        g.select_frame(9)
        g.clear_data()
        assert g.selection() is None
        assert list(g._sc_selection.data["x"]) == []
