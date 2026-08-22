import pytest

from core.view_range import MIN_ZOOM_FRAMES
from ui.zoom_bar import ZoomBarWidget


def make_bar(qtbot, frame_count=1000, analysis=(100, 900)):
    bar = ZoomBarWidget()
    qtbot.addWidget(bar)
    bar.resize(1020, bar.height())  # wide track so pixel<->frame math has real resolution
    bar.reset(frame_count)
    if analysis is not None:
        bar.set_analysis_bounds(*analysis)
    return bar


class TestZoomBarSetup:
    def test_reset_sets_outer_domain(self, qtbot):
        bar = make_bar(qtbot, analysis=None)
        assert bar.lo == 0.0
        assert bar.hi == 999.0

    def test_inert_before_analysis_bounds_set(self, qtbot):
        bar = make_bar(qtbot, analysis=None)
        assert bar._hit_test(500) is None

    def test_set_analysis_bounds_resets_visible_window_to_full(self, qtbot):
        bar = make_bar(qtbot)
        assert (bar.visible_start, bar.visible_end) == (100, 900)

    def test_set_analysis_bounds_is_silent(self, qtbot):
        bar = make_bar(qtbot, analysis=None)
        seen = []
        bar.range_changed.connect(lambda s, e: seen.append((s, e)))
        bar.set_analysis_bounds(100, 900)
        assert seen == []

    def test_analysis_bounds_collapse_goes_inert(self, qtbot):
        bar = make_bar(qtbot)
        bar.set_analysis_bounds(0, -1)  # clear_data sentinel
        assert bar._hit_test(500) is None
        assert (bar.visible_start, bar.visible_end) == (0, 0)


class TestZoomBarSetRange:
    def test_set_range_is_silent(self, qtbot):
        bar = make_bar(qtbot)
        seen = []
        bar.range_changed.connect(lambda s, e: seen.append((s, e)))
        bar.set_range(200, 300)
        assert (bar.visible_start, bar.visible_end) == (200, 300)
        assert seen == []

    def test_set_range_clamps_to_analysis_bounds_not_outer_domain(self, qtbot):
        bar = make_bar(qtbot)
        bar.set_range(0, 2000)  # wider than both analysis (100-900) and outer (0-999)
        assert (bar.visible_start, bar.visible_end) == (100, 900)


class TestZoomBarHandleDrag:
    def test_drag_end_handle_resizes_and_emits(self, qtbot):
        bar = make_bar(qtbot)
        seen = []
        bar.range_changed.connect(lambda s, e: seen.append((s, e)))
        bar._drag_target = "end"
        bar._apply_drag(bar._frame_to_x(500))
        assert bar.visible_start == 100
        assert bar.visible_end == pytest.approx(500, abs=1)
        assert seen

    def test_drag_end_handle_floors_at_min_zoom_width(self, qtbot):
        bar = make_bar(qtbot)
        bar._drag_target = "end"
        bar._apply_drag(bar._frame_to_x(50))  # left of start(100) entirely
        assert bar.visible_end == bar.visible_start + MIN_ZOOM_FRAMES

    def test_drag_start_handle_cannot_pass_analysis_lo(self, qtbot):
        bar = make_bar(qtbot)
        bar._drag_target = "start"
        bar._apply_drag(bar._frame_to_x(0))  # before analysis_lo=100
        assert bar.visible_start == 100

    def test_drag_start_handle_floors_at_min_zoom_width_from_end(self, qtbot):
        bar = make_bar(qtbot)
        bar._drag_target = "start"
        bar._apply_drag(bar._frame_to_x(950))  # past end(900) entirely
        assert bar.visible_start == bar.visible_end - MIN_ZOOM_FRAMES


class TestZoomBarMiddleDrag:
    def test_pans_without_changing_width(self, qtbot):
        bar = make_bar(qtbot)
        bar.set_range(300, 500)
        seen = []
        bar.range_changed.connect(lambda s, e: seen.append((s, e)))
        bar._drag_target = "middle"
        bar._drag_start_x = bar._frame_to_x(400)
        bar._drag_start_range = (300, 500)
        bar._apply_drag(bar._frame_to_x(450))  # drag right ~50 frames
        assert bar.visible_end - bar.visible_start == pytest.approx(200, abs=1)
        assert bar.visible_start == pytest.approx(350, abs=2)
        assert seen

    def test_clamps_at_analysis_edge_without_changing_width(self, qtbot):
        bar = make_bar(qtbot)
        bar.set_range(100, 300)  # already pinned at analysis_lo
        bar._drag_target = "middle"
        bar._drag_start_x = bar._frame_to_x(200)
        bar._drag_start_range = (100, 300)
        bar._apply_drag(bar._frame_to_x(0))  # try to drag further left
        assert bar.visible_start == 100
        assert bar.visible_end - bar.visible_start == 200


class TestZoomBarReset:
    def test_reset_to_full_restores_analysis_range_and_emits(self, qtbot):
        bar = make_bar(qtbot)
        bar.set_range(300, 500)
        seen = []
        bar.range_changed.connect(lambda s, e: seen.append((s, e)))
        bar._reset_to_full()
        assert (bar.visible_start, bar.visible_end) == (100, 900)
        assert seen == [(100, 900)]

    def test_reset_to_full_noop_when_inert(self, qtbot):
        bar = make_bar(qtbot, analysis=None)
        seen = []
        bar.range_changed.connect(lambda s, e: seen.append((s, e)))
        bar._reset_to_full()
        assert seen == []
