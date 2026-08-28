"""
MainWindow integration tests against the synthetic clip (see conftest).

Extraction runs the real QThread; tests wait for the built-in
QThread.finished cleanup slot to clear MainWindow._extractor.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from core.roi import ROI
from tests.conftest import SYNTH_LATENCY, SYNTH_W, SYNTH_H
from ui.main_window import (COL_DISP_FIRST, COL_ORIG_FIRST, SUMMARY_METRICS,
                            MainWindow, _existing_file)

ROI_ORIG = ROI(2, 2, SYNTH_W // 2 - 4, SYNTH_H - 4)
ROI_DISP = ROI(SYNTH_W // 2 + 2, 2, SYNTH_W // 2 - 4, SYNTH_H - 4)


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def loaded(window, synth_video):
    window.open_file(synth_video)
    window.frame_view.set_roi("original", ROI_ORIG)
    window.frame_view.set_roi("display", ROI_DISP)
    return window


def analyze(win, qtbot):
    win._on_analyze_clicked()
    qtbot.waitUntil(lambda: win._extractor is None, timeout=10000)


def _multi_pair_arrays(latency=2):
    """3 rising + 3 falling transitions per signal, disp shifted `latency`
    frames later than orig -- more pairs than synth_video's 1+1, for tests
    that need partial exclusion to be meaningful."""
    block = [20.0] * 6
    pattern = (block + [220.0] * 6) * 3 + block
    orig = np.array(pattern, dtype=np.float64)
    disp = np.concatenate([np.full(latency, 20.0), orig[:-latency]])
    return orig, disp


def load_multi_pairs(win):
    """Loads 3 rising + 3 falling matched pairs directly into the graph,
    bypassing extraction, and pins the results panel to show them."""
    orig, disp = _multi_pair_arrays()
    win.brightness_graph.set_data(orig, disp, in_point=0)
    win._results_polarity = "both"
    win._update_results_table()
    return win


class TestAnalysisLifecycle:
    def test_summary_tables_placeholder_before_analysis(self, window):
        """Regression: each direction's Mean/Min/Max/Median summary must
        always be visible, showing a placeholder rather than being blank
        when nothing is loaded."""
        for model in (window._rise_summary_model, window._fall_summary_model):
            assert [model.headerData(c, Qt.Orientation.Horizontal) for c in range(5)] == \
                ["Metric", "Mean", "Min", "Max", "Median"]
            assert model.rowCount() == len(SUMMARY_METRICS)
            for row, (label, _) in enumerate(SUMMARY_METRICS):
                assert model.item(row, 0).text() == label
                assert [model.item(row, c).text() for c in range(1, 5)] == ["--.- ms"] * 4

    def test_summary_tables_populated_after_analysis(self, loaded, qtbot):
        """The synthetic clip's transitions are instantaneous, so first = full =
        anchor and all three metric rows report the same number as the tool did
        before the metrics were split apart."""
        analyze(loaded, qtbot)
        fps = loaded.reader.fps_effective
        rise_pairs = loaded.brightness_graph.get_pairs_for("rising")
        fall_pairs = loaded.brightness_graph.get_pairs_for("falling")
        rise_expected = f"{rise_pairs[0].delta_ms(fps):.1f} ms"
        fall_expected = f"{fall_pairs[0].delta_ms(fps):.1f} ms"
        for row in range(len(SUMMARY_METRICS)):
            assert [loaded._rise_summary_model.item(row, c).text() for c in range(1, 5)] == [rise_expected] * 4
            assert [loaded._fall_summary_model.item(row, c).text() for c in range(1, 5)] == [fall_expected] * 4

    def test_analysis_finds_known_latency(self, loaded, qtbot):
        analyze(loaded, qtbot)
        pairs = loaded.brightness_graph.get_pairs()
        assert len(pairs) == 2  # one rising, one falling
        assert all(p.delta_frames() == SYNTH_LATENCY for p in pairs)
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1
        assert loaded.export_csv_btn.isEnabled()

    def test_results_table_refreshes_on_delta_change(self, loaded, qtbot):
        """Regression: adjusting Min ΔBrightness after analysis must repopulate
        the results tables live, not leave them stuck on the pre-adjustment
        (possibly empty) pairing."""
        loaded.delta_spin.setValue(255)  # exceeds the synthetic clip's max step -> 0 pairs
        analyze(loaded, qtbot)
        assert loaded._rise_results_model.rowCount() == 0
        assert loaded._fall_results_model.rowCount() == 0

        loaded.delta_spin.setValue(30)
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1

    def test_results_tables_split_by_polarity(self, loaded, qtbot):
        """Regression: rise table must show only rising pairs, fall table only
        falling pairs — no cross-contamination between the two panels."""
        analyze(loaded, qtbot)
        rise_pairs = loaded.brightness_graph.get_pairs_for("rising")
        fall_pairs = loaded.brightness_graph.get_pairs_for("falling")
        assert loaded._rise_results_model.rowCount() == len(rise_pairs) == 1
        assert loaded._fall_results_model.rowCount() == len(fall_pairs) == 1
        assert int(loaded._rise_results_model.item(0, loaded._orig_frame_col).text()) == rise_pairs[0].orig_frame
        assert int(loaded._fall_results_model.item(0, loaded._orig_frame_col).text()) == fall_pairs[0].orig_frame

    def test_results_panels_always_visible(self, loaded, qtbot):
        """Regression: both direction panels stay visible at all times —
        before any analysis, and after a single-direction analysis — only
        their row/summary content goes blank, the panel itself never hides."""
        assert loaded.rise_results_container.isVisible()
        assert loaded.fall_results_container.isVisible()

        idx = loaded.polarity_combo.findData("rising")
        loaded.polarity_combo.setCurrentIndex(idx)
        analyze(loaded, qtbot)
        assert loaded.rise_results_container.isVisible()
        assert loaded.fall_results_container.isVisible()
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 0
        for row in range(len(SUMMARY_METRICS)):
            assert [loaded._fall_summary_model.item(row, c).text() for c in range(1, 5)] == ["--.- ms"] * 4

    def test_results_panel_pinned_until_next_analyze(self, loaded, qtbot):
        """Regression: the results panel content must stay pinned to the
        polarity used for the last Analyze — the Direction pulldown alone
        must not reshuffle a loaded result set. Only clicking Analyze again
        should apply a new pulldown selection."""
        idx = loaded.polarity_combo.findData("rising")
        loaded.polarity_combo.setCurrentIndex(idx)
        analyze(loaded, qtbot)
        assert loaded._fall_results_model.rowCount() == 0

        idx = loaded.polarity_combo.findData("both")
        loaded.polarity_combo.setCurrentIndex(idx)
        assert loaded._fall_results_model.rowCount() == 0

        analyze(loaded, qtbot)
        assert loaded._fall_results_model.rowCount() == 1

    def test_analyze_button_reenabled_after_completion(self, loaded, qtbot):
        analyze(loaded, qtbot)
        assert loaded.analyze_btn.isEnabled()

    def test_analyze_button_reenabled_after_cancel(self, loaded, qtbot):
        """Regression: after Cancel the button used to stay disabled until
        an unrelated control was touched."""
        loaded._on_analyze_clicked()
        loaded._on_cancel_clicked()
        qtbot.waitUntil(lambda: loaded._extractor is None, timeout=10000)
        assert loaded.analyze_btn.isEnabled()

    def test_cancel_never_yields_results(self, loaded, qtbot):
        """Regression: a result queued in the instant before Cancel was
        clicked still populated the UI the user had just cancelled."""
        loaded._on_analyze_clicked()
        loaded._on_cancel_clicked()
        qtbot.waitUntil(lambda: loaded._extractor is None, timeout=10000)
        assert loaded.brightness_graph.get_pairs() == []
        assert loaded._rise_results_model.rowCount() == 0
        assert loaded._fall_results_model.rowCount() == 0

    def test_results_table_headers(self, window):
        """Regression: columns used to label orig_frame 'Display Frame'."""
        expected = ["Exclude", "⚠", "Original\n1st Pixel", "Display\n1st Pixel",
                    "First (ms)", "Avg (ms)", "Full (ms)"]
        for model in (window._rise_results_model, window._fall_results_model):
            headers = [
                model.headerData(c, Qt.Orientation.Horizontal)
                for c in range(model.columnCount())
            ]
            assert headers == expected


class TestDeltaThreshold:
    def test_cli_min_delta_applied_once(self, loaded, qtbot):
        """Regression: the CLI value used to clobber the spinbox on every
        re-analysis."""
        loaded._cli_args = SimpleNamespace(min_delta=60, max_latency=None)
        analyze(loaded, qtbot)
        assert loaded.delta_spin.value() == 60
        assert loaded._cli_args.min_delta is None

        loaded.delta_spin.setValue(25)  # signals live -> marks user-set
        analyze(loaded, qtbot)
        assert loaded.delta_spin.value() == 25

    def test_user_delta_survives_reanalysis(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded.delta_spin.setValue(33)
        analyze(loaded, qtbot)
        assert loaded.delta_spin.value() == 33

    def test_auto_delta_shown_equals_effective(self, loaded, qtbot):
        """Regression: spinbox showed a rounded int while the graph kept the
        float, so the displayed threshold was not the one in effect."""
        analyze(loaded, qtbot)
        assert float(loaded.delta_spin.value()) == loaded.brightness_graph._delta


class TestMaxLatencyDefault:
    def test_stays_unlimited_when_period_unavailable(self, loaded, qtbot):
        # synth_video has only 1 rising + 1 falling transition -> no period.
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 0

    def test_auto_value_is_half_orig_period(self, loaded, qtbot, monkeypatch):
        # Isolate the wiring from real period detection by stubbing the period.
        monkeypatch.setattr(
            loaded.brightness_graph, "get_orig_period_frames", lambda polarity: 20.0
        )
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 10

    def test_cli_max_latency_applied_once(self, loaded, qtbot):
        loaded._cli_args = SimpleNamespace(min_delta=None, max_latency=7)
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 7
        assert loaded._cli_args.max_latency is None

        loaded.max_latency_spin.setValue(3)  # signals live -> marks user-set
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 3

    def test_cli_max_latency_zero_is_explicit_unlimited(self, loaded, qtbot):
        loaded._cli_args = SimpleNamespace(min_delta=None, max_latency=0)
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 0
        assert loaded._cli_args.max_latency is None
        # Re-analyze without new CLI input: must not get silently re-defaulted.
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 0

    def test_user_max_latency_survives_reanalysis(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded.max_latency_spin.setValue(4)
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 4

    def test_fresh_file_load_clears_user_set_flag(self, loaded, qtbot, synth_video):
        analyze(loaded, qtbot)
        loaded.max_latency_spin.setValue(4)
        assert loaded._max_latency_user_set is True
        loaded.open_file(synth_video)
        assert loaded._max_latency_user_set is False

    def test_auto_button_disabled_until_analysis(self, loaded, qtbot):
        assert loaded.max_latency_auto_btn.isEnabled() is False
        analyze(loaded, qtbot)
        assert loaded.max_latency_auto_btn.isEnabled() is True

    def test_auto_button_sets_half_orig_period(self, loaded, qtbot, monkeypatch):
        monkeypatch.setattr(
            loaded.brightness_graph, "get_orig_period_frames", lambda polarity: 20.0
        )
        analyze(loaded, qtbot)
        loaded.max_latency_spin.setValue(999)
        loaded.max_latency_auto_btn.click()
        assert loaded.max_latency_spin.value() == 10
        assert loaded.brightness_graph._max_latency == 10

    def test_auto_button_click_counts_as_user_edit(self, loaded, qtbot, monkeypatch):
        monkeypatch.setattr(
            loaded.brightness_graph, "get_orig_period_frames", lambda polarity: 20.0
        )
        analyze(loaded, qtbot)
        loaded.max_latency_auto_btn.click()
        assert loaded._max_latency_user_set is True

        # Period changes, but a re-analysis must not silently override the
        # value the Auto button just applied -- it's a one-time snap, not a
        # standing auto-mode.
        monkeypatch.setattr(
            loaded.brightness_graph, "get_orig_period_frames", lambda polarity: 40.0
        )
        analyze(loaded, qtbot)
        assert loaded.max_latency_spin.value() == 10


class TestRoiInvalidation:
    def test_undo_clears_stale_results(self, loaded, qtbot):
        """Regression: Ctrl+Z restored ROIs but left pairs/table/summary from
        the pre-undo ROI on screen."""
        analyze(loaded, qtbot)
        assert loaded.export_csv_btn.isEnabled()
        loaded._undo_roi()
        assert loaded.brightness_graph.get_pairs() == []
        assert loaded._rise_results_model.rowCount() == 0
        assert loaded._fall_results_model.rowCount() == 0
        assert not loaded.export_csv_btn.isEnabled()


class TestSessionInvalidation:
    def test_stale_result_from_old_session_dropped(self, loaded, qtbot):
        """Regression: a queued extraction_done from an invalidated session
        (file re-opened, analysis restarted) used to overwrite the current
        session's results."""
        import numpy as np

        analyze(loaded, qtbot)
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1
        stale = np.zeros(5, dtype=np.float32)
        loaded._on_extract_finished(
            stale, stale, 0, session=loaded._extraction_session - 1
        )
        assert loaded.brightness_graph._n == 40  # untouched
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1

    def test_roi_change_mid_analysis_discards_results(self, loaded, qtbot):
        """Regression: editing an ROI while extraction ran left results for
        the old ROI on screen under the new ROI's overlay."""
        loaded._on_analyze_clicked()
        loaded.frame_view.set_roi(
            "original", ROI(1, 1, SYNTH_W // 2 - 2, SYNTH_H - 2)
        )
        qtbot.waitUntil(lambda: loaded._extractor is None, timeout=10000)
        assert loaded.brightness_graph.get_pairs() == []
        assert loaded._rise_results_model.rowCount() == 0
        assert loaded._fall_results_model.rowCount() == 0
        assert not loaded.export_csv_btn.isEnabled()

    def test_failed_open_preserves_session(self, loaded, qtbot, monkeypatch):
        """Regression: a bad path released the current reader before failing,
        bricking scrubbing and wiping results."""
        monkeypatch.setattr("ui.main_window.QMessageBox.warning", lambda *a, **k: None)
        analyze(loaded, qtbot)
        frames_before = loaded.reader.frame_count
        loaded.open_file("/nonexistent/nope.mp4")
        assert "Error" in loaded.status_label.text()
        assert loaded.reader.frame_count == frames_before
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1
        loaded.show_frame(5)  # reader still usable

    def test_failed_open_shows_message_box(self, loaded, qtbot, monkeypatch):
        """The status-bar text alone is easy to miss; a bad path (regardless
        of how open_file() was reached) must also raise a modal dialog."""
        calls = []
        monkeypatch.setattr(
            "ui.main_window.QMessageBox.warning",
            lambda *a, **k: calls.append(a),
        )
        loaded.open_file("/nonexistent/nope.mp4")
        assert len(calls) == 1


class TestDragDropOpen:
    @staticmethod
    def _drag_enter_event(mime: QMimeData) -> QDragEnterEvent:
        event = QDragEnterEvent(
            QPoint(0, 0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        # QDragEnterEvent doesn't hold a Python reference to `mime`, only a
        # raw pointer -- without this, mime is garbage-collected as soon as
        # this function returns, leaving the event with a dangling pointer
        # (crashes with an access violation on the next mimeData() access).
        event._mime = mime
        return event

    @staticmethod
    def _drop_event(paths: list[str]) -> QDropEvent:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        event = QDropEvent(
            QPointF(0, 0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        event._mime = mime  # see _drag_enter_event
        return event

    def test_drag_enter_accepts_local_file(self, window):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("C:/video.mp4")])
        event = self._drag_enter_event(mime)
        window.dragEnterEvent(event)
        assert event.isAccepted()

    def test_drag_enter_ignores_non_file_data(self, window):
        mime = QMimeData()
        mime.setText("hello")
        event = self._drag_enter_event(mime)
        window.dragEnterEvent(event)
        assert not event.isAccepted()

    def test_drop_opens_file(self, window, synth_video):
        """Dropping a valid video routes through the same open_file() loader
        as the Open Video button."""
        event = self._drop_event([synth_video])
        window.dropEvent(event)
        assert event.isAccepted()
        assert window.reader is not None
        assert window.file_label.text() == Path(synth_video).name

    def test_drop_bad_path_preserves_session(self, loaded, qtbot, monkeypatch):
        """Same regression as test_failed_open_preserves_session, via drop
        instead of the dialog/CLI path."""
        monkeypatch.setattr("ui.main_window.QMessageBox.warning", lambda *a, **k: None)
        analyze(loaded, qtbot)
        frames_before = loaded.reader.frame_count
        event = self._drop_event(["/nonexistent/nope.mp4"])
        loaded.dropEvent(event)
        assert "Error" in loaded.status_label.text()
        assert loaded.reader.frame_count == frames_before
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1


class TestCliCommand:
    def test_min_delta_omitted_before_first_analysis(self, loaded):
        """Regression: Show CLI printed the spinbox default (10), a threshold
        the session never used."""
        assert "--min-delta" not in loaded._build_cli_command()

    def test_min_delta_included_after_analysis(self, loaded, qtbot):
        analyze(loaded, qtbot)
        assert f"--min-delta {loaded.delta_spin.value()}" in loaded._build_cli_command()

    def test_out_of_bounds_cli_roi_warns(self, loaded):
        """Regression: an ROI from a higher-res recording clipped to a 1px
        sliver — possibly on the wrong screen — with no warning."""
        args = SimpleNamespace(
            fps=None, direction=None,
            roi_original=(5000, 5000, 100, 100), roi_display=None,
            min_delta=None, min_spacing=None, max_latency=None, edge_sigma=None,
            in_point=None, out_point=None,
        )
        loaded.apply_cli_args(args)
        assert "Warning" in loaded.status_label.text()

    def test_conflicting_in_out_points_warns(self, loaded):
        """Regression: --out-point silently clamped to in_point+1 when it
        conflicted with --in-point, with no indication it wasn't honoured."""
        args = SimpleNamespace(
            fps=None, direction=None,
            roi_original=None, roi_display=None,
            min_delta=None, min_spacing=None, max_latency=None, edge_sigma=None,
            in_point=30, out_point=20,
        )
        loaded.apply_cli_args(args)
        assert loaded.timeline.out_point == 31
        assert "Warning" in loaded.status_label.text()
        assert "--out-point" in loaded.status_label.text()

    def test_cli_missing_file_rejected(self):
        """Regression: a bad CLI filename must refuse to run, not launch the
        GUI anyway with just a status-bar error."""
        with pytest.raises(argparse.ArgumentTypeError):
            _existing_file("/nonexistent/nope.mp4")

    def test_cli_existing_file_passes_through(self, synth_video):
        assert _existing_file(synth_video) == synth_video


class TestKeyboardNavigation:
    def test_arrow_steps_frame_via_window(self, loaded, qtbot):
        assert loaded.timeline.current_frame == 0
        qtbot.keyClick(loaded, Qt.Key.Key_Right)
        assert loaded.timeline.current_frame == 1
        qtbot.keyClick(loaded, Qt.Key.Key_Left)
        assert loaded.timeline.current_frame == 0

    def test_focused_spinbox_keeps_arrow_keys(self, loaded, qtbot):
        """Regression: window-level shortcuts used to steal Up/Down from the
        FPS spinbox, making arrow-increment impossible."""
        before_frame = loaded.timeline.current_frame
        fps_before = loaded.fps_spin.value()
        loaded.fps_spin.setFocus()
        qtbot.keyClick(loaded.fps_spin, Qt.Key.Key_Up)
        assert loaded.fps_spin.value() == pytest.approx(fps_before + 1.0)
        assert loaded.timeline.current_frame == before_frame

    def test_in_out_marking(self, loaded, qtbot):
        loaded.show_frame(15)
        qtbot.keyClick(loaded, Qt.Key.Key_I)
        assert loaded.timeline.in_point == 15
        loaded.show_frame(30)
        qtbot.keyClick(loaded, Qt.Key.Key_O)
        assert loaded.timeline.out_point == 30

    def test_escape_cancels_running_analysis(self, loaded, qtbot):
        loaded._on_analyze_clicked()
        qtbot.keyClick(loaded, Qt.Key.Key_Escape)
        qtbot.waitUntil(lambda: loaded._extractor is None, timeout=10000)
        assert loaded.brightness_graph.get_pairs() == []
        assert loaded._rise_results_model.rowCount() == 0
        assert loaded._fall_results_model.rowCount() == 0

    def test_no_scrolling_widget_can_steal_navigation_keys(self, loaded):
        """Regression, twice over. QScrollArea (wrapping the left column) and
        BrightnessGraphWidget (a pyqtgraph PlotWidget, therefore a
        QGraphicsView) are both QAbstractScrollArea subclasses. That class
        handles the arrow keys itself to scroll its viewport, and defaults to
        StrongFocus — so any one of them that can take focus swallows every
        navigation key before MainWindow.keyPressEvent ever sees it. Frame
        stepping and transition jumps just stop working after a click.

        Asserted over every such widget rather than the two known offenders, so
        a third one added later fails here instead of in the user's hands.
        """
        from PyQt6.QtWidgets import QAbstractScrollArea

        offenders = sorted(
            type(w).__name__
            for w in loaded.findChildren(QAbstractScrollArea)
            if w.isVisible() and w.focusPolicy() != Qt.FocusPolicy.NoFocus
        )
        assert offenders == []

    def test_arrows_survive_a_click_on_the_graph(self, loaded, qtbot):
        """Dispatches through the focus chain, unlike
        test_arrow_steps_frame_via_window which sends keys straight at the
        window — bypassing focus entirely, which is exactly how that test
        stayed green while the real app was broken.

        Clicking the graph is the specific thing that broke it: click-to-seek
        made it a routine action, and before that nobody clicked there."""
        qtbot.mouseClick(loaded.brightness_graph, Qt.MouseButton.LeftButton)
        before = loaded.timeline.current_frame
        target = loaded.focusWidget() or loaded
        qtbot.keyClick(target, Qt.Key.Key_Right)
        assert loaded.timeline.current_frame == before + 1

    def test_arrows_survive_a_click_on_the_timeline(self, loaded, qtbot):
        """Regression: dragging the in/out/playhead handle gave TimelineWidget
        ClickFocus. It doesn't handle keys itself, so the ignored keypress
        propagated to left_scroll (the QAbstractScrollArea wrapping the left
        column), which swallows arrow keys to scroll its viewport -- same
        failure mode as test_arrows_survive_a_click_on_the_graph, just via a
        plain QWidget descendant instead of the scroll area taking focus
        directly."""
        qtbot.mouseClick(loaded.timeline, Qt.MouseButton.LeftButton)
        before = loaded.timeline.current_frame
        target = loaded.focusWidget() or loaded
        qtbot.keyClick(target, Qt.Key.Key_Right)
        assert loaded.timeline.current_frame == before + 1

    def test_arrows_survive_a_click_on_the_video(self, loaded, qtbot):
        """Regression: same failure mode via the video preview (RoiFrameView),
        also nested inside left_scroll."""
        qtbot.mouseClick(loaded.frame_view, Qt.MouseButton.LeftButton)
        before = loaded.timeline.current_frame
        target = loaded.focusWidget() or loaded
        qtbot.keyClick(target, Qt.Key.Key_Right)
        assert loaded.timeline.current_frame == before + 1

    @pytest.mark.parametrize("table_attr", ["rise_results_table", "fall_results_table"])
    def test_table_click_does_not_steal_keyboard_focus(self, loaded, qtbot, table_attr):
        """Regression: results tables had no focus policy, so QTableView's
        default StrongFocus let a click steal focus and swallow the
        keyPressEvent-based navigation shortcuts (arrows, Home/End, I/O, etc.)."""
        table = getattr(loaded, table_attr)
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton)
        assert not table.hasFocus()


class TestZoomPanWiring:
    def test_zoom_bar_outer_domain_set_on_open(self, loaded):
        from tests.conftest import SYNTH_FRAME_COUNT
        assert loaded.zoom_bar.lo == 0.0
        assert loaded.zoom_bar.hi == float(SYNTH_FRAME_COUNT - 1)

    def test_zoom_bar_analysis_bounds_follow_graph_domain(self, loaded, qtbot):
        analyze(loaded, qtbot)
        assert loaded.zoom_bar.analysis_lo == loaded.brightness_graph._range_lo
        assert loaded.zoom_bar.analysis_hi == loaded.brightness_graph._range_hi
        assert (loaded.zoom_bar.visible_start, loaded.zoom_bar.visible_end) == (
            loaded.zoom_bar.analysis_lo, loaded.zoom_bar.analysis_hi,
        )

    def test_dragging_bar_handle_updates_graph_visible_range(self, loaded, qtbot):
        analyze(loaded, qtbot)
        bar = loaded.zoom_bar
        bar.resize(400, bar.height())
        bar._drag_target = "end"
        bar._apply_drag(bar._frame_to_x(bar.analysis_lo + 5))
        assert (loaded.brightness_graph._visible_start, loaded.brightness_graph._visible_end) == (
            bar.visible_start, bar.visible_end,
        )

    def test_graph_wheel_zoom_updates_bar(self, loaded, qtbot):
        analyze(loaded, qtbot)
        g = loaded.brightness_graph
        anchor = (g._range_lo + g._range_hi) / 2
        g._apply_zoom(0.5, anchor)
        assert (loaded.zoom_bar.visible_start, loaded.zoom_bar.visible_end) == (
            g._visible_start, g._visible_end,
        )

    def test_new_analysis_resets_bar_to_full_after_zoom(self, loaded, qtbot):
        analyze(loaded, qtbot)
        g = loaded.brightness_graph
        anchor = (g._range_lo + g._range_hi) / 2
        g._apply_zoom(0.5, anchor)
        assert (loaded.zoom_bar.visible_start, loaded.zoom_bar.visible_end) != (
            loaded.zoom_bar.analysis_lo, loaded.zoom_bar.analysis_hi,
        )
        analyze(loaded, qtbot)  # re-Analyze
        assert (loaded.zoom_bar.visible_start, loaded.zoom_bar.visible_end) == (
            loaded.zoom_bar.analysis_lo, loaded.zoom_bar.analysis_hi,
        )


class TestStartupWindowState:
    def test_starts_filling_available_screen_geometry(self, window):
        """Regression: showMaximized()'s automatic geometry calculation could
        report WindowMaximized while filling only ~2/3 of the screen on
        multi-monitor Windows setups. Geometry is set explicitly from the
        screen's available area; allow a little slack for window-frame/title
        bar bookkeeping (varies by platform), which isn't the bug being
        guarded against."""
        avail = window.screen().availableGeometry()
        geo = window.geometry()
        assert window.windowState() & Qt.WindowState.WindowMaximized
        assert abs(geo.width() - avail.width()) <= 10
        assert abs(geo.height() - avail.height()) <= 10


class TestNavButtonLabels:
    def test_labels_follow_arrow_word_convention(self, window):
        assert window.prev_button.text() == "<< Frame"
        assert window.next_button.text() == "Frame >>"
        assert window.prev_trans_button.text() == "<< Transition"
        assert window.next_trans_button.text() == "Transition >>"
        assert window.prev_unmatched_button.text() == "<< Unmatched"
        assert window.next_unmatched_button.text() == "Unmatched >>"


class TestUnmatchedNavButtons:
    def test_disabled_before_analysis_enabled_after(self, loaded, qtbot):
        assert not loaded.prev_unmatched_button.isEnabled()
        assert not loaded.next_unmatched_button.isEnabled()
        analyze(loaded, qtbot)
        assert loaded.prev_unmatched_button.isEnabled()
        assert loaded.next_unmatched_button.isEnabled()

    def test_disabled_again_when_analysis_is_invalidated(self, loaded, qtbot):
        """Mirrors the existing prev/next_trans_button lifecycle: an ROI
        change invalidates the completed analysis and clears its results."""
        analyze(loaded, qtbot)
        loaded._on_clear_rois()
        assert not loaded.prev_unmatched_button.isEnabled()
        assert not loaded.next_unmatched_button.isEnabled()

    def test_next_unmatched_click_moves_playhead_to_unmatched_frame(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded.brightness_graph._unmatched_frames = [30]
        loaded.show_frame(0)
        qtbot.mouseClick(loaded.next_unmatched_button, Qt.MouseButton.LeftButton)
        assert loaded.timeline.current_frame == 30

    def test_prev_unmatched_click_moves_playhead_to_unmatched_frame(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded.brightness_graph._unmatched_frames = [5]
        loaded.show_frame(39)
        qtbot.mouseClick(loaded.prev_unmatched_button, Qt.MouseButton.LeftButton)
        assert loaded.timeline.current_frame == 5

    def test_click_with_no_unmatched_transitions_is_a_noop(self, loaded, qtbot):
        analyze(loaded, qtbot)  # the synthetic clip's transitions are all matched
        loaded.show_frame(10)
        qtbot.mouseClick(loaded.next_unmatched_button, Qt.MouseButton.LeftButton)
        assert loaded.timeline.current_frame == 10


def _varied_latency_arrays():
    """3 rising + 3 falling transitions per signal, disp offset by 2/4/6
    frames respectively -- unlike load_multi_pairs's uniform latency, this
    lets a test prove excluding a specific pair actually changes the
    summary's numeric value, not just its row count."""
    orig = np.array(
        [20.0] * 5 + [220.0] * 5 + [20.0] * 5 + [220.0] * 5 + [20.0] * 5 + [220.0] * 5 + [20.0] * 10,
        dtype=np.float64,
    )  # rising @ 5,15,25 -- falling @ 10,20,30
    disp = np.array(
        [20.0] * 7 + [220.0] * 5 + [20.0] * 7 + [220.0] * 5 + [20.0] * 7 + [220.0] * 5 + [20.0] * 4,
        dtype=np.float64,
    )  # rising @ 7,19,31 -- falling @ 12,24,36  (latencies: 2, 4, 6)
    assert len(orig) == len(disp)
    return orig, disp


class TestExcludePairs:
    def test_exclude_column_present_and_checkable(self, loaded):
        load_multi_pairs(loaded)
        item = loaded._rise_results_model.item(0, loaded._exclude_col)
        assert item.isCheckable()
        assert item.checkState() == Qt.CheckState.Unchecked

    def test_checking_exclude_removes_pair_from_summary(self, loaded):
        orig, disp = _varied_latency_arrays()
        loaded.brightness_graph.set_data(orig, disp, in_point=0)
        loaded._results_polarity = "both"
        loaded._update_results_table()
        fps = loaded.reader.fps_effective
        rise_pairs = loaded.brightness_graph.get_pairs_for("rising", active="both")
        assert [p.delta_frames() for p in rise_pairs] == [2, 4, 6]

        # Exclude the 6-frame outlier (row 2).
        loaded._rise_results_model.item(2, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)

        # Column 0 is the metric name now, so the stats start at column 1.
        remaining_ms = [p.avg_delta_ms(fps) for p in rise_pairs[:2]]
        avg_row = [m for m, _ in SUMMARY_METRICS].index("Average")
        assert loaded._rise_summary_model.item(avg_row, 1).text() == \
            f"{sum(remaining_ms) / len(remaining_ms):.1f} ms"
        assert loaded._rise_summary_model.item(avg_row, 3).text() == f"{max(remaining_ms):.1f} ms"

    def test_excluded_row_stays_visible_and_checked_in_normal_view(self, loaded):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        assert loaded._rise_results_model.rowCount() == 3
        assert loaded.rise_results_table.model().rowCount() == 3
        assert loaded._rise_results_model.item(0, loaded._exclude_col).checkState() == Qt.CheckState.Checked

    def test_show_excluded_filters_to_only_excluded_rows(self, loaded, qtbot):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(1, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        qtbot.mouseClick(loaded.rise_show_excluded_btn, Qt.MouseButton.LeftButton)
        proxy = loaded.rise_results_table.model()
        assert proxy.rowCount() == 1
        assert int(proxy.index(0, loaded._orig_frame_col).data()) == loaded.brightness_graph._rise_pairs[1].orig_frame

    def test_show_excluded_does_not_change_summary_stats(self, loaded, qtbot):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        before = [loaded._rise_summary_model.item(0, c).text() for c in range(4)]
        qtbot.mouseClick(loaded.rise_show_excluded_btn, Qt.MouseButton.LeftButton)
        after = [loaded._rise_summary_model.item(0, c).text() for c in range(4)]
        assert before == after

    def test_clear_all_clears_exclusions_and_turns_off_show_excluded(self, loaded, qtbot):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        qtbot.mouseClick(loaded.rise_show_excluded_btn, Qt.MouseButton.LeftButton)
        assert loaded.rise_show_excluded_btn.isChecked()

        qtbot.mouseClick(loaded.rise_clear_excluded_btn, Qt.MouseButton.LeftButton)
        assert not loaded.rise_show_excluded_btn.isChecked()
        assert loaded.rise_results_table.model().rowCount() == 3
        assert loaded._rise_results_model.item(0, loaded._exclude_col).checkState() == Qt.CheckState.Unchecked

    def test_exclusion_independent_per_direction(self, loaded):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        assert loaded._fall_results_model.item(0, loaded._exclude_col).checkState() == Qt.CheckState.Unchecked
        assert loaded.rise_clear_excluded_btn.isEnabled()
        assert not loaded.fall_clear_excluded_btn.isEnabled()

    @pytest.mark.parametrize("trigger", ["delta", "spacing", "max_latency"])
    def test_redetect_clears_all_exclusions(self, loaded, trigger):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        loaded._fall_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        assert loaded._rise_excluded and loaded._fall_excluded

        graph = loaded.brightness_graph
        if trigger == "delta":
            graph.set_delta(graph._delta)
        elif trigger == "spacing":
            graph.set_min_spacing(graph._min_spacing)
        else:
            graph.set_max_latency(graph._max_latency)

        assert loaded._rise_excluded == set()
        assert loaded._fall_excluded == set()

    def test_new_analyze_clears_all_exclusions(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        assert loaded._rise_excluded
        analyze(loaded, qtbot)
        assert loaded._rise_excluded == set()

    def test_row_click_jumps_to_correct_frame_through_proxy(self, loaded, qtbot):
        """Regression: the results table's model is now a QSortFilterProxyModel,
        which has no .item() -- the row-click handler must resolve the frame
        via index.sibling() so it works both filtered and unfiltered."""
        load_multi_pairs(loaded)
        proxy = loaded.rise_results_table.model()
        index = proxy.index(1, loaded._exclude_col)
        loaded._on_results_row_clicked(index)
        assert loaded.timeline.current_frame == loaded.brightness_graph._rise_pairs[1].orig_frame

        loaded._rise_results_model.item(2, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        qtbot.mouseClick(loaded.rise_show_excluded_btn, Qt.MouseButton.LeftButton)
        assert proxy.rowCount() == 1
        loaded._on_results_row_clicked(proxy.index(0, loaded._exclude_col))
        assert loaded.timeline.current_frame == loaded.brightness_graph._rise_pairs[2].orig_frame

    def test_buttons_disabled_when_nothing_excluded(self, loaded):
        load_multi_pairs(loaded)
        assert not loaded.rise_clear_excluded_btn.isEnabled()
        assert not loaded.rise_show_excluded_btn.isEnabled()

        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)
        assert loaded.rise_clear_excluded_btn.isEnabled()
        assert loaded.rise_show_excluded_btn.isEnabled()

        loaded._rise_results_model.item(0, loaded._exclude_col).setCheckState(Qt.CheckState.Unchecked)
        assert not loaded.rise_clear_excluded_btn.isEnabled()
        assert not loaded.rise_show_excluded_btn.isEnabled()

    def test_export_csv_includes_excluded_column_with_current_state(self, loaded, tmp_path, monkeypatch):
        load_multi_pairs(loaded)
        loaded._rise_results_model.item(1, loaded._exclude_col).setCheckState(Qt.CheckState.Checked)

        out_path = tmp_path / "export.csv"
        monkeypatch.setattr(
            "ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out_path), ""))
        loaded._on_export_csv()

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert lines[0].endswith(",Excluded")
        excluded_frame = str(loaded.brightness_graph._rise_pairs[1].orig_frame)
        matches = [ln for ln in lines[1:] if ln.split(",")[1] == excluded_frame]
        assert len(matches) == 1
        assert matches[0].endswith(",Y")
        assert sum(1 for ln in lines[1:] if ln.endswith(",N")) == 5

    def test_export_csv_follows_results_polarity_not_live_pulldown(self, loaded, tmp_path, monkeypatch):
        load_multi_pairs(loaded)  # pins _results_polarity to "both"
        idx = loaded.polarity_combo.findData("rising")
        loaded.polarity_combo.setCurrentIndex(idx)  # live pulldown now says "rising"

        out_path = tmp_path / "export2.csv"
        monkeypatch.setattr(
            "ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out_path), ""))
        loaded._on_export_csv()

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) - 1 == 6  # 3 rise + 3 fall, not filtered down by the live pulldown


class TestPlayheadRowHighlight:
    _HIGHLIGHT_RGBA = (255, 255, 255, 40)

    def test_landing_on_matched_frame_highlights_correct_row(self, loaded):
        load_multi_pairs(loaded)
        target = loaded.brightness_graph._rise_pairs[1]
        loaded.brightness_graph.set_frame(target.orig_frame)

        item = loaded._rise_results_model.item(1, loaded._orig_frame_col)
        assert item.background().color().getRgb() == self._HIGHLIGHT_RGBA
        # The Exclude column is deliberately skipped (avoids re-firing itemChanged).
        excl_item = loaded._rise_results_model.item(1, loaded._exclude_col)
        assert excl_item.background().style() == Qt.BrushStyle.NoBrush

    def test_moving_away_clears_previous_row_highlight(self, loaded):
        load_multi_pairs(loaded)
        r0 = loaded.brightness_graph._rise_pairs[0]
        r1 = loaded.brightness_graph._rise_pairs[1]
        loaded.brightness_graph.set_frame(r0.orig_frame)
        loaded.brightness_graph.set_frame(r1.orig_frame)

        old_item = loaded._rise_results_model.item(0, loaded._orig_frame_col)
        assert old_item.background().style() == Qt.BrushStyle.NoBrush
        new_item = loaded._rise_results_model.item(1, loaded._orig_frame_col)
        assert new_item.background().color().getRgb() == self._HIGHLIGHT_RGBA

    def test_landing_on_unmatched_frame_clears_highlight(self, loaded):
        load_multi_pairs(loaded)
        r0 = loaded.brightness_graph._rise_pairs[0]
        loaded.brightness_graph.set_frame(r0.orig_frame)
        loaded.brightness_graph.set_frame(0)  # frame 0 is inside the leading dark block -- unmatched

        item = loaded._rise_results_model.item(0, loaded._orig_frame_col)
        assert item.background().style() == Qt.BrushStyle.NoBrush

    def test_highlighting_does_not_change_exclude_state(self, loaded):
        """Regression: recoloring the Exclude column's own checkbox item
        would spuriously re-fire _on_exclude_toggled; skipping that column
        must keep exclude state completely untouched by playhead movement."""
        load_multi_pairs(loaded)
        target = loaded.brightness_graph._rise_pairs[0]
        loaded.brightness_graph.set_frame(target.orig_frame)
        assert loaded._rise_excluded == set()
        assert loaded._fall_excluded == set()

    def test_scroll_suppressed_during_playback(self, loaded, monkeypatch):
        load_multi_pairs(loaded)
        calls = []
        monkeypatch.setattr(
            type(loaded.rise_results_table), "scrollTo",
            lambda self, *a, **k: calls.append(a))
        loaded._playback_timer.start(1000)
        try:
            target = loaded.brightness_graph._rise_pairs[2]
            loaded.brightness_graph.set_frame(target.orig_frame)
        finally:
            loaded._playback_timer.stop()
        assert calls == []

    def test_scroll_happens_when_not_playing(self, loaded, monkeypatch):
        load_multi_pairs(loaded)
        calls = []
        monkeypatch.setattr(
            type(loaded.rise_results_table), "scrollTo",
            lambda self, *a, **k: calls.append(a))
        target = loaded.brightness_graph._rise_pairs[2]
        loaded.brightness_graph.set_frame(target.orig_frame)
        assert len(calls) == 1

    def test_highlight_survives_a_table_repopulate(self, loaded):
        load_multi_pairs(loaded)
        target = loaded.brightness_graph._rise_pairs[1]
        loaded.brightness_graph.set_frame(target.orig_frame)
        assert loaded._rise_results_model.item(1, loaded._orig_frame_col).background().color().getRgb() \
            == self._HIGHLIGHT_RGBA

        loaded._update_results_table()  # e.g. as triggered by an FPS change, Clear All, etc.

        item = loaded._rise_results_model.item(1, loaded._orig_frame_col)
        assert item.background().color().getRgb() == self._HIGHLIGHT_RGBA

    def test_no_crash_when_pair_direction_gated_off_by_pinned_polarity(self, loaded):
        load_multi_pairs(loaded)
        target = loaded.brightness_graph._rise_pairs[0]
        loaded.brightness_graph.set_frame(target.orig_frame)

        loaded._results_polarity = "falling"  # rising is no longer covered
        loaded._update_results_table()  # must not raise


class TestGraphClickToSeek:
    def test_frame_clicked_is_wired_to_show_frame(self, loaded, qtbot):
        """Mirrors timeline.frame_changed's wiring to show_frame -- a click
        on the graph should seek the same way scrubbing the timeline does."""
        analyze(loaded, qtbot)
        target = loaded.brightness_graph._rise_pairs[0].orig_frame
        loaded.brightness_graph.frame_clicked.emit(target)
        assert loaded.timeline.current_frame == target


def _drifting_arrays():
    """Six flash cycles whose dark level climbs steadily, as it would if the
    display slowly moved within an oversized ROI. Amplitude is constant, so
    only the baseline check should fire."""
    out = []
    for c in range(6):
        base = 20.0 + c * 30.0
        out.append(np.full(8, base))
        out.append(np.full(8, base + 200.0))
    arr = np.concatenate(out).astype(np.float64)
    return arr, arr.copy()


def _per_pair_flagged_arrays():
    """Each dark stretch tilts steadily upward instead of sitting flat, so the
    region every transition measures its baseline from is not a level. That
    trips the per-transition `unsteady-level` check.

    Distinct from _drifting_arrays on purpose: under pure drift each transition
    is still locally well-measured, so the SIGNAL is flagged but no individual
    pair is. Here it is the individual measurements that are suspect. The tilt's
    per-frame steps stay well under the auto delta, so detection still finds
    exactly one transition per edge."""
    out = []
    for _ in range(4):
        out.append(np.linspace(20.0, 100.0, 10))
        out.append(np.full(10, 220.0))
    arr = np.concatenate(out).astype(np.float64)
    return arr, arr.copy()


def _asymmetric_ramp_arrays():
    """Mirrors the real LED: one partially-lit frame, then a hard step. The
    steepest single-frame change is the second one, so the anchor lands a frame
    AFTER first-light — on the real clip 2.7 -> 84.2 -> 217.4 gave anchor 1153
    with first-light at 1152. A symmetric ramp cannot show this, because its
    steepest step is its first."""
    def shape(delay):
        d = np.full(40, 20.0)
        d[10 + delay] = 84.0
        d[11 + delay : 26 + delay] = 220.0
        d[26 + delay] = 84.0
        d[27 + delay :] = 20.0
        return d
    return shape(0), shape(3)


def _ramped_arrays():
    """Transitions that take several frames, so the three metrics differ."""
    def shape(delay):
        d = np.full(60, 20.0)
        for start in (10, 34):
            d[start + delay : start + 4 + delay] = [70.0, 120.0, 170.0, 210.0]
            d[start + 4 + delay : start + 12 + delay] = 220.0
            d[start + 12 + delay : start + 16 + delay] = [170.0, 120.0, 70.0, 30.0]
        return d
    return shape(0), shape(3)


class TestThreeMetricColumns:
    def test_row_carries_all_three_metrics(self, loaded):
        orig, disp = _ramped_arrays()
        loaded.brightness_graph.set_data(orig, disp, in_point=0)
        loaded._results_polarity = "both"
        loaded._update_results_table()

        fps = loaded.reader.fps_effective
        pair = loaded.brightness_graph.get_pairs_for("rising", active="both")[0]
        model = loaded._rise_results_model
        headers = [model.headerData(c, Qt.Orientation.Horizontal)
                   for c in range(model.columnCount())]
        row = {h: model.item(0, c).text() for c, h in enumerate(headers)}
        assert row["First (ms)"] == f"{pair.first_delta_ms(fps):.1f}"
        assert row["Avg (ms)"] == f"{pair.avg_delta_ms(fps):.1f}"
        assert row["Full (ms)"] == f"{pair.full_delta_ms(fps):.1f}"

    def test_frame_columns_hold_first_pixel_not_the_anchor(self, loaded):
        """The frame columns used to show the steepest-step anchor, which is an
        internal matching detail — not drawn on the graph, not one of the three
        reported metrics, and not where Up/Down navigation lands. On real
        footage that made the column read 1153 where first-pixel was 1152."""
        orig, disp = _asymmetric_ramp_arrays()
        loaded.brightness_graph.set_data(orig, disp, in_point=0)
        loaded._results_polarity = "both"
        loaded._update_results_table()

        pair = loaded.brightness_graph.get_pairs_for("rising", active="both")[0]
        assert pair.orig_first_frame() != pair.orig_frame,             "fixture must ramp, or this asserts nothing"

        model = loaded._rise_results_model
        headers = [model.headerData(c, Qt.Orientation.Horizontal)
                   for c in range(model.columnCount())]
        row = {h: model.item(0, c).text() for c, h in enumerate(headers)}
        assert row[COL_ORIG_FIRST] == str(pair.orig_first_frame())
        assert row[COL_DISP_FIRST] == str(pair.disp_first_frame())

    def test_headers_do_not_bold_only_the_populated_panel(self, loaded, qtbot):
        """Qt bolds the header section holding the current item, so a panel
        with rows rendered bold while an empty one didn't — reading as a
        deliberate distinction that was never intended."""
        analyze(loaded, qtbot)
        for name in ("rise_results_table", "fall_results_table"):
            header = getattr(loaded, name).horizontalHeader()
            assert not header.highlightSections()

    def test_instantaneous_clip_reports_the_same_number_three_times(self, loaded, qtbot):
        """Compatibility check on the real synthetic clip: square-wave
        transitions make first = full = anchor, so the tool still reports
        exactly what it did before the metrics were split apart."""
        analyze(loaded, qtbot)
        fps = loaded.reader.fps_effective
        pair = loaded.brightness_graph.get_pairs_for("rising")[0]
        expected = f"{SYNTH_LATENCY / fps * 1000.0:.1f}"
        model = loaded._rise_results_model
        headers = [model.headerData(c, Qt.Orientation.Horizontal)
                   for c in range(model.columnCount())]
        row = {h: model.item(0, c).text() for c, h in enumerate(headers)}
        assert row["First (ms)"] == row["Avg (ms)"] == row["Full (ms)"] == expected


class TestQualityBanner:
    def test_hidden_for_a_clean_clip(self, loaded, qtbot):
        analyze(loaded, qtbot)
        assert not loaded.quality_label.isVisible()
        assert loaded.quality_label.text() == ""

    def _load(self, win, arrays):
        orig, disp = arrays
        win.brightness_graph.set_data(orig, disp, in_point=0)
        win._results_polarity = "both"
        win._update_results_table()
        return win.quality_label.text()

    def test_drift_with_clean_measurements_reads_as_information(self, loaded):
        """Regression, and the shape of the real reference clip: the Display
        baseline moved across the whole clip while every pair still measured
        correctly. The banner used to raise a warning and tell the user to
        re-draw their ROI. Both were wrong — a warning on a correct measurement
        is one the user learns to skip, and the tool cannot know the cause."""
        text = self._load(loaded, _drifting_arrays())
        pairs = loaded.brightness_graph.get_pairs_for("rising", active="both")
        assert all(p.is_clean() for p in pairs), "fixture should measure cleanly"

        assert loaded.quality_label.isVisible()
        assert "⚠" not in text
        assert "baseline varies" in text
        assert "auto-exposure" in text  # offered as explanation, not diagnosis

    def test_information_register_never_blames_the_roi(self, loaded):
        """The signature of a moving baseline is identical whether it comes
        from auto-exposure on the device under test, ROI framing, changing
        light or a nudged camera. Asserting one of them sends the user to the
        wrong place — on the reference clip, to re-frame an ROI that measured
        89% participating."""
        text = self._load(loaded, _drifting_arrays()).lower()
        for blame in ("re-draw", "stays inside", "check that each roi"):
            assert blame not in text

    def test_flagged_measurements_read_as_a_warning(self, loaded):
        text = self._load(loaded, _per_pair_flagged_arrays())
        pairs = loaded.brightness_graph.get_pairs_for("rising", active="both")
        assert any(not p.is_clean() for p in pairs), "fixture should flag pairs"

        assert "⚠" in text
        assert "pairs flagged" in text
        # Names the checks that failed, so there is something to act on.
        assert any(w in text for w in ("unsteady-level", "low-snr",
                                       "ambiguous-edge", "slow-ramp"))

    def test_roi_framing_is_only_suggested_for_low_snr(self, loaded):
        """low-snr is the one flag where framing genuinely is implicated: too
        little of the screen inside the box leaves the step in the noise."""
        from core.edges import W_LOW_SNR

        text = self._load(loaded, _per_pair_flagged_arrays())
        pairs = loaded.brightness_graph.get_pairs_for("rising", active="both")
        has_low_snr = any(W_LOW_SNR in p.quality_warnings() for p in pairs)
        assert ("too little of its screen" in text) == has_low_snr

    def test_warning_column_marks_flagged_rows_with_a_tooltip(self, loaded):
        orig, disp = _per_pair_flagged_arrays()
        loaded.brightness_graph.set_data(orig, disp, in_point=0)
        loaded._results_polarity = "both"
        loaded._update_results_table()

        pairs = loaded.brightness_graph.get_pairs_for("rising", active="both")
        flagged = [i for i, p in enumerate(pairs) if not p.is_clean()]
        assert flagged, "expected the drifting signal to flag at least one pair"
        item = loaded._rise_results_model.item(flagged[0], loaded._warn_col)
        assert item.text() == "⚠"
        # Names the specific checks, not a generic "bad data".
        assert item.toolTip().startswith("Measurement quality: ")
        assert len(item.toolTip()) > len("Measurement quality: ")


class TestExcludeFlagged:
    def _load_flagged(self, win):
        orig, disp = _per_pair_flagged_arrays()
        win.brightness_graph.set_data(orig, disp, in_point=0)
        win._results_polarity = "both"
        win._update_results_table()
        return win.brightness_graph.get_pairs_for("rising", active="both")

    def test_button_disabled_when_nothing_is_flagged(self, loaded, qtbot):
        analyze(loaded, qtbot)
        assert not loaded.rise_exclude_flagged_btn.isEnabled()

    def test_excludes_exactly_the_flagged_pairs(self, loaded):
        pairs = self._load_flagged(loaded)
        expected = {i for i, p in enumerate(pairs) if not p.is_clean()}
        assert loaded.rise_exclude_flagged_btn.isEnabled()
        loaded.rise_exclude_flagged_btn.click()
        assert loaded._rise_excluded == expected

    def test_uses_the_same_exclusion_machinery_as_the_checkboxes(self, loaded):
        """No new mechanism: the graph muting, the Clear All button and the CSV
        Excluded column all keep working because this just ticks the boxes."""
        pairs = self._load_flagged(loaded)
        flagged = {i for i, p in enumerate(pairs) if not p.is_clean()}
        loaded.rise_exclude_flagged_btn.click()
        for i in flagged:
            item = loaded._rise_results_model.item(i, loaded._exclude_col)
            assert item.checkState() == Qt.CheckState.Checked
        assert loaded.rise_clear_excluded_btn.isEnabled()
        loaded.rise_clear_excluded_btn.click()
        assert loaded._rise_excluded == set()

    def test_button_disables_once_everything_flagged_is_excluded(self, loaded):
        self._load_flagged(loaded)
        loaded.rise_exclude_flagged_btn.click()
        assert not loaded.rise_exclude_flagged_btn.isEnabled()


class TestEdgeSensitivityControl:
    def test_disabled_until_a_file_is_loaded(self, window):
        assert not window.edge_sigma_spin.isEnabled()

    def test_enabled_after_analysis(self, loaded, qtbot):
        analyze(loaded, qtbot)
        assert loaded.edge_sigma_spin.isEnabled()

    def test_changing_it_reaches_the_graph(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded.edge_sigma_spin.setValue(7.5)
        assert loaded.brightness_graph._sigma_k == pytest.approx(7.5)

    def test_cli_value_is_applied(self, loaded):
        args = SimpleNamespace(
            fps=None, direction=None, roi_original=None, roi_display=None,
            min_delta=None, min_spacing=None, max_latency=None, edge_sigma=5.5,
            in_point=None, out_point=None,
        )
        loaded.apply_cli_args(args)
        assert loaded.edge_sigma_spin.value() == pytest.approx(5.5)

    def test_appears_in_the_generated_cli_command(self, loaded, qtbot):
        analyze(loaded, qtbot)
        loaded.edge_sigma_spin.setValue(4.5)
        assert "--edge-sigma 4.5" in loaded._build_cli_command()
