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
from ui.main_window import MainWindow, _existing_file

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
            assert [model.headerData(c, Qt.Orientation.Horizontal) for c in range(4)] == \
                ["Mean", "Min", "Max", "Median"]
            assert [model.item(0, c).text() for c in range(4)] == ["--.- ms"] * 4

    def test_summary_tables_populated_after_analysis(self, loaded, qtbot):
        analyze(loaded, qtbot)
        fps = loaded.reader.fps_effective
        rise_pairs = loaded.brightness_graph.get_pairs_for("rising")
        fall_pairs = loaded.brightness_graph.get_pairs_for("falling")
        rise_expected = f"{rise_pairs[0].delta_ms(fps):.1f} ms"
        fall_expected = f"{fall_pairs[0].delta_ms(fps):.1f} ms"
        assert [loaded._rise_summary_model.item(0, c).text() for c in range(4)] == [rise_expected] * 4
        assert [loaded._fall_summary_model.item(0, c).text() for c in range(4)] == [fall_expected] * 4

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
        assert [loaded._fall_summary_model.item(0, c).text() for c in range(4)] == ["--.- ms"] * 4

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
        expected = ["Exclude", "Original Frame", "Display Frame", "Latency (fr)", "Latency (ms)"]
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
            min_delta=None, min_spacing=None, max_latency=None,
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
            min_delta=None, min_spacing=None, max_latency=None,
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

        remaining_ms = [p.delta_ms(fps) for p in rise_pairs[:2]]
        assert loaded._rise_summary_model.item(0, 0).text() == \
            f"{sum(remaining_ms) / len(remaining_ms):.1f} ms"
        assert loaded._rise_summary_model.item(0, 2).text() == f"{max(remaining_ms):.1f} ms"

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
