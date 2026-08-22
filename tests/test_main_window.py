"""
MainWindow integration tests against the synthetic clip (see conftest).

Extraction runs the real QThread; tests wait for the built-in
QThread.finished cleanup slot to clear MainWindow._extractor.
"""

import argparse
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt

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

    def test_results_tables_split_by_polarity(self, loaded, qtbot):
        """Regression: rise table must show only rising pairs, fall table only
        falling pairs — no cross-contamination between the two panels."""
        analyze(loaded, qtbot)
        rise_pairs = loaded.brightness_graph.get_pairs_for("rising")
        fall_pairs = loaded.brightness_graph.get_pairs_for("falling")
        assert loaded._rise_results_model.rowCount() == len(rise_pairs) == 1
        assert loaded._fall_results_model.rowCount() == len(fall_pairs) == 1
        assert int(loaded._rise_results_model.item(0, 0).text()) == rise_pairs[0].orig_frame
        assert int(loaded._fall_results_model.item(0, 0).text()) == fall_pairs[0].orig_frame

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
        expected = ["Original Frame", "Display Frame", "Latency (fr)", "Latency (ms)"]
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

    def test_failed_open_preserves_session(self, loaded, qtbot):
        """Regression: a bad path released the current reader before failing,
        bricking scrubbing and wiping results."""
        analyze(loaded, qtbot)
        frames_before = loaded.reader.frame_count
        loaded.open_file("/nonexistent/nope.mp4")
        assert "Error" in loaded.status_label.text()
        assert loaded.reader.frame_count == frames_before
        assert loaded._rise_results_model.rowCount() == 1
        assert loaded._fall_results_model.rowCount() == 1
        loaded.show_frame(5)  # reader still usable


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
