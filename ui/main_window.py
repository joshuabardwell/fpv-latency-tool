"""
Main window: wires the whole app together — video scrubbing with in/out
points, ROI selection with live brightness readout, threaded brightness
extraction, detection-parameter controls, results table, CSV export, and
CLI argument handling.

Keyboard shortcuts are listed in README.md and in the in-app help (F1 / ?).
Architecture and data flow are described in DESIGN.md.
"""

import argparse
import os
import statistics
import sys
from dataclasses import replace
from typing import NamedTuple

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QObject, QSortFilterProxyModel, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.edges import (
    DEFAULT_SIGMA_K,
    W_AMBIGUOUS_EDGE,
    W_LOW_SNR,
    W_SLOW_RAMP,
    W_UNSTEADY_LEVEL,
)
from core.export import write_pairs_csv
from core.extractor import BrightnessExtractor
from core.latency import default_max_latency_frames
from core.manual import EditTarget, ManualEdit, find_edit, set_frame, upsert
from core.roi import ROI
from core.session import SessionState, load_session, save_session, sidecar_path_for
from core.video_io import VideoReader
from ui.brightness_graph import BrightnessGraphWidget
from ui.edit_panel import (
    SelectionInfo,
    TransitionEditPanel,
    polarity_label,
    roi_label,
)
from ui.roi_frame_view import RoiFrameView
from ui.timeline import TimelineWidget
from ui.zoom_bar import ZoomBarWidget


def _release_spinbox_focus() -> None:
    """Deselect and defocus the current focus widget (called via singleShot)."""
    fw = QApplication.focusWidget()
    if fw is not None:
        if hasattr(fw, 'deselect'):
            fw.deselect()
        fw.clearFocus()


class _ReleaseFocusOnCommit(QObject):
    """Event filter installed on the FPS spinbox and its internal QLineEdit.

    QAbstractSpinBox calls selectAll() on its line edit *after* processing
    Enter, so an immediate deselect/clearFocus is overwritten.  We schedule
    the work via singleShot(0) so it runs after the spinbox finishes its own
    key handling.
    """
    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.KeyPress and
                event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape)):
            QTimer.singleShot(0, _release_spinbox_focus)
        return False  # let the widget handle the key normally as well


class _ClickAwayReleasesSpinboxFocus(QObject):
    """App-wide filter: a mouse press anywhere outside the currently-focused
    detection-parameter spinbox releases its focus.

    Every such spinbox uses ClickFocus so its own arrow/Home/End editing
    works while the user is actually in it, but Qt never reclaims that focus
    on its own: the graph, timeline, video preview, results tables and
    buttons are all NoFocus precisely so they don't steal it *back* — so
    nothing ever un-focuses the spinbox, and it goes on eating every
    navigation/editing key (see keyPressEvent) until the user manually Tabs
    away or commits with Enter/Escape.

    Deliberately doesn't key off the clicked widget's own focusPolicy: a
    composite widget's internal parts (e.g. a QAbstractScrollArea's
    viewport) can report a stronger nominal policy than the widget it
    belongs to even though that widget's own overridden mouse handling means
    the policy is never actually acted on -- checking it here would only
    reproduce that inconsistency. Clearing focus unconditionally on
    "clicked something else" is safe: if the click target *does* want focus
    (another spinbox, a combo box), Qt assigns it right after this filter
    returns, overriding the clear.
    """
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            focus = QApplication.focusWidget()
            if isinstance(focus, QAbstractSpinBox):
                target = QApplication.widgetAt(event.globalPosition().toPoint())
                if target is None or (target is not focus and not focus.isAncestorOf(target)):
                    focus.clearFocus()
        return False  # let the click be handled normally as well


# Human-readable rendering of core.edges' warning slugs: a short label for the
# banner, and a sentence saying what was actually seen and why it matters for
# trusting the number. The slugs themselves name the *check*, which tells a
# reader of this code what fired but tells a user of the app nothing.
#
# Keyed by the imported constants so renaming a slug breaks loudly here instead
# of silently falling back to raw text. Only the UI translates: the CSV keeps
# the slugs, which are greppable and stable in a way a sentence is not.
WARNING_TEXT = {
    W_UNSTEADY_LEVEL: (
        "Drifting levels",
        "the brightness either side of this transition was still drifting "
        "rather than holding steady, so the levels this measurement is "
        "compared against are approximate. Common when the device under test "
        "has auto-exposure.",
    ),
    W_LOW_SNR: (
        "Low contrast",
        "the brightness step is small next to the image noise, so the exact "
        "first-lit and fully-lit frames are uncertain. Often means the ROI "
        "holds too little of its screen.",
    ),
    W_AMBIGUOUS_EDGE: (
        "Ambiguous start",
        "brightness crossed the first-light threshold more than once, so "
        "there is more than one candidate for the frame light first appeared. "
        "Usually movement during the transition.",
    ),
    W_SLOW_RAMP: (
        "Never settles",
        "the brightness is still changing when the next transition arrives, "
        "so the fully-lit frame is approximate rather than measured.",
    ),
}


def _warning_label(slug: str) -> str:
    """Short label for the banner. An unrecognised slug falls through unchanged
    rather than being dropped — a warning the user can't parse still beats a
    warning that silently disappears."""
    entry = WARNING_TEXT.get(slug)
    return entry[0] if entry else slug


def _warning_tooltip(warnings) -> str:
    """One line per flag: label, then what was seen and why it matters."""
    lines = []
    for slug in warnings:
        entry = WARNING_TEXT.get(slug)
        lines.append(f"{entry[0]} — {entry[1]}" if entry else slug)
    return "\n".join(lines)


# Two-line headers: the frame columns report first-pixel, not the internal
# steepest-step anchor, and "Original Frame" gave no hint which.
COL_ORIG_FIRST = "Original\n1st Pixel"
COL_DISP_FIRST = "Display\n1st Pixel"

# The three reported metrics, in the order they appear in the summary table.
# Each entry is (row label, LatencyPair accessor taking fps and returning ms).
SUMMARY_METRICS = [
    ("First pixel", "first_delta_ms"),
    ("Average", "avg_delta_ms"),
    ("Full frame", "full_delta_ms"),
]


class ResultsPanel(NamedTuple):
    container: QWidget
    model: QStandardItemModel
    table: QTableView
    proxy: "ExcludedFilterProxy"
    summary_model: QStandardItemModel
    clear_all_btn: QPushButton
    show_excluded_btn: QPushButton
    exclude_flagged_btn: QPushButton


class ExcludedFilterProxy(QSortFilterProxyModel):
    """Filters a results table to only its checked ("Exclude") rows, when
    Show Excluded is on. The exclude column's check state is the only
    exclusion bookkeeping this proxy does — MainWindow owns the actual
    exclude sets; this just decides what's visible."""

    def __init__(self, exclude_col: int, parent=None):
        super().__init__(parent)
        self._exclude_col = exclude_col
        self._show_only = False

    def set_show_only(self, show_only: bool) -> None:
        if show_only != self._show_only:
            self._show_only = show_only
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent) -> bool:
        if not self._show_only:
            return True
        source = self.sourceModel()
        item = source.item(source_row, self._exclude_col)
        return item is not None and item.checkState() == Qt.CheckState.Checked


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame to a QPixmap at full resolution."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # QImage wraps the NumPy buffer without copying; .copy() detaches it so
    # the pixmap never depends on the lifetime of the local array.
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glass-to-Glass Latency Tool")
        self.resize(1400, 750)
        self.show()
        # On multi-monitor Windows setups, showMaximized()'s automatic
        # geometry calculation can be stale/wrong even after show(), leaving
        # a window that reports WindowMaximized without filling the screen.
        # Querying screen() after show() (so it reflects the monitor the
        # window actually landed on) and setting geometry explicitly avoids
        # relying on that calculation.
        self.setWindowState(Qt.WindowState.WindowMaximized)
        screen = self.screen()
        if screen is not None:
            self.setGeometry(screen.availableGeometry())
        self.setAcceptDrops(True)

        self.reader: VideoReader | None = None
        self._current_frame: np.ndarray | None = None
        self._fps_reported: float = 0.0

        self._playback_timer = QTimer(self)

        self._extractor: BrightnessExtractor | None = None
        self._brightness_original: np.ndarray | None = None
        self._brightness_display: np.ndarray | None = None
        self._extraction_in_point: int = 0
        self._extraction_requested: int = 0
        # Monotonic token: bumped whenever extraction state is invalidated
        # (new analysis, ROI change, file open). Worker signals carry the
        # token they were started with; a mismatch means the payload belongs
        # to a session that no longer exists and must be dropped — a queued
        # cross-thread signal cannot be un-sent, only ignored.
        self._extraction_session: int = 0
        self._delta_user_set: bool = False
        self._max_latency_user_set: bool = False
        self._cli_args = None
        # The polarity in effect when Analyze was last clicked — the results
        # panel (which table(s) show, and their rows) is pinned to this until
        # the next Analyze, so it doesn't shift out from under a loaded result
        # set just because the user is browsing the Direction pulldown.
        self._results_polarity: str | None = None

        # Manual outlier exclusion, per direction. Indices are positions into
        # the current rise_pairs/fall_pairs list and are only meaningful
        # until the next redetect — _on_pairs_rebuilt clears both sets (and
        # both show-only flags) whenever pairs_updated fires.
        self._rise_excluded: set[int] = set()
        self._fall_excluded: set[int] = set()
        self._rise_show_excluded_only: bool = False
        self._fall_show_excluded_only: bool = False

        # Signature of the pair list the exclusion sets above were built
        # against: (orig_frame, disp_frame) per pair, per direction. Exclusions
        # are positional, so they are cleared when — and only when — this
        # actually changes. A nudge cannot change it (pairing keys on anchors,
        # which nudging never touches), so a review pass no longer throws away
        # the Exclude ticks it has already made.
        self._pair_signature: tuple | None = None

        # Manual transition edits (core.manual). MainWindow owns the
        # authoritative list — it is what the sidecar persists — and pushes it
        # to the graph, which re-resolves it onto moved anchors and hands the
        # updated list back via manual_edits_rebound.
        self._manual_edits: list[ManualEdit] = []
        self._sidecar_enabled: bool = True
        # Written only once a clip has been analyzed, so merely opening and
        # scrubbing footage never drops files beside it.
        self._analyzed_once: bool = False
        self._sidecar_timer = QTimer(self)
        self._sidecar_timer.setSingleShot(True)
        self._sidecar_timer.setInterval(500)

        # Results-table row highlight tracking the playhead's matched pair
        # (see _apply_playhead_highlight). Independent of exclusion state.
        self._current_playhead_pair = None
        self._highlighted_row: tuple[QStandardItemModel, int] | None = None

        self._build_ui()
        self._wire_events()
        self._set_controls_enabled(False)
        self._update_results_table()

        self._click_away_filter = _ClickAwayReleasesSpinboxFocus()
        QApplication.instance().installEventFilter(self._click_away_filter)

    # ---------------------------------------------------------- UI setup

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        # ── File bar ──────────────────────────────────────────────────────
        file_bar = QHBoxLayout()
        self.open_button = QPushButton("Open Video...")
        self.open_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.show_cli_btn = QPushButton("Show CLI Options")
        self.show_cli_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.show_cli_btn.setToolTip("Show the command-line invocation for the current configuration")
        self.show_cli_btn.setEnabled(False)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray;")
        self.help_button = QPushButton("?")
        self.help_button.setFixedWidth(28)
        self.help_button.setToolTip("Keyboard shortcuts (F1)")
        self.help_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        file_bar.addWidget(self.open_button)
        file_bar.addWidget(self.show_cli_btn)
        file_bar.addWidget(self.file_label, stretch=1)
        file_bar.addWidget(self.help_button)
        layout.addLayout(file_bar)

        # ── Grouped controls row ───────────────────────────────────────────
        controls_row = QHBoxLayout()
        controls_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ROI Selection group
        roi_group = QGroupBox("ROI Selection")
        roi_group_layout = QVBoxLayout(roi_group)

        self.roi_original_btn = QPushButton("Set Original ROI")
        self.roi_original_btn.setCheckable(True)
        self.roi_original_btn.setToolTip("Draw a rectangle on the original-signal screen")
        self.roi_original_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.roi_display_btn = QPushButton("Set Display ROI")
        self.roi_display_btn.setCheckable(True)
        self.roi_display_btn.setToolTip("Draw a rectangle on the delayed-display screen")
        self.roi_display_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_rois_btn = QPushButton("Clear ROIs")
        self.clear_rois_btn.setToolTip("Remove both ROI rectangles")
        self.clear_rois_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        roi_btns = QHBoxLayout()
        roi_btns.addWidget(self.roi_original_btn)
        roi_btns.addWidget(self.roi_display_btn)
        roi_btns.addWidget(self.clear_rois_btn)
        roi_btns.addStretch()

        self.brightness_original_label = QLabel("Original: --")
        self.brightness_original_label.setStyleSheet("color: #00e600; font-weight: bold;")
        self.brightness_display_label = QLabel("Display: --")
        self.brightness_display_label.setStyleSheet("color: #ffa000; font-weight: bold;")
        roi_readout = QHBoxLayout()
        roi_readout.addWidget(self.brightness_original_label)
        roi_readout.addWidget(self.brightness_display_label)
        roi_readout.addStretch()

        roi_group_layout.addLayout(roi_btns)
        roi_group_layout.addLayout(roi_readout)

        # Analysis group
        analysis_group = QGroupBox("Analysis")
        analysis_group_layout = QVBoxLayout(analysis_group)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setToolTip("Extract brightness from both ROIs over the in/out range")
        self.analyze_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.analyze_btn.setEnabled(False)
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItem("Both transitions",  "both")
        self.polarity_combo.addItem("Dark → Light only", "rising")
        self.polarity_combo.addItem("Light → Dark only", "falling")
        self.polarity_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.polarity_combo.setToolTip("Which transition direction to detect and navigate")

        analysis_row = QHBoxLayout()
        analysis_row.addWidget(self.analyze_btn)
        analysis_row.addWidget(self.polarity_combo)
        analysis_row.addStretch()
        analysis_group_layout.addLayout(analysis_row)

        # FPS group
        fps_group = QGroupBox("FPS")
        fps_group_layout = QVBoxLayout(fps_group)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Effective FPS:"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 20000.0)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        _commit_filter = _ReleaseFocusOnCommit(self)
        self.fps_spin.installEventFilter(_commit_filter)
        self.fps_spin.lineEdit().installEventFilter(_commit_filter)
        self.fps_reset_btn = QPushButton("↺")
        self.fps_reset_btn.setFixedWidth(28)
        self.fps_reset_btn.setToolTip("Restore FPS to the value reported by the file")
        self.fps_reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        fps_row.addWidget(self.fps_spin)
        fps_row.addWidget(self.fps_reset_btn)
        fps_row.addStretch()

        self.fps_verify_widget = QWidget()
        fps_verify_inner = QVBoxLayout(self.fps_verify_widget)
        fps_verify_inner.setContentsMargins(0, 0, 0, 0)
        fps_verify_inner.setSpacing(2)
        self.period_meas_label = QLabel("Orig period: --")
        self.period_meas_label.setStyleSheet("color: #00e600;")
        self.known_period_spin = QDoubleSpinBox()
        self.known_period_spin.setRange(10.0, 10000.0)
        self.known_period_spin.setDecimals(0)
        self.known_period_spin.setValue(1000.0)
        self.known_period_spin.setSuffix(" ms")
        self.known_period_spin.setToolTip("The true period of the test pattern (e.g. 1000 ms for a 1 Hz flash)")
        self.known_period_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.computed_fps_label = QLabel("Computed FPS: --")
        known_period_row = QHBoxLayout()
        known_period_row.addWidget(QLabel("Known period:"))
        known_period_row.addWidget(self.known_period_spin)
        known_period_row.addStretch()
        fps_verify_inner.addWidget(self.period_meas_label)
        fps_verify_inner.addLayout(known_period_row)
        fps_verify_inner.addWidget(self.computed_fps_label)
        self.fps_verify_widget.hide()

        fps_group_layout.addLayout(fps_row)
        fps_group_layout.addWidget(self.fps_verify_widget)

        controls_row.addWidget(roi_group)
        controls_row.addWidget(analysis_group)
        controls_row.addWidget(fps_group, stretch=1)
        layout.addLayout(controls_row)

        # ── Analysis progress row (hidden until extraction is running) ────
        self.analysis_widget = QWidget()
        analysis_row = QHBoxLayout(self.analysis_widget)
        analysis_row.setContentsMargins(0, 0, 0, 0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        analysis_row.addWidget(self.cancel_btn)
        analysis_row.addWidget(self.progress_bar, stretch=1)
        self.analysis_widget.hide()
        layout.addWidget(self.analysis_widget)

        # ── Frame display ────────────────────────────────────────────────
        self.frame_view = RoiFrameView()
        layout.addWidget(self.frame_view, stretch=1)

        # ── Detection parameters ──────────────────────────────────────────
        detection_group = QGroupBox("Detection Parameters")
        detection_group_layout = QHBoxLayout(detection_group)

        self.delta_label = QLabel("Min ΔBrightness:")
        self.delta_spin = QSpinBox()
        self.delta_spin.setRange(1, 255)
        self.delta_spin.setValue(10)
        self.delta_spin.setToolTip("Minimum per-frame brightness change to count as a transition")
        self.delta_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.delta_spin.setEnabled(False)

        self.spacing_label = QLabel("Min Spacing:")
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(1, 9999)
        self.spacing_spin.setValue(1)
        self.spacing_spin.setSuffix(" fr")
        self.spacing_spin.setToolTip("Minimum frames between two transitions of the same type on the same signal")
        self.spacing_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.spacing_spin.setEnabled(False)

        self.max_latency_label = QLabel("Max Latency:")
        self.max_latency_spin = QSpinBox()
        self.max_latency_spin.setRange(0, 9999)
        self.max_latency_spin.setValue(0)
        self.max_latency_spin.setSuffix(" fr")
        self.max_latency_spin.setSpecialValueText("unlimited")
        self.max_latency_spin.setToolTip(
            "Maximum frames between Original and Display transition to form a pair (0 = no limit)"
        )
        self.max_latency_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.max_latency_spin.setEnabled(False)

        self.edge_sigma_label = QLabel("Edge Sensitivity:")
        self.edge_sigma_spin = QDoubleSpinBox()
        self.edge_sigma_spin.setRange(0.0, 20.0)
        self.edge_sigma_spin.setSingleStep(0.5)
        self.edge_sigma_spin.setDecimals(1)
        self.edge_sigma_spin.setValue(DEFAULT_SIGMA_K)
        self.edge_sigma_spin.setSuffix(" σ")
        self.edge_sigma_spin.setToolTip(
            "How far above the noise a signal must move to count as changing.\n"
            "Higher = first-light later and fully-lit earlier (a narrower\n"
            "measured transition). Has no effect on noise-free footage."
        )
        self.edge_sigma_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.edge_sigma_spin.setEnabled(False)

        self.max_latency_auto_btn = QPushButton("Auto")
        self.max_latency_auto_btn.setToolTip(
            "Sets Max Latency to 1/2 the measured period of the test signal."
        )
        self.max_latency_auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.max_latency_auto_btn.setEnabled(False)

        detection_group_layout.addWidget(self.delta_label)
        detection_group_layout.addWidget(self.delta_spin)
        detection_group_layout.addWidget(self.spacing_label)
        detection_group_layout.addWidget(self.spacing_spin)
        detection_group_layout.addWidget(self.max_latency_label)
        detection_group_layout.addWidget(self.max_latency_spin)
        detection_group_layout.addWidget(self.max_latency_auto_btn)
        detection_group_layout.addWidget(self.edge_sigma_label)
        detection_group_layout.addWidget(self.edge_sigma_spin)
        detection_group_layout.addStretch()
        layout.addWidget(detection_group)

        # ── Brightness graph ──────────────────────────────────────────────
        self.brightness_graph = BrightnessGraphWidget()
        layout.addWidget(self.brightness_graph)

        # ── Manual transition editing ─────────────────────────────────────
        # Directly under the graph it acts on: reviewing is a loop between the
        # marker, the video frame and this readout, and putting it anywhere
        # else would make that loop cross the window.
        self.edit_panel = TransitionEditPanel()
        layout.addWidget(self.edit_panel)

        # ── Pairs summary ─────────────────────────────────────────────────
        self.pairs_label = QLabel("")
        self.pairs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pairs_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(self.pairs_label)

        # Data-quality banner. Hidden entirely when everything is clean, so its
        # mere presence means something needs looking at.
        self.quality_label = QLabel("")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet("color: #e65ae6; font-size: 11px;")
        self.quality_label.setVisible(False)
        layout.addWidget(self.quality_label)

        # ── Timeline with in/out handles, zoom/pan bar above it ────────────
        # zoom_bar and timeline are stacked in the same QVBoxLayout cell
        # (rather than each spanning the row independently) so Qt gives them
        # identical width — their frame-to-pixel mappings line up exactly,
        # regardless of how wide the flanking nav-button columns are.
        scrub_bar = QHBoxLayout()

        self.prev_trans_button = QPushButton("<< Transition")
        self.prev_trans_button.setToolTip("Jump to previous transition (Up)")
        self.prev_trans_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.prev_trans_button.setEnabled(False)
        self.prev_unmatched_button = QPushButton("<< Unmatched")
        self.prev_unmatched_button.setToolTip("Jump to previous unmatched transition")
        self.prev_unmatched_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.prev_unmatched_button.setEnabled(False)
        self.prev_button = QPushButton("<< Frame")
        self.prev_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Top to bottom: Transition (nearest the graph above), Unmatched,
        # Frame (nearest the timeline below).
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.addWidget(self.prev_trans_button)
        left_col.addWidget(self.prev_unmatched_button)
        left_col.addWidget(self.prev_button)

        self.zoom_bar = ZoomBarWidget()
        self.timeline = TimelineWidget()
        timeline_col = QVBoxLayout()
        timeline_col.setContentsMargins(0, 0, 0, 0)
        timeline_col.setSpacing(2)
        timeline_col.addWidget(self.zoom_bar)
        timeline_col.addWidget(self.timeline)

        self.next_trans_button = QPushButton("Transition >>")
        self.next_trans_button.setToolTip("Jump to next transition (Down)")
        self.next_trans_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_trans_button.setEnabled(False)
        self.next_unmatched_button = QPushButton("Unmatched >>")
        self.next_unmatched_button.setToolTip("Jump to next unmatched transition")
        self.next_unmatched_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_unmatched_button.setEnabled(False)
        self.next_button = QPushButton("Frame >>")
        self.next_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.addWidget(self.next_trans_button)
        right_col.addWidget(self.next_unmatched_button)
        right_col.addWidget(self.next_button)

        scrub_bar.addLayout(left_col)
        scrub_bar.addLayout(timeline_col, stretch=1)
        scrub_bar.addLayout(right_col)
        layout.addLayout(scrub_bar)

        # ── In/out readout ────────────────────────────────────────────────
        self.inout_label = QLabel("In: --    Out: --")
        self.inout_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inout_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.inout_label)

        # ── Frame / time status ───────────────────────────────────────────
        self.status_label = QLabel("Frame: -- / --    Time: -- s")
        layout.addWidget(self.status_label)

        # Wrapped in a scroll area so that when the column's minimum height
        # (video preview + fixed-height widgets stacked with no give)
        # exceeds the available window height, the column scrolls instead
        # of forcing the top-level window to grow past the screen/taskbar.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # NoFocus for the same reason every other widget in this column has it:
        # QAbstractScrollArea handles the arrow keys itself (to scroll its
        # viewport) and defaults to StrongFocus, so once it took click focus it
        # swallowed every navigation key before MainWindow.keyPressEvent could
        # see them. Wheel and scrollbar-drag scrolling do not need focus.
        left_scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ── Results panel (right side) ────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Latency-in-frames is gone from the table: with three metrics it would
        # triple the width for a number nobody reads once ms is present, and the
        # average is a half-frame value that reads badly as an integer. All
        # three still go to the CSV in frames.
        # ✎ sits beside ⚠ rather than replacing it: they answer different
        # questions ("did a person place this" vs "does the signal look
        # trustworthy") and a pair can carry both.
        result_columns = [
            "Exclude", "⚠", "✎", COL_ORIG_FIRST, COL_DISP_FIRST,
            "First (ms)", "Avg (ms)", "Full (ms)",
        ]
        self._exclude_col = result_columns.index("Exclude")
        self._warn_col = result_columns.index("⚠")
        self._manual_col = result_columns.index("✎")
        self._orig_frame_col = result_columns.index(COL_ORIG_FIRST)

        rise_panel = self._build_results_table("Dark To Light Transitions", result_columns)
        self.rise_results_container = rise_panel.container
        self._rise_results_model = rise_panel.model
        self.rise_results_table = rise_panel.table
        self._rise_results_proxy = rise_panel.proxy
        self._rise_summary_model = rise_panel.summary_model
        self.rise_clear_excluded_btn = rise_panel.clear_all_btn
        self.rise_show_excluded_btn = rise_panel.show_excluded_btn
        self.rise_exclude_flagged_btn = rise_panel.exclude_flagged_btn

        fall_panel = self._build_results_table("Light To Dark Transitions", result_columns)
        self.fall_results_container = fall_panel.container
        self._fall_results_model = fall_panel.model
        self.fall_results_table = fall_panel.table
        self._fall_results_proxy = fall_panel.proxy
        self._fall_summary_model = fall_panel.summary_model
        self.fall_clear_excluded_btn = fall_panel.clear_all_btn
        self.fall_show_excluded_btn = fall_panel.show_excluded_btn
        self.fall_exclude_flagged_btn = fall_panel.exclude_flagged_btn

        right_layout.addWidget(self.rise_results_container, 1)
        right_layout.addWidget(self.fall_results_container, 1)

        self.export_csv_btn = QPushButton("Export CSV…")
        self.export_csv_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_csv_btn.setEnabled(False)
        right_layout.addWidget(self.export_csv_btn)

        splitter.addWidget(right_widget)
        splitter.setSizes([800, 600])
        self.setCentralWidget(splitter)

    def _build_results_table(self, title: str, columns: list[str]) -> ResultsPanel:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        container_layout.addWidget(title_label)

        # One row per metric now, not one row total.
        summary_model = QStandardItemModel(len(SUMMARY_METRICS), 5)
        summary_model.setHorizontalHeaderLabels(["Metric", "Mean", "Min", "Max", "Median"])
        summary_table = QTableView()
        summary_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        summary_table.setModel(summary_model)
        summary_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        summary_table.setSelectionMode(QTableView.SelectionMode.NoSelection)
        summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        summary_table.verticalHeader().setVisible(False)
        summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        summary_table.setFixedHeight(
            summary_table.horizontalHeader().height()
            + sum(summary_table.rowHeight(r) for r in range(len(SUMMARY_METRICS)))
            + 2 * summary_table.frameWidth() + 2
        )
        container_layout.addWidget(summary_table)

        model = QStandardItemModel(0, len(columns))
        model.setHorizontalHeaderLabels(columns)
        proxy = ExcludedFilterProxy(self._exclude_col)
        proxy.setSourceModel(model)
        table = QTableView()
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setModel(proxy)
        table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # Off, or Qt bolds whichever header holds the current item — making a
        # populated panel's headers bold and an empty panel's plain, which
        # reads as a deliberate distinction and isn't one.
        table.horizontalHeader().setHighlightSections(False)
        table.verticalHeader().setVisible(False)
        container_layout.addWidget(table)

        button_row = QHBoxLayout()
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_all_btn.setEnabled(False)
        show_excluded_btn = QPushButton("Show Excluded")
        show_excluded_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        show_excluded_btn.setCheckable(True)
        show_excluded_btn.setEnabled(False)
        exclude_flagged_btn = QPushButton("Exclude Flagged")
        exclude_flagged_btn.setToolTip(
            "Tick Exclude on every pair carrying a measurement-quality warning."
        )
        exclude_flagged_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        exclude_flagged_btn.setEnabled(False)
        button_row.addWidget(clear_all_btn)
        button_row.addWidget(show_excluded_btn)
        button_row.addWidget(exclude_flagged_btn)
        button_row.addStretch(1)
        container_layout.addLayout(button_row)

        return ResultsPanel(container, model, table, proxy, summary_model,
                            clear_all_btn, show_excluded_btn, exclude_flagged_btn)

    def _wire_events(self) -> None:
        self.open_button.clicked.connect(self.on_open_file)
        self.prev_button.clicked.connect(lambda: self.timeline.step(-1))
        self.next_button.clicked.connect(lambda: self.timeline.step(1))
        self.fps_spin.valueChanged.connect(self.on_fps_override_changed)
        self.fps_reset_btn.clicked.connect(self._reset_fps)

        self.timeline.frame_changed.connect(self.show_frame)
        self.brightness_graph.frame_clicked.connect(self.show_frame)
        self.timeline.in_point_changed.connect(self._on_in_point_changed)
        self.timeline.out_point_changed.connect(self._on_out_point_changed)

        self.zoom_bar.range_changed.connect(self.brightness_graph.set_visible_range)
        self.brightness_graph.visible_range_changed.connect(self.zoom_bar.set_range)
        self.brightness_graph.domain_changed.connect(self.zoom_bar.set_analysis_bounds)

        self.prev_trans_button.clicked.connect(self._goto_prev_transition)
        self.next_trans_button.clicked.connect(self._goto_next_transition)
        self.prev_unmatched_button.clicked.connect(self._goto_prev_unmatched)
        self.next_unmatched_button.clicked.connect(self._goto_next_unmatched)

        # Navigation keys are handled in keyPressEvent, NOT as QShortcuts:
        # window-context shortcuts intercept keys before the focused widget
        # sees them, which broke arrow/Home/End editing in the spinboxes.
        # Only chords and function keys stay as shortcuts.
        QShortcut(QKeySequence("Ctrl+Z"),        self).activated.connect(self._undo_roi)
        QShortcut(QKeySequence("F1"),            self).activated.connect(self._show_help)
        QShortcut(QKeySequence("?"),             self).activated.connect(self._show_help)

        self._playback_timer.timeout.connect(self._playback_tick)
        self.help_button.clicked.connect(self._show_help)

        self.roi_original_btn.toggled.connect(
            lambda checked: self._on_roi_mode_toggled("original", checked)
        )
        self.roi_display_btn.toggled.connect(
            lambda checked: self._on_roi_mode_toggled("display", checked)
        )
        self.clear_rois_btn.clicked.connect(self._on_clear_rois)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.polarity_combo.currentIndexChanged.connect(self._on_polarity_changed)
        self.delta_spin.valueChanged.connect(self._on_delta_spin_changed)
        self.spacing_spin.valueChanged.connect(lambda v: self.brightness_graph.set_min_spacing(v))
        # No auto-computed default, so this needs none of the one-shot CLI
        # override dance that Min Delta and Max Latency require.
        self.edge_sigma_spin.valueChanged.connect(lambda v: self.brightness_graph.set_sigma_k(v))
        self.max_latency_spin.valueChanged.connect(self._on_max_latency_spin_changed)
        self.max_latency_auto_btn.clicked.connect(self._on_max_latency_auto_clicked)
        self.brightness_graph.pairs_updated.connect(self._update_pairs_label)
        self.brightness_graph.pairs_updated.connect(self._update_export_csv_enabled)
        self.brightness_graph.pairs_updated.connect(self._update_fps_verify_row)
        self.brightness_graph.pairs_updated.connect(self._on_pairs_rebuilt)
        self.brightness_graph.playhead_pair_changed.connect(self._on_playhead_pair_changed)
        self.brightness_graph.selection_changed.connect(self._on_selection_changed)
        # Store-only, deliberately: see _on_manual_edits_rebound.
        self.brightness_graph.manual_edits_rebound.connect(self._on_manual_edits_rebound)

        self.edit_panel.nudge_requested.connect(self._nudge_selected)
        self.edit_panel.set_to_playhead_requested.connect(self._set_selected_to_playhead)
        self.edit_panel.delete_toggled.connect(self._delete_selected)
        self.edit_panel.reset_requested.connect(self._reset_selected)
        self.edit_panel.reset_all_requested.connect(self._on_reset_all_clicked)

        # Sidecar persistence. The detection spinboxes need no entry here: they
        # all drive a redetect, and _on_pairs_rebuilt schedules a save at the
        # end of one. These are the settings that change without one.
        self._sidecar_timer.timeout.connect(self._save_sidecar)
        self.fps_spin.valueChanged.connect(lambda _: self._schedule_sidecar_save())
        self.polarity_combo.currentIndexChanged.connect(lambda _: self._schedule_sidecar_save())
        self.frame_view.roi_changed.connect(lambda *_: self._schedule_sidecar_save())
        self.timeline.in_point_changed.connect(lambda _: self._schedule_sidecar_save())
        self.timeline.out_point_changed.connect(lambda _: self._schedule_sidecar_save())
        self.known_period_spin.valueChanged.connect(self._update_fps_verify_row)
        self.show_cli_btn.clicked.connect(self._on_show_cli)
        self.export_csv_btn.clicked.connect(self._on_export_csv)
        self.rise_results_table.clicked.connect(self._on_results_row_clicked)
        self.fall_results_table.clicked.connect(self._on_results_row_clicked)
        self._rise_results_model.itemChanged.connect(
            lambda item: self._on_exclude_toggled("rising", item))
        self._fall_results_model.itemChanged.connect(
            lambda item: self._on_exclude_toggled("falling", item))
        self.rise_clear_excluded_btn.clicked.connect(lambda: self._on_clear_excluded("rising"))
        self.fall_clear_excluded_btn.clicked.connect(lambda: self._on_clear_excluded("falling"))
        self.rise_exclude_flagged_btn.clicked.connect(lambda: self._on_exclude_flagged("rising"))
        self.fall_exclude_flagged_btn.clicked.connect(lambda: self._on_exclude_flagged("falling"))
        self.rise_show_excluded_btn.toggled.connect(
            lambda checked: self._on_toggle_show_excluded("rising", checked))
        self.fall_show_excluded_btn.toggled.connect(
            lambda checked: self._on_toggle_show_excluded("falling", checked))
        self.frame_view.roi_changed.connect(self._on_roi_changed)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.timeline,
            self.prev_button,
            self.next_button,
            self.fps_spin,
            self.fps_reset_btn,
            self.roi_original_btn,
            self.roi_display_btn,
            self.clear_rois_btn,
        ):
            w.setEnabled(enabled)

    # ------------------------------------------------------------ actions

    def on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v);;All Files (*)",
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str) -> None:
        """Load a video file by path (used by the dialog and by CLI argument)."""
        # Open the new file before tearing anything down: a bad path must
        # leave the current session (reader, results) fully intact.
        try:
            new_reader = VideoReader(path)
        except (FileNotFoundError, IOError) as e:
            self.status_label.setText(f"Error: {e}")
            QMessageBox.warning(self, "Failed to Open Video", str(e))
            return

        self._playback_timer.stop()
        self._stop_extractor()
        self._clear_brightness()
        self._delta_user_set = False
        self._max_latency_user_set = False
        if self.reader is not None:
            self.reader.release()
        self.reader = new_reader

        meta = self.reader.metadata
        self.file_label.setText(meta.path.name)

        self._fps_reported = meta.fps_reported if meta.fps_reported > 0 else 30.0
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(self._fps_reported)
        self.fps_spin.blockSignals(False)
        self.reader.fps_effective = self.fps_spin.value()

        self.timeline.reset(meta.frame_count)
        self.zoom_bar.reset(meta.frame_count)
        self._reset_roi_state()
        self._set_controls_enabled(True)
        self.show_cli_btn.setEnabled(True)
        self._load_sidecar()
        self._update_analyze_button()
        self._update_inout_label()
        self.show_frame(0)

    # ------------------------------------------------------------- sidecar

    def set_sidecar_enabled(self, enabled: bool) -> None:
        """Turn the per-clip settings file off entirely (--no-sidecar). Must be
        called before open_file to suppress a read."""
        self._sidecar_enabled = enabled

    def _load_sidecar(self) -> None:
        """Restore this clip's saved settings and manual edits, if it has any.

        Deliberately does NOT start an analysis. Extraction is a full decode
        pass, and kicking one off unasked would lock the UI for as long as the
        clip is long every time the file is opened; the ROIs and every
        parameter come back, and Analyze is one click away."""
        self._manual_edits = []
        self.brightness_graph.set_manual_edits([])
        self._analyzed_once = False
        if not self._sidecar_enabled or self.reader is None:
            return
        state = load_session(self.reader.metadata.path)
        if state is None:
            return
        self._apply_settings(state, source="sidecar")
        # A restored threshold is a decision, not a default: without these the
        # auto-compute in _on_extract_finished would overwrite it on Analyze.
        if state.min_delta is not None:
            self._delta_user_set = True
        if state.max_latency is not None:
            self._max_latency_user_set = True
        self._manual_edits = list(state.manual_edits)
        self.brightness_graph.set_manual_edits(self._manual_edits)
        self.status_label.setText(
            f"Restored settings from {sidecar_path_for(self.reader.metadata.path).name}"
            + (f" ({len(self._manual_edits)} manual edits)" if self._manual_edits else "")
        )

    def _current_session_state(self) -> SessionState:
        roi_orig = self.frame_view.get_roi("original")
        roi_disp = self.frame_view.get_roi("display")

        def as_tuple(roi):
            return None if roi is None else (roi.x, roi.y, roi.width, roi.height)

        return SessionState(
            fps=self.fps_spin.value(),
            roi_original=as_tuple(roi_orig),
            roi_display=as_tuple(roi_disp),
            direction=self.polarity_combo.currentData(),
            min_delta=self.delta_spin.value(),
            min_spacing=self.spacing_spin.value(),
            edge_sigma=self.edge_sigma_spin.value(),
            max_latency=self.max_latency_spin.value(),
            in_point=self.timeline.in_point,
            out_point=self.timeline.out_point,
            manual_edits=list(self._manual_edits),
        )

    def _schedule_sidecar_save(self) -> None:
        """Debounced: nudging holds down Shift+→ and every repeat would
        otherwise be a file write."""
        if self._sidecar_enabled and self._analyzed_once and self.reader is not None:
            self._sidecar_timer.start()

    def _save_sidecar(self) -> None:
        if not self._sidecar_enabled or not self._analyzed_once or self.reader is None:
            return
        try:
            path = save_session(self.reader.metadata.path, self._current_session_state())
        except OSError as e:
            # Read-only media, a full disk, a network share that went away. The
            # measurement is unaffected, so say so and carry on rather than
            # interrupting a review pass with a dialog.
            self.status_label.setText(f"Could not save settings: {e}")
            return
        self.status_label.setText(f"Settings saved to {path.name}")

    def on_fps_override_changed(self, value: float) -> None:
        if self.reader is not None:
            self.reader.fps_effective = value
            self.show_frame(self.timeline.current_frame)
            self._update_inout_label()
            self._update_pairs_label()
            self._update_results_table()
            self._update_fps_verify_row()

    def _reset_fps(self) -> None:
        if self._fps_reported > 0:
            self.fps_spin.setValue(self._fps_reported)

    def show_frame(self, index: int) -> None:
        if self.reader is None:
            return
        try:
            frame = self.reader.read_frame(index)
        except (IndexError, IOError) as e:
            self.status_label.setText(f"Error reading frame {index}: {e}")
            return

        self._current_frame = frame
        h, w = frame.shape[:2]
        self.frame_view.set_frame(bgr_to_qpixmap(frame), w, h)
        self._update_brightness()

        # Keep timeline playhead in sync when show_frame is called directly
        # (e.g. from on_open_file or on_fps_override_changed)
        self.timeline.set_frame(index)
        self.brightness_graph.set_frame(index)

        ts = self.reader.frame_to_timestamp(index)
        total = self.reader.frame_count
        self.status_label.setText(f"Frame: {index} / {total - 1}    Time: {ts:.4f} s")

    # ----------------------------------------------------------- ROI handlers

    def _on_roi_mode_toggled(self, name: str, checked: bool) -> None:
        if checked:
            other = self.roi_display_btn if name == "original" else self.roi_original_btn
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            self.frame_view.draw_mode = name
        else:
            self.frame_view.draw_mode = None

    def _cancel_running_extraction(self) -> None:
        """An in-flight extraction samples ROIs that no longer exist —
        cancel it, and bump the session so even an already-queued result
        from it is dropped on delivery."""
        if self._extractor is not None:
            self._extractor.cancel()
            self._extraction_session += 1

    def _on_clear_rois(self) -> None:
        self.frame_view.clear_rois()
        self.frame_view.draw_mode = None
        for btn in (self.roi_original_btn, self.roi_display_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._cancel_running_extraction()
        self._update_brightness()
        self._clear_brightness()
        self._update_analyze_button()

    def _on_roi_changed(self, name: str, roi) -> None:
        self._cancel_running_extraction()
        self._update_brightness()
        self._clear_brightness()
        self._update_analyze_button()

    def _reset_roi_state(self) -> None:
        self.frame_view.clear_rois()
        self.frame_view.draw_mode = None
        self._current_frame = None
        for btn in (self.roi_original_btn, self.roi_display_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self.brightness_original_label.setText("Original: --")
        self.brightness_display_label.setText("Display: --")

    def _update_brightness(self) -> None:
        frame = self._current_frame
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        for name, label in (
            ("original", self.brightness_original_label),
            ("display",  self.brightness_display_label),
        ):
            roi = self.frame_view.get_roi(name)
            if roi is not None and roi.is_valid():
                b = roi.clipped(fw, fh).mean_brightness(frame)
                label.setText(f"{name.capitalize()}: {b:.1f}")
            else:
                label.setText(f"{name.capitalize()}: --")

    # ---------------------------------------------------- drag-and-drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        local_files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if not local_files:
            return
        event.acceptProposedAction()
        self.open_file(local_files[0])

    # ------------------------------------------------- keyboard shortcuts

    def keyPressEvent(self, event) -> None:
        """Navigation and editing keys, reached only when no focused widget
        consumed them (a focused spinbox keeps its own arrow/Home/End handling).

        The editing layer sits on Shift+arrows plus M and Delete, all of which
        were free: this method has only ever dispatched on unmodified keys.
        Vertically the keys choose WHICH point (first-light / fully-lit),
        horizontally they MOVE it. Plain arrows stay pure playhead stepping —
        they are how the user checks a marker against the footage, and taking
        them would cost the loop the whole feature exists for."""
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if mods == Qt.KeyboardModifier.ShiftModifier:
            shift_handler = {
                Qt.Key.Key_Left:  lambda: self._nudge_selected(-1),
                Qt.Key.Key_Right: lambda: self._nudge_selected(1),
                Qt.Key.Key_Up:    lambda: self._select_end("first"),
                Qt.Key.Key_Down:  lambda: self._select_end("full"),
            }.get(event.key())
            if shift_handler is not None:
                shift_handler()
                event.accept()
                return
        handlers = {
            Qt.Key.Key_Left:     lambda: self.timeline.step(-1),
            Qt.Key.Key_Right:    lambda: self.timeline.step(1),
            Qt.Key.Key_Up:       self._goto_prev_transition,
            Qt.Key.Key_Down:     self._goto_next_transition,
            Qt.Key.Key_PageUp:   self._step_large_back,
            Qt.Key.Key_PageDown: self._step_large_fwd,
            Qt.Key.Key_I:        self._mark_in,
            Qt.Key.Key_O:        self._mark_out,
            Qt.Key.Key_Home:     self._goto_in,
            Qt.Key.Key_End:      self._goto_out,
            Qt.Key.Key_Space:    self._toggle_playback,
            Qt.Key.Key_Escape:   self._on_escape,
            # M for "mark", matching I and O marking the in and out points.
            Qt.Key.Key_M:        self._set_selected_to_playhead,
            Qt.Key.Key_Delete:   self._delete_selected,
        }
        handler = handlers.get(event.key())
        plain = not mods
        if handler is not None and plain:
            handler()
            event.accept()
            return
        super().keyPressEvent(event)

    def _mark_in(self) -> None:
        if self.reader is not None:
            self.timeline.set_in_point(self.timeline.current_frame)

    def _mark_out(self) -> None:
        if self.reader is not None:
            self.timeline.set_out_point(self.timeline.current_frame)

    def _goto_in(self) -> None:
        if self.reader is not None:
            self.show_frame(self.timeline.in_point)

    def _goto_out(self) -> None:
        if self.reader is not None:
            self.show_frame(self.timeline.out_point)

    def _goto_prev_transition(self) -> None:
        self._goto_transition(self.brightness_graph.prev_transition)

    def _goto_next_transition(self) -> None:
        self._goto_transition(self.brightness_graph.next_transition)

    def _goto_transition(self, find) -> None:
        """Walk to the next/previous transition AND select the marker there.

        Selecting is what turns Up/Down into a review pass: without it every
        transition would need a mouse trip to the graph before it could be
        nudged, which is the one flow this is all for."""
        if self.reader is None:
            return
        frame = find(self.timeline.current_frame)
        if frame is not None:
            self.show_frame(frame)
            self.brightness_graph.select_frame(frame)

    def _goto_prev_unmatched(self) -> None:
        if self.reader is None:
            return
        frame = self.brightness_graph.prev_unmatched(self.timeline.current_frame)
        if frame is not None:
            self.show_frame(frame)

    def _goto_next_unmatched(self) -> None:
        if self.reader is None:
            return
        frame = self.brightness_graph.next_unmatched(self.timeline.current_frame)
        if frame is not None:
            self.show_frame(frame)

    def _step_large_back(self) -> None:
        if self.reader is not None:
            self.timeline.step(-max(1, round(self.reader.fps_effective)))

    def _step_large_fwd(self) -> None:
        if self.reader is not None:
            self.timeline.step(max(1, round(self.reader.fps_effective)))

    def _toggle_playback(self) -> None:
        if self.reader is None:
            return
        if self._playback_timer.isActive():
            self._playback_timer.stop()
        else:
            # If parked at or past the out point, restart from in point
            if self.timeline.current_frame >= self.timeline.out_point:
                self.show_frame(self.timeline.in_point)
            interval = max(1, round(1000 / self.reader.fps_effective))
            self._playback_timer.start(interval)

    def _playback_tick(self) -> None:
        self.timeline.step(1)
        if self.timeline.current_frame >= self.timeline.out_point:
            self._playback_timer.stop()

    def _undo_roi(self) -> None:
        # Same invalidation as a normal ROI edit — the restored ROIs make
        # any existing extraction stale.
        if self.frame_view.undo_roi():
            self._cancel_running_extraction()
            self._update_brightness()
            self._clear_brightness()
            self._update_analyze_button()

    def _show_help(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "NAVIGATION\n"
            "Left / Right         Step one frame\n"
            "Up / Down            Previous / next transition (also selects it)\n"
            "PgUp / PgDn          Jump ~1 second (≈ fps frames)\n"
            "Space                Play / pause\n"
            "I                    Set in point at playhead\n"
            "O                    Set out point at playhead\n"
            "Home                 Jump playhead to in point\n"
            "End                  Jump playhead to out point\n"
            "\n"
            "CORRECTING A TRANSITION\n"
            "Shift+Left / Right   Move the selected marker one frame\n"
            "Shift+Up / Down      Select first-light / fully-lit\n"
            "M                    Move the selected marker to the playhead\n"
            "Delete               Delete the selected transition / restore it\n"
            "Esc                  Clear the selection\n"
            "\n"
            "Walk the transitions with Up/Down, step either side of a marker\n"
            "with Left/Right to check it against the video, and correct it with\n"
            "the keys above. Corrections survive every parameter change and are\n"
            "saved beside the clip.\n"
            "\n"
            "OTHER\n"
            "Ctrl+Z               Undo last ROI change\n"
            "F1  /  ?             Show this help",
        )

    # -------------------------------------------------------- in/out handlers

    def _on_in_point_changed(self, frame: int) -> None:
        self._update_inout_label()

    def _on_out_point_changed(self, frame: int) -> None:
        self._update_inout_label()

    def _update_inout_label(self) -> None:
        if self.reader is None:
            self.inout_label.setText("In: --    Out: --")
            return
        in_f  = self.timeline.in_point
        out_f = self.timeline.out_point
        in_t  = self.reader.frame_to_timestamp(in_f)
        out_t = self.reader.frame_to_timestamp(out_f)
        self.inout_label.setText(
            f"In: {in_f} ({in_t:.4f} s)    Out: {out_f} ({out_t:.4f} s)"
        )

    # ------------------------------------------------------------ extraction

    def _update_analyze_button(self) -> None:
        running = self._extractor is not None and self._extractor.isRunning()
        roi_ok = (
            self.frame_view.get_roi("original") is not None
            and self.frame_view.get_roi("original").is_valid()
            and self.frame_view.get_roi("display") is not None
            and self.frame_view.get_roi("display").is_valid()
        )
        self.analyze_btn.setEnabled(
            self.reader is not None and roi_ok and not running
        )

    def _on_analyze_clicked(self) -> None:
        if self.reader is None:
            return
        roi_orig = self.frame_view.get_roi("original")
        roi_disp = self.frame_view.get_roi("display")
        if roi_orig is None or roi_disp is None:
            return

        self._results_polarity = self.polarity_combo.currentData()
        # Clicking Analyze is an explicit "start this measurement over", so it
        # drops the exclusions unconditionally — even if the run happens to
        # produce byte-identical pairs, which the signature comparison in
        # _on_pairs_rebuilt would otherwise treat as nothing having changed.
        self._pair_signature = None

        # A replaced-but-still-running worker would be garbage collected
        # while its thread is alive (hard crash) — make sure it is done.
        if self._extractor is not None:
            self._extractor.cancel()
            self._extractor.wait()

        meta = self.reader.metadata
        self._extraction_requested = (
            self.timeline.out_point - self.timeline.in_point + 1
        )
        self._extractor = BrightnessExtractor(
            path=str(meta.path),
            in_point=self.timeline.in_point,
            out_point=self.timeline.out_point,
            roi_original=roi_orig,
            roi_display=roi_disp,
            frame_w=meta.width,
            frame_h=meta.height,
        )
        self._extraction_session += 1
        sess = self._extraction_session
        self._extractor.progress.connect(
            lambda done, total, s=sess: self._on_extract_progress(done, total, s))
        self._extractor.extraction_done.connect(
            lambda o, d, f, s=sess: self._on_extract_finished(o, d, f, s))
        self._extractor.error.connect(
            lambda msg, s=sess: self._on_extract_error(msg, s))
        # Built-in QThread.finished: fires when the thread has actually
        # exited, on every path (completed, cancelled, errored) — the one
        # place Analyze can safely be re-enabled.
        self._extractor.finished.connect(self._on_extractor_thread_exit)

        self.progress_bar.setValue(0)
        self.analysis_widget.show()
        self.analyze_btn.setEnabled(False)
        self._extractor.start()

    def _on_cancel_clicked(self) -> None:
        # Bumps the session too: a result that finished in the instant
        # before the click is already queued and must not land after it.
        self._cancel_running_extraction()
        self.analysis_widget.hide()

    def _on_extractor_thread_exit(self) -> None:
        if self.sender() is not self._extractor:
            return  # stale notification from an already-replaced worker
        self._extractor = None
        self.analysis_widget.hide()
        self._update_analyze_button()

    def _on_extract_progress(self, done: int, total: int, session: int | None = None) -> None:
        if session is not None and session != self._extraction_session:
            return
        pct = round(100 * done / total) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"Analyzing… {done} / {total} frames ({pct}%)")

    def _on_extract_finished(
        self, orig: np.ndarray, disp: np.ndarray, first_frame: int,
        session: int | None = None,
    ) -> None:
        # A queued delivery from an invalidated session (different file
        # opened, ROIs changed, analysis restarted) would populate results
        # that belong to state which no longer exists.
        if session is not None and session != self._extraction_session:
            return
        self._brightness_original = orig
        self._brightness_display = disp
        self._extraction_in_point = first_frame
        auto_delta = self.brightness_graph.set_data(orig, disp, first_frame)
        if self._cli_args is not None and self._cli_args.min_delta is not None:
            # CLI value applies to the first analysis only; afterwards it is
            # the user's spinbox that rules.
            effective_delta = float(self._cli_args.min_delta)
            self._cli_args.min_delta = None
            self._delta_user_set = True
            self.brightness_graph.set_delta(effective_delta)
        elif self._delta_user_set:
            effective_delta = float(self.delta_spin.value())
            self.brightness_graph.set_delta(effective_delta)
        else:
            # Round the auto value and push it back so the integer shown in
            # the spinbox is exactly the threshold in effect.
            effective_delta = float(int(round(auto_delta)))
            self.brightness_graph.set_delta(effective_delta)
        self.delta_spin.blockSignals(True)
        self.delta_spin.setValue(int(effective_delta))
        self.delta_spin.blockSignals(False)
        self.delta_spin.setEnabled(True)

        period_fr = self.brightness_graph.get_orig_period_frames("both")
        auto_max_latency = default_max_latency_frames(period_fr)
        if self._cli_args is not None and self._cli_args.max_latency is not None:
            # CLI value applies to the first analysis only; afterwards it is
            # the user's spinbox that rules.
            effective_max_latency = int(self._cli_args.max_latency)
            self._cli_args.max_latency = None
            self._max_latency_user_set = True
        elif self._max_latency_user_set:
            effective_max_latency = self.max_latency_spin.value()
        else:
            effective_max_latency = auto_max_latency
        self.max_latency_spin.blockSignals(True)
        self.max_latency_spin.setValue(effective_max_latency)
        self.max_latency_spin.blockSignals(False)
        self.brightness_graph.set_max_latency(effective_max_latency)

        self.spacing_spin.setEnabled(True)
        self.edge_sigma_spin.setEnabled(True)
        self.max_latency_spin.setEnabled(True)
        self.max_latency_auto_btn.setEnabled(True)
        self.prev_trans_button.setEnabled(True)
        self.next_trans_button.setEnabled(True)
        self.prev_unmatched_button.setEnabled(True)
        self.next_unmatched_button.setEnabled(True)
        self.analysis_widget.hide()
        self.fps_verify_widget.show()
        # Snapshot the results panel now, at the true end of analysis — set_delta
        # / set_max_latency above may have re-run pairing since set_data(); this
        # is the final pairs state, gated by the polarity captured on click.
        self._update_results_table()
        count = len(orig)
        if count < self._extraction_requested:
            self.status_label.setText(
                f"Analysis stopped early — file ended: {count} of "
                f"{self._extraction_requested} frames extracted "
                f"(frames {first_frame}–{first_frame + count - 1})"
            )
        else:
            self.status_label.setText(
                f"Analysis complete — {count} frames extracted "
                f"(frames {first_frame}–{first_frame + count - 1})"
            )
        # From here on this clip's settings are worth keeping. Gating on a
        # completed analysis is what stops a sidecar appearing beside footage
        # somebody only opened and scrubbed through.
        self._analyzed_once = True
        self._schedule_sidecar_save()

    def _on_results_row_clicked(self, index) -> None:
        value = index.sibling(index.row(), self._orig_frame_col).data()
        if value is not None and self.reader is not None:
            frame = int(value)
            self.show_frame(frame)
            # The table is a way into the review, not just a readout: clicking
            # a row arrives with that transition selected and ready to nudge.
            self.brightness_graph.select_frame(frame)

    # ------------------------------------------- manual transition editing

    def _set_manual_edits(self, edits: list[ManualEdit], message: str | None = None) -> None:
        """The one place the edit list is replaced. Pushing to the graph re-runs
        the pipeline, which emits pairs_updated and refreshes everything
        downstream of it."""
        self._manual_edits = list(edits)
        self.brightness_graph.set_manual_edits(self._manual_edits)
        self._refresh_edit_panel()
        self._schedule_sidecar_save()
        if message:
            self.status_label.setText(message)

    def _on_manual_edits_rebound(self, edits) -> None:
        """A redetect re-resolved the edits onto moved anchors. STORE ONLY —
        pushing them back would re-enter the redetect that produced them."""
        self._manual_edits = list(edits)

    def _on_selection_changed(self, _target) -> None:
        self._refresh_edit_panel()

    def _refresh_edit_panel(self) -> None:
        if hasattr(self, "edit_panel"):
            self.edit_panel.show_selection(self._selection_info(), len(self._manual_edits))

    def _selection_info(self) -> SelectionInfo | None:
        graph = self.brightness_graph
        target = graph.selection()
        if target is None:
            return None
        deleted = graph.is_deleted(target)
        edit = find_edit(
            self._manual_edits, target.roi, target.polarity, target.anchor_frame
        )
        frames = graph.resolved_frames(target)
        if frames is None:
            # No edge: either the transition was deleted (its anchor is dropped
            # before characterization runs) or it could not be characterized at
            # all. Either way the marker is drawn at the anchor, and the panel
            # has to agree with the ring rather than go blank.
            first = target.anchor_frame if edit is None or edit.first_frame is None else edit.first_frame
            full = first if edit is None or edit.full_frame is None else edit.full_frame
            frames = (first, full)
        auto_first, auto_full = graph.auto_frames(target) or (None, None)
        position, total = graph.transition_position(target)
        return SelectionInfo(
            roi=target.roi,
            polarity=target.polarity,
            which=target.which,
            first_frame=frames[0],
            full_frame=frames[1],
            auto_first=auto_first,
            auto_full=auto_full,
            deleted=deleted,
            has_edit=edit is not None,
            position=position,
            total=total,
            edit_count=len(self._manual_edits),
        )

    def _require_selection(self) -> EditTarget | None:
        target = self.brightness_graph.selection()
        if target is None:
            self.status_label.setText(
                "Select a transition marker first — click one, or press ↑/↓"
            )
        return target

    def _edit_for(self, target: EditTarget) -> ManualEdit:
        """This transition's edit, or a fresh empty one to build on."""
        return find_edit(
            self._manual_edits, target.roi, target.polarity, target.anchor_frame
        ) or ManualEdit(target.roi, target.polarity, target.anchor_frame)

    def _place_selected(self, value: int) -> None:
        target = self._require_selection()
        if target is None:
            return
        graph = self.brightness_graph
        edge = graph.edge_for(target)
        if edge is None:
            self.status_label.setText(
                "That transition has no measured extent, so there is nothing to move."
            )
            return
        lo, hi = graph.analysis_bounds()
        before = graph.resolved_frames(target)
        self._set_manual_edits(
            upsert(self._manual_edits, set_frame(self._edit_for(target), edge, target.which, value, lo, hi))
        )
        after = graph.resolved_frames(target)
        if after is None:
            return
        moved = after[0] if target.which == "first" else after[1]
        # The playhead follows the marker: the whole point of the loop is to
        # look at the frame you just decided on.
        self.show_frame(moved)
        idx = 0 if target.which == "first" else 1
        name = "first-light" if idx == 0 else "fully-lit"
        other = "fully-lit" if idx == 0 else "first-light"
        was = before[idx] if before is not None else moved
        # Say so when the push carried the other end along, or it would move
        # invisibly — the two can share a frame, where the graph draws only one
        # marker (see core.manual.set_frame).
        detail = (
            f" (pushed {other} to {after[1 - idx]})"
            if before is not None and before[1 - idx] != after[1 - idx]
            else ""
        )
        self.status_label.setText(
            f"{roi_label(target.roi)} {polarity_label(target.polarity)} "
            f"{name} {was} → {moved}{detail}"
        )

    def _nudge_selected(self, delta: int) -> None:
        target = self._require_selection()
        if target is None:
            return
        frames = self.brightness_graph.resolved_frames(target)
        if frames is None:
            self.status_label.setText(
                "That transition has no measured extent, so there is nothing to move."
            )
            return
        current = frames[0] if target.which == "first" else frames[1]
        self._place_selected(current + delta)

    def _set_selected_to_playhead(self) -> None:
        self._place_selected(self.timeline.current_frame)

    def _select_end(self, which: str) -> None:
        """Shift+↑/↓: first-light is the earlier end, fully-lit the later one,
        so up/down map to them directly rather than toggling — pressing the same
        key twice must not walk back to where it started."""
        target = self._require_selection()
        if target is None:
            return
        if target.which != which:
            self.brightness_graph.set_selection(replace(target, which=which))
        selected = self.brightness_graph.selection()
        if selected is None:
            return  # the transition went away under us; nothing to look at
        frame = self.brightness_graph.marker_frame(selected)
        if frame is not None:
            self.show_frame(frame)

    def _delete_selected(self) -> None:
        target = self._require_selection()
        if target is None:
            return
        edit = self._edit_for(target)
        restoring = edit.deleted
        label = f"{roi_label(target.roi)} {polarity_label(target.polarity)} transition"
        self._set_manual_edits(
            upsert(self._manual_edits, replace(edit, deleted=not restoring)),
            f"Restored {label}" if restoring
            else f"Deleted {label} — it no longer takes part in pairing",
        )

    def _reset_selected(self) -> None:
        target = self._require_selection()
        if target is None:
            return
        remaining = [
            e for e in self._manual_edits
            if (e.roi, e.polarity, e.anchor_frame) != target.transition
        ]
        if len(remaining) == len(self._manual_edits):
            return
        self._set_manual_edits(remaining, "Restored the measured transition")
        selected = self.brightness_graph.selection()
        if selected is None:
            return
        frame = self.brightness_graph.marker_frame(selected)
        if frame is not None:
            self.show_frame(frame)

    def _on_reset_all_clicked(self) -> None:
        if not self._manual_edits:
            return
        answer = QMessageBox.question(
            self, "Discard manual edits",
            f"Discard all {len(self._manual_edits)} manual edits on this clip?\n"
            "Every transition goes back to what the algorithm measured.",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            self._reset_all_edits()

    def _reset_all_edits(self) -> None:
        count = len(self._manual_edits)
        if not count:
            return
        self._set_manual_edits(
            [], f"Discarded {count} manual edit" + ("" if count == 1 else "s")
        )

    def _on_escape(self) -> None:
        """Cancel wins while an extraction is running; otherwise Escape means
        "never mind" about the selection."""
        if self._extractor is not None and self._extractor.isRunning():
            self._on_cancel_clicked()
        else:
            self.brightness_graph.clear_selection()

    def _update_export_csv_enabled(self) -> None:
        self.export_csv_btn.setEnabled(bool(self.brightness_graph.get_pairs()))

    def _current_direction_pairs(self) -> tuple[list, list, float]:
        """(rise_pairs, fall_pairs, fps) for the polarity pinned at the last
        Analyze click — the same gating _update_results_table uses."""
        mode = self._results_polarity
        fps = self.reader.fps_effective if self.reader else 30.0
        rise_pairs = self.brightness_graph.get_pairs_for("rising", active=mode) if mode else []
        fall_pairs = self.brightness_graph.get_pairs_for("falling", active=mode) if mode else []
        return rise_pairs, fall_pairs, fps

    def _pairs_signature(self) -> tuple:
        """What the positional exclusion indices actually depend on: which pairs
        exist, in which order, per direction."""
        rise_pairs, fall_pairs, _ = self._current_direction_pairs()
        return tuple(
            tuple((p.orig_frame, p.disp_frame) for p in pairs)
            for pairs in (rise_pairs, fall_pairs)
        )

    def _on_pairs_rebuilt(self) -> None:
        """pairs_updated fired because the pipeline ran again. Exclusions are
        positions in the pairs list, so they are dropped when the list of pairs
        itself changed — and only then.

        Comparing the pair list rather than reacting to the signal matters now
        that a marker can be nudged mid-review: a nudge re-runs the pipeline but
        provably cannot re-pair anything (pairing keys on anchors, which nudging
        never touches), so wiping the user's Exclude ticks on every keypress
        would make the two features unusable together. A deletion, a threshold
        change that moves anchors, or a fresh Analyze all do change the list and
        still clear, which is what the rule was protecting against."""
        signature = self._pairs_signature()
        if signature != self._pair_signature:
            self._pair_signature = signature
            self._rise_excluded.clear()
            self._fall_excluded.clear()
            self._rise_show_excluded_only = False
            self._fall_show_excluded_only = False
        self._update_results_table()
        self._refresh_edit_panel()
        self._schedule_sidecar_save()

    def _sync_exclude_controls(self, direction: str) -> None:
        rise_pairs, fall_pairs, _ = self._current_direction_pairs()
        if direction == "rising":
            excluded, show_only = self._rise_excluded, self._rise_show_excluded_only
            clear_btn, show_btn = self.rise_clear_excluded_btn, self.rise_show_excluded_btn
            flagged_btn, pairs = self.rise_exclude_flagged_btn, rise_pairs
        else:
            excluded, show_only = self._fall_excluded, self._fall_show_excluded_only
            clear_btn, show_btn = self.fall_clear_excluded_btn, self.fall_show_excluded_btn
            flagged_btn, pairs = self.fall_exclude_flagged_btn, fall_pairs
        has_excluded = bool(excluded)
        clear_btn.setEnabled(has_excluded)
        show_btn.setEnabled(has_excluded)
        show_btn.blockSignals(True)
        show_btn.setChecked(show_only)
        show_btn.blockSignals(False)
        # Enabled by there being something left to exclude, not by there being
        # something already excluded — unlike the two buttons above.
        flagged_btn.setEnabled(
            any(i not in excluded and not p.is_clean() for i, p in enumerate(pairs))
        )

    def _on_exclude_toggled(self, direction: str, item: QStandardItem) -> None:
        if item.column() != self._exclude_col:
            return
        excluded = self._rise_excluded if direction == "rising" else self._fall_excluded
        idx = item.row()  # source-model row == pair index; the proxy only filters, never reorders
        if item.checkState() == Qt.CheckState.Checked:
            excluded.add(idx)
        else:
            excluded.discard(idx)

        rise_pairs, fall_pairs, fps = self._current_direction_pairs()
        self.brightness_graph.set_excluded_pairs(self._rise_excluded, self._fall_excluded)
        if direction == "rising":
            included = [p for i, p in enumerate(rise_pairs) if i not in self._rise_excluded]
            self._populate_summary_model(self._rise_summary_model, included, fps)
        else:
            included = [p for i, p in enumerate(fall_pairs) if i not in self._fall_excluded]
            self._populate_summary_model(self._fall_summary_model, included, fps)
        self._sync_exclude_controls(direction)

    def _on_clear_excluded(self, direction: str) -> None:
        if direction == "rising":
            self._rise_excluded.clear()
            self._rise_show_excluded_only = False
        else:
            self._fall_excluded.clear()
            self._fall_show_excluded_only = False
        self._update_results_table()

    def _on_toggle_show_excluded(self, direction: str, checked: bool) -> None:
        if direction == "rising":
            self._rise_show_excluded_only = checked
        else:
            self._fall_show_excluded_only = checked
        self._update_results_table()

    def _update_results_table(self) -> None:
        """Repopulates both rise/fall panels (table + summary) for whatever
        polarity was in effect at the last Analyze click (self._results_polarity),
        not the live Direction pulldown — call this only when results actually
        change (analysis completes, or results are invalidated), never as a
        reaction to the pulldown alone. Both panels are always visible; a
        direction with no matching pairs just shows blank/placeholder rows."""
        rise_pairs, fall_pairs, fps = self._current_direction_pairs()

        self.brightness_graph.set_excluded_pairs(self._rise_excluded, self._fall_excluded)
        self._rise_results_proxy.set_show_only(self._rise_show_excluded_only)
        self._fall_results_proxy.set_show_only(self._fall_show_excluded_only)

        rise_included = [p for i, p in enumerate(rise_pairs) if i not in self._rise_excluded]
        fall_included = [p for i, p in enumerate(fall_pairs) if i not in self._fall_excluded]

        self._populate_results_model(self._rise_results_model, rise_pairs, fps, self._rise_excluded)
        self._populate_results_model(self._fall_results_model, fall_pairs, fps, self._fall_excluded)
        self._populate_summary_model(self._rise_summary_model, rise_included, fps)
        self._populate_summary_model(self._fall_summary_model, fall_included, fps)

        self._sync_exclude_controls("rising")
        self._sync_exclude_controls("falling")
        # Driven from here rather than straight off pairs_updated: the banner
        # counts flagged pairs over the PINNED polarity, and that is not set
        # until Analyze, so firing on the signal alone left it stale.
        self._update_quality_label()

        self._apply_playhead_highlight()

    def _on_playhead_pair_changed(self, pair) -> None:
        self._current_playhead_pair = pair
        self._apply_playhead_highlight()

    def _apply_playhead_highlight(self) -> None:
        """Tints the results-table row for the playhead's current matched
        pair (see BrightnessGraphWidget.playhead_pair_changed) and scrolls it
        into view unless playback is running. Idempotent and safe to call
        after any table repopulate — always clears first, then re-resolves
        self._current_playhead_pair fresh against the current table, so it
        self-heals regardless of whether playhead_pair_changed or a table
        repopulate happened first."""
        self._clear_playhead_highlight()
        pair = self._current_playhead_pair
        if pair is None:
            return
        rise_pairs, fall_pairs, _ = self._current_direction_pairs()
        if pair.polarity == "rising":
            model, proxy, table, pairs = (
                self._rise_results_model, self._rise_results_proxy, self.rise_results_table, rise_pairs)
        else:
            model, proxy, table, pairs = (
                self._fall_results_model, self._fall_results_proxy, self.fall_results_table, fall_pairs)
        if pair not in pairs:
            return  # e.g. polarity-gated off from the pinned _results_polarity
        row = pairs.index(pair)
        for col in range(model.columnCount()):
            if col == self._exclude_col:
                continue  # avoid spuriously re-firing _on_exclude_toggled via itemChanged
            item = model.item(row, col)
            if item is not None:
                item.setBackground(QBrush(QColor(255, 255, 255, 40)))
        self._highlighted_row = (model, row)
        if not self._playback_timer.isActive():
            proxy_index = proxy.mapFromSource(model.index(row, 0))
            if proxy_index.isValid():
                table.scrollTo(proxy_index)

    def _clear_playhead_highlight(self) -> None:
        if self._highlighted_row is None:
            return
        model, row = self._highlighted_row
        if row < model.rowCount():
            for col in range(model.columnCount()):
                if col == self._exclude_col:
                    continue
                item = model.item(row, col)
                if item is not None:
                    item.setData(None, Qt.ItemDataRole.BackgroundRole)
        self._highlighted_row = None

    def _populate_results_model(
        self, model: QStandardItemModel, pairs: list, fps: float, excluded: set[int]
    ) -> None:
        model.setRowCount(0)
        for idx, p in enumerate(pairs):
            exclude_item = QStandardItem()
            exclude_item.setCheckable(True)
            exclude_item.setCheckState(
                Qt.CheckState.Checked if idx in excluded else Qt.CheckState.Unchecked)
            exclude_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            warnings = p.quality_warnings()
            warn_item = QStandardItem("⚠" if warnings else "")
            warn_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if warnings:
                # Explain what was seen, not which check fired — "unsteady-level"
                # names the code, not the problem.
                warn_item.setToolTip(_warning_tooltip(warnings))
                warn_item.setForeground(QBrush(QColor(230, 90, 230)))

            manual_ends = p.manual_ends()
            manual_item = QStandardItem("✎" if manual_ends else "")
            manual_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if manual_ends:
                manual_item.setToolTip(
                    "Placed by hand: "
                    + " and ".join(manual_ends)
                    + ". Reset it from the Transition Editing panel."
                )
                manual_item.setForeground(QBrush(QColor(255, 255, 255)))

            model.appendRow([
                exclude_item,
                warn_item,
                manual_item,
                QStandardItem(str(p.orig_first_frame())),
                QStandardItem(str(p.disp_first_frame())),
                QStandardItem(f"{p.first_delta_ms(fps):.1f}"),
                QStandardItem(f"{p.avg_delta_ms(fps):.1f}"),
                QStandardItem(f"{p.full_delta_ms(fps):.1f}"),
            ])

    def _populate_summary_model(self, model: QStandardItemModel, pairs: list, fps: float) -> None:
        for row, (label, accessor) in enumerate(SUMMARY_METRICS):
            model.setItem(row, 0, QStandardItem(label))
            if not pairs:
                values = ["--.- ms"] * 4
            else:
                latencies = [getattr(p, accessor)(fps) for p in pairs]
                values = [
                    f"{sum(latencies) / len(latencies):.1f} ms",
                    f"{min(latencies):.1f} ms",
                    f"{max(latencies):.1f} ms",
                    f"{statistics.median(latencies):.1f} ms",
                ]
            for col, text in enumerate(values, start=1):
                model.setItem(row, col, QStandardItem(text))

    def _on_export_csv(self) -> None:
        rise_pairs, fall_pairs, fps = self._current_direction_pairs()
        combined = sorted(
            [(p, i in self._rise_excluded) for i, p in enumerate(rise_pairs)]
            + [(p, i in self._fall_excluded) for i, p in enumerate(fall_pairs)],
            key=lambda t: t[0].orig_frame,
        )
        if not combined:
            return
        pairs = [p for p, _ in combined]
        flags = [f for _, f in combined]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        write_pairs_csv(path, pairs, fps, excluded_flags=flags)

    def _update_quality_label(self) -> None:
        """Report signal shape and measurement trust in two distinct registers,
        and never assert a cause.

        The per-ROI checks answer "what shape is this signal?", not "is anything
        wrong". A baseline that moves across the clip is normal when the device
        under test has auto-exposure — the Display ROI shows that camera's
        image, so its AE opens up through every dark stretch. Framing, changing
        light and a nudged camera produce the identical signature, so the tool
        cannot tell them apart and must not pretend to. It reports what it
        measured and leaves the cause to the person who set the shot up.

        Whether that reads as information or as a warning is decided by the
        per-transition flags, which DO answer "is this number trustworthy". On
        the reference clip every pair measured correctly while the per-ROI check
        fired, and warning there taught nothing except to ignore the banner.
        """
        # Pinned polarity, matching the results tables and the CSV — counting
        # over the live pulldown instead would let the banner report pairs the
        # tables aren't showing.
        rise_pairs, fall_pairs, _ = self._current_direction_pairs()
        pairs = rise_pairs + fall_pairs
        orig_quality, disp_quality = self.brightness_graph.get_signal_quality()
        flagged = [p for p in pairs if not p.is_clean()]

        observations: list[str] = []
        for name, quality in (("Original", orig_quality), ("Display", disp_quality)):
            if quality.unstable_baseline:
                observations.append(f"{name} baseline varies across the clip")
            if quality.inconsistent_amplitude:
                observations.append(f"{name} contrast varies between transitions")

        if not observations and not flagged:
            self.quality_label.setVisible(False)
            self.quality_label.setText("")
            return

        bits: list[str] = []
        if flagged:
            # Labels, not slugs, and lower-cased because they sit mid-sentence.
            checks = sorted(
                {_warning_label(w).lower()
                 for p in flagged for w in p.quality_warnings()}
            )
            bits.append(
                f"⚠ {len(flagged)} of {len(pairs)} pairs flagged: "
                + ", ".join(checks) + "."
            )
        if observations:
            # The explanation belongs to the observation, not to the register —
            # an em-dash ties it to "contrast varies" rather than letting it
            # read as dismissing the ⚠ above, which is about something else.
            # Offered as the common explanation, never as a diagnosis.
            sentence = "; ".join(observations)
            bits.append(
                sentence[:1].upper() + sentence[1:]
                + " — normal when the device under test has auto-exposure, "
                "and compensated for."
            )
        # The only flag where ROI framing genuinely is implicated: too little of
        # the screen inside the box leaves the step buried in the noise.
        if any(W_LOW_SNR in p.quality_warnings() for p in flagged):
            bits.append(
                "Low contrast can mean an ROI holds too little of its screen."
            )

        self.quality_label.setText(" ".join(bits))
        self.quality_label.setStyleSheet(
            "color: %s; font-size: 11px;" % ("#e65ae6" if flagged else "#9a9a9a")
        )
        self.quality_label.setVisible(True)

    def _on_exclude_flagged(self, direction: str) -> None:
        """Tick the Exclude box on every flagged pair. Deliberately a button
        rather than automatic: a flag is a prompt to look, not a verdict, and
        silently dropping data from a measurement would be worse than reporting
        it with a warning attached."""
        rise_pairs, fall_pairs, _ = self._current_direction_pairs()
        pairs = rise_pairs if direction == "rising" else fall_pairs
        excluded = self._rise_excluded if direction == "rising" else self._fall_excluded
        excluded.update(i for i, p in enumerate(pairs) if not p.is_clean())
        # Same repopulate path the Clear All button uses; it re-applies the
        # exclusion sets to the graph and both summaries on its own.
        self._update_results_table()

    def _update_pairs_label(self) -> None:
        pairs = self.brightness_graph.get_pairs()
        unmatched_orig, unmatched_disp = self.brightness_graph.get_unmatched_counts()
        if not pairs and unmatched_orig == 0 and unmatched_disp == 0:
            self.pairs_label.setText("")
            return
        text = f"{len(pairs)} pairs"
        parts = []
        if unmatched_orig:
            parts.append(f"{unmatched_orig} unmatched orig")
        if unmatched_disp:
            parts.append(f"{unmatched_disp} unmatched disp")
        if parts:
            text += "  |  " + ", ".join(parts)
        self.pairs_label.setText(text)

    def _update_fps_verify_row(self) -> None:
        polarity = self.polarity_combo.currentData()
        period_fr = self.brightness_graph.get_orig_period_frames(polarity)
        fps = self.fps_spin.value()
        if period_fr is not None:
            period_ms = period_fr / fps * 1000.0
            self.period_meas_label.setText(
                f"Orig period: {period_fr:.1f} fr = {period_ms:.1f} ms"
            )
            known_ms = self.known_period_spin.value()  # spin range floor is 10, never 0
            computed = period_fr / (known_ms / 1000.0)
            self.computed_fps_label.setText(f"Computed FPS: {computed:.3f}")
        else:
            self.period_meas_label.setText("Orig period: -- (need ≥2 transitions)")
            self.computed_fps_label.setText("Computed FPS: --")

    def apply_cli_args(self, args) -> None:
        """CLI arguments are applied AFTER any sidecar the opened clip had, so
        a flag the user actually typed wins and every flag they omitted comes
        from the sidecar. main() relies on that ordering: open_file first, this
        second."""
        self._cli_args = args
        if getattr(args, "no_sidecar", False):
            self._sidecar_enabled = False
        self._apply_settings(args)

    def _apply_settings(self, args, source: str = "cli") -> None:
        """Apply a settings bundle — a CLI namespace or a restored
        SessionState, which carry the same field names precisely so this is the
        only place that knows what any of them mean. A None field means "says
        nothing about this" and leaves the control alone.

        `source` only names things in the warnings: a clamped value restored
        from a settings file must not be reported as a bad command line."""
        def opt(name: str) -> str:
            return f"--{name}" if source == "cli" else f"the saved {name}"

        if args.fps is not None and self.reader is not None:
            self.fps_spin.setValue(args.fps)
        if args.direction is not None:
            idx = self.polarity_combo.findData(args.direction)
            if idx >= 0:
                self.polarity_combo.setCurrentIndex(idx)
        warnings: list[str] = []
        oob: list[str] = []
        for name, arg in (("original", args.roi_original), ("display", args.roi_display)):
            if arg is None:
                continue
            x, y, w, h = arg
            self.frame_view.set_roi(name, ROI(x, y, w, h))
            if self.reader is not None:
                meta = self.reader.metadata
                if x < 0 or y < 0 or x + w > meta.width or y + h > meta.height:
                    oob.append(name)
        if oob and self.reader is not None:
            # A CLI ROI from a higher-res recording clips to a sliver at the
            # frame edge — possibly sampling the wrong screen with no visual
            # hint. Warn loudly instead of failing silently.
            meta = self.reader.metadata
            warnings.append(
                f"{' and '.join(opt(f'roi-{n}') for n in oob)} extends outside the "
                f"video frame ({meta.width}x{meta.height}) and will be clipped — "
                f"check the ROI overlay before trusting results"
            )
        if args.min_delta is not None:
            self.delta_spin.setValue(args.min_delta)
        if args.min_spacing is not None:
            self.spacing_spin.setValue(args.min_spacing)
        if args.edge_sigma is not None:
            self.edge_sigma_spin.setValue(args.edge_sigma)
        if args.max_latency is not None:
            self.max_latency_spin.setValue(args.max_latency)
        if args.in_point is not None:
            self.timeline.set_in_point(args.in_point)
            if self.timeline.in_point != args.in_point:
                warnings.append(
                    f"{opt('in-point')} {args.in_point} is out of range and was "
                    f"clamped to {self.timeline.in_point}"
                )
        if args.out_point is not None:
            self.timeline.set_out_point(args.out_point)
            if self.timeline.out_point != args.out_point:
                # set_out_point floors at in_point + 1, so a conflicting
                # --in-point/--out-point pair used to clamp silently to a
                # different range than requested, with no indication.
                warnings.append(
                    f"{opt('out-point')} {args.out_point} conflicts with the in "
                    f"point and was clamped to {self.timeline.out_point}"
                )
        if warnings:
            self.status_label.setText("Warning: " + "; ".join(warnings))

    def _build_cli_command(self) -> str:
        parts = ["python main.py"]
        if self.reader is not None:
            parts.append(f'"{self.reader.metadata.path}"')
        parts.append(f"--fps {self.fps_spin.value():.3f}")
        for name in ("original", "display"):
            roi = self.frame_view.get_roi(name)
            if roi is not None:
                parts.append(f"--roi-{name} {roi.x},{roi.y},{roi.width},{roi.height}")
        parts.append(f"--direction {self.polarity_combo.currentData()}")
        # Before the first analysis the spinbox holds its widget default,
        # which was never an effective threshold — printing it would force a
        # threshold the current session never used. Only a value that has
        # actually applied (post-analysis, or user-set) reproduces the run.
        if self.delta_spin.isEnabled() or self._delta_user_set:
            parts.append(f"--min-delta {self.delta_spin.value()}")
        parts.append(f"--min-spacing {self.spacing_spin.value()}")
        parts.append(f"--edge-sigma {self.edge_sigma_spin.value():g}")
        parts.append(f"--max-latency {self.max_latency_spin.value()}")
        parts.append(f"--in-point {self.timeline.in_point}")
        parts.append(f"--out-point {self.timeline.out_point}")
        if not self._sidecar_enabled:
            parts.append("--no-sidecar")
        return " ".join(parts)

    def _on_show_cli(self) -> None:
        cmd = self._build_cli_command()

        dlg = QDialog(self)
        dlg.setWindowTitle("CLI Options")
        dlg.resize(780, 110)
        dlg_layout = QVBoxLayout(dlg)
        text_edit = QPlainTextEdit(cmd)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        dlg_layout.addWidget(text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        copy_btn = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(cmd))
        buttons.accepted.connect(dlg.accept)
        dlg_layout.addWidget(buttons)
        dlg.exec()

    def _on_polarity_changed(self, index: int) -> None:
        mode = self.polarity_combo.itemData(index)
        self.brightness_graph.set_polarity(mode)
        self._update_analyze_button()

    def _on_delta_spin_changed(self, value: int) -> None:
        # A user-chosen threshold survives re-analysis; only an untouched
        # spinbox gets the auto-computed value (programmatic updates go
        # through blockSignals and don't land here).
        self._delta_user_set = True
        self.brightness_graph.set_delta(float(value))

    def _on_max_latency_spin_changed(self, value: int) -> None:
        # Same pattern as _on_delta_spin_changed: a user-chosen cap survives
        # re-analysis; only an untouched spinbox gets the auto-computed value.
        self._max_latency_user_set = True
        self.brightness_graph.set_max_latency(value)

    def _on_max_latency_auto_clicked(self) -> None:
        period_fr = self.brightness_graph.get_orig_period_frames("both")
        value = default_max_latency_frames(period_fr)
        # Always mark as a deliberate override, even if the computed value
        # happens to match what's already shown (setValue alone wouldn't
        # emit valueChanged, and _max_latency_user_set must still flip).
        self._max_latency_user_set = True
        self.max_latency_spin.blockSignals(True)
        self.max_latency_spin.setValue(value)
        self.max_latency_spin.blockSignals(False)
        self.brightness_graph.set_max_latency(value)

    def _on_extract_error(self, msg: str, session: int | None = None) -> None:
        if session is not None and session != self._extraction_session:
            return  # stale error from an invalidated session
        self.analysis_widget.hide()
        self.status_label.setText(f"Analysis error: {msg}")

    def _clear_brightness(self) -> None:
        self._brightness_original = None
        self._brightness_display = None
        self._extraction_in_point = 0
        if hasattr(self, "brightness_graph"):
            self.brightness_graph.clear_data()
        if hasattr(self, "prev_trans_button"):
            self.prev_trans_button.setEnabled(False)
            self.next_trans_button.setEnabled(False)
            self.prev_unmatched_button.setEnabled(False)
            self.next_unmatched_button.setEnabled(False)
        if hasattr(self, "delta_spin"):
            self.delta_spin.setEnabled(False)
        if hasattr(self, "spacing_spin"):
            self.spacing_spin.setEnabled(False)
        if hasattr(self, "max_latency_spin"):
            self.max_latency_spin.setEnabled(False)
        if hasattr(self, "max_latency_auto_btn"):
            self.max_latency_auto_btn.setEnabled(False)
        if hasattr(self, "pairs_label"):
            self.pairs_label.setText("")
        if hasattr(self, "fps_verify_widget"):
            self.fps_verify_widget.hide()
        if hasattr(self, "_rise_results_model"):
            self._results_polarity = None
            self._update_results_table()
        if hasattr(self, "export_csv_btn"):
            self.export_csv_btn.setEnabled(False)

    def _stop_extractor(self) -> None:
        self._extraction_session += 1  # drop anything already queued
        if self._extractor is not None and self._extractor.isRunning():
            self._extractor.cancel()
            self._extractor.wait()
        self._extractor = None
        self.analysis_widget.hide()

    # ----------------------------------------------------------------- close

    def closeEvent(self, event) -> None:
        self._playback_timer.stop()
        self._stop_extractor()
        if self.reader is not None:
            self.reader.release()
        QApplication.instance().removeEventFilter(self._click_away_filter)
        event.accept()


def _parse_roi(s: str):
    try:
        parts = [int(p) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ROI must be x,y,w,h — got: {s!r}")


def _existing_file(s: str) -> str:
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f"video file not found: {s!r}")
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Glass-to-Glass Latency Tool")
    parser.add_argument("file",           nargs="?", type=_existing_file, help="Video file to open")
    parser.add_argument("--fps",          type=float,        metavar="FLOAT")
    parser.add_argument("--roi-original", type=_parse_roi,   metavar="x,y,w,h")
    parser.add_argument("--roi-display",  type=_parse_roi,   metavar="x,y,w,h")
    parser.add_argument("--direction",    choices=["both", "rising", "falling"])
    parser.add_argument("--min-delta",    type=int,          metavar="BRIGHTNESS")
    parser.add_argument("--min-spacing",  type=int,          metavar="FRAMES")
    parser.add_argument("--edge-sigma",   type=float,        metavar="SIGMAS")
    parser.add_argument("--max-latency",  type=int,          metavar="FRAMES")
    parser.add_argument("--in-point",     type=int,          metavar="FRAME")
    parser.add_argument("--out-point",    type=int,          metavar="FRAME")
    parser.add_argument(
        "--no-sidecar", action="store_true",
        help="Don't read or write the <video>.latency.json settings file",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # Before open_file, which is what reads the sidecar — apply_cli_args runs
    # too late to suppress a read that has already happened.
    window.set_sidecar_enabled(not args.no_sidecar)
    if args.file:
        window.open_file(args.file)
    # After open_file, so a flag the user typed overrides the restored value
    # and an omitted flag leaves it alone.
    window.apply_cli_args(args)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
