"""
BrightnessGraphWidget — pyqtgraph plot showing mean-brightness traces for the
Original (green) and Display (amber) ROIs, with derivative-based transition
detection, transition pairing, and a yellow playhead.

Transition detection:
  - Per-frame brightness *change* (np.diff) is compared against a delta threshold.
  - Consecutive frames that all exceed the threshold are collapsed into one event
    at the frame of steepest change.
  - Delta is auto-computed on set_data() (10% of combined brightness range, min 5)
    and can be overridden via set_delta().

Pairing:
  - Each Original transition is paired with the nearest following Display transition
    of the same polarity (greedy, one-to-one).
  - Paired markers keep their original color (green / amber).
  - Unmatched markers are overlaid in red.
  - Connector lines link paired orig→disp markers at the top of the plot area.

Transition markers:
  ▲ triangle-up   = dark → light (rising)
  ▼ triangle-down = light → dark (falling)
"""

import math
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPainterPath

from core.detection import apply_min_spacing, find_falling, find_rising
from core.edges import DEFAULT_SIGMA_K, SignalQuality, TransitionEdge, characterize_signal
from core.latency import LatencyPair, pair_transitions
from core.manual import (
    EditTarget,
    ManualEdit,
    apply_overrides,
    delete_anchors,
    rebind,
)
from core.view_range import MIN_ZOOM_FRAMES, center_range, clamp_range, pan_range, zoom_range

_WHEEL_ZOOM_IN  = 0.85          # per-notch scale factor when zooming in
_WHEEL_ZOOM_OUT = 1.0 / _WHEEL_ZOOM_IN
_HOVER_HIT_R_PX = 10            # hover hit-test radius around a matched marker
_CLICK_DRAG_THRESHOLD_PX = 4    # net displacement since press below which a
                                 # release counts as a click, not a drag

_GREEN     = (0, 230, 0)
_AMBER     = (255, 160, 0)
_YELLOW    = (255, 204, 0)
_RED       = (220, 50, 50)
_HIGHLIGHT = (255, 255, 255)

# Darker tints for transition markers, distinct from the line colors above
# so a marker doesn't disappear into a noisy/jittery line of the same hue.
_GREEN_MARKER = (0, 140, 0)
_AMBER_MARKER = (194, 120, 0)

# Muted gray for a manually-excluded (but still matched) transition marker/connector.
_MUTED_MARKER = (110, 110, 110)

# Outline for a transition whose measurement failed a quality check. An
# OUTLINE rather than a fill because every fill is already spoken for
# (green/amber signal, red unmatched, gray excluded) and an outline composes
# with all of them instead of competing. Magenta because it is the one hue not
# already carrying a meaning here — red would read as "unmatched".
_WARN_OUTLINE = (230, 90, 230)

# Midpoint dot: the (first + full) / 2 position the average latency is measured
# at. Dimmer and smaller than the two real markers on purpose — it is a derived
# midpoint, not a frame anything was observed at, and must not be mistaken for
# one.
_MID_MARKER = (150, 150, 150)
_MID_MARKER_SIZE = 5

# Outline for a marker the user has placed by hand (core.manual). Like the
# warning outline above it composes with any fill, and it wins over the warning
# outline when a marker would carry both: a hand-placed edge has already had the
# question the warning was asking answered by a person looking at the footage.
_MANUAL_OUTLINE = (255, 255, 255)

# A transition the user has deleted. Still drawn — a deletion the user cannot
# see is one they cannot check or undo — but as an inert cross rather than a
# triangle, and dim enough not to compete with the transitions that count.
_DELETED_MARKER = (120, 120, 120)

# The selection ring. Cyan is the one hue not already carrying a meaning here:
# white is the matched-pair highlight, magenta the quality warning, red
# unmatched, gray excluded/deleted.
_SELECT_RING = (0, 210, 255)

_PLAYHEAD_SYMBOL_SIZE    = 16  # ScatterPlotItem `size=` -- must be >= 2x the
                               # farthest path point from the anchor (7px here)
                               # or pyqtgraph's own sprite canvas clips it.
_PLAYHEAD_TRI_BASE_PX    = 10  # matches TimelineWidget's playhead triangle base
_PLAYHEAD_TRI_H_PX       = 7   # matches TimelineWidget's playhead triangle height
_PLAYHEAD_EDGE_MARGIN_PX = 2   # antialiasing safety gap above the ViewBox's
                               # rendered bottom edge (belt-and-braces on top of
                               # the structural anchor-at-base fix)


def _make_playhead_symbol() -> QPainterPath:
    """Upward-pointing triangle whose local origin (0, 0) is its BASE
    midpoint, not its centroid, so the stalk can start at the apex instead
    of passing through the triangle body. Sized to match
    TimelineWidget._draw_playhead's 10x7px triangle."""
    half_w = (_PLAYHEAD_TRI_BASE_PX / _PLAYHEAD_SYMBOL_SIZE) / 2
    h = _PLAYHEAD_TRI_H_PX / _PLAYHEAD_SYMBOL_SIZE
    path = QPainterPath()
    path.moveTo(-half_w, 0.0)   # base, left
    path.lineTo(half_w, 0.0)    # base, right
    path.lineTo(0.0, -h)        # apex -- negative local y renders "up" on screen
    path.closeSubpath()
    return path


_PLAYHEAD_SYMBOL = _make_playhead_symbol()


# ------------------------------------------------------------------ widget

class BrightnessGraphWidget(pg.PlotWidget):

    pairs_updated = pyqtSignal()
    visible_range_changed = pyqtSignal(float, float)  # X-axis zoom/pan window changed
    domain_changed = pyqtSignal(float, float)          # plotted-data range changed (set_data/clear_data)
    playhead_pair_changed = pyqtSignal(object)          # LatencyPair | None -- playhead only, not hover
    frame_clicked = pyqtSignal(int)                     # a click (not a drag) seeked here
    selection_changed = pyqtSignal(object)              # EditTarget | None -- the marker being edited
    # Emitted when a redetect re-resolved the user's edits onto moved anchors
    # (core.manual.rebind). MainWindow owns the list and must only STORE what
    # arrives here — calling set_manual_edits back would re-enter _redetect,
    # the same non-reentrancy rule that governs set_excluded_pairs.
    manual_edits_rebound = pyqtSignal(object)           # list[ManualEdit]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setBackground("#1a1a1a")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMouseTracking(True)  # hover cursor for click-drag pan, without a button held
        # A PlotWidget is a QGraphicsView, hence a QAbstractScrollArea, which
        # handles the arrow keys itself to scroll its viewport and defaults to
        # StrongFocus. Every interaction here is mouse-driven (wheel zoom, drag
        # pan, click-to-seek), so it has no use for keyboard focus — and taking
        # it on click swallowed MainWindow's frame/transition navigation keys.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        pi = self.getPlotItem()
        pi.hideAxis("bottom")
        pi.getAxis("left").setWidth(35)
        pi.setYRange(0, 255, padding=0.04)
        pi.disableAutoRange()
        self._vb = pi.getViewBox()

        # Signal lines
        self._line_orig = self.plot(pen=pg.mkPen(_GREEN, width=1.5))
        self._line_disp = self.plot(pen=pg.mkPen(_AMBER, width=1.5))
        # Paint the line above the transition markers (default zValue=0) so
        # it reads as continuous instead of being cut out at each marker.
        self._line_orig.setZValue(1)
        self._line_disp.setZValue(1)

        # Transition scatter items: rising ▲ and falling ▼, per signal
        self._sc_rise_orig = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_GREEN_MARKER), size=9, symbol="t1")
        self._sc_fall_orig = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_GREEN_MARKER), size=9, symbol="t")
        self._sc_rise_disp = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_AMBER_MARKER), size=9, symbol="t1")
        self._sc_fall_disp = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_AMBER_MARKER), size=9, symbol="t")
        for sc in (self._sc_rise_orig, self._sc_fall_orig, self._sc_rise_disp, self._sc_fall_disp):
            self.addItem(sc)

        # Midpoint dots, one per transition, at the derived (first+full)/2
        # position. Separate items from the triangles above because their x is
        # a half-frame that no brightness sample exists at, so their y has to be
        # interpolated rather than looked up.
        self._sc_mid_rise_orig = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_MID_MARKER), size=_MID_MARKER_SIZE, symbol="o")
        self._sc_mid_fall_orig = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_MID_MARKER), size=_MID_MARKER_SIZE, symbol="o")
        self._sc_mid_rise_disp = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_MID_MARKER), size=_MID_MARKER_SIZE, symbol="o")
        self._sc_mid_fall_disp = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_MID_MARKER), size=_MID_MARKER_SIZE, symbol="o")
        for sc in (self._sc_mid_rise_orig, self._sc_mid_fall_orig,
                   self._sc_mid_rise_disp, self._sc_mid_fall_disp):
            self.addItem(sc)

        # Unmatched transition markers (overlaid in red)
        self._sc_unmatched_rise = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_RED), size=10, symbol="t1")
        self._sc_unmatched_fall = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_RED), size=10, symbol="t")
        self.addItem(self._sc_unmatched_rise)
        self.addItem(self._sc_unmatched_fall)

        # Transitions the user has deleted. One item for both polarities and
        # both signals: a deleted transition has no polarity-dependent meaning
        # left — it is out of pairing entirely — so the ▲/▼ distinction would be
        # telling the user about a difference that no longer matters.
        self._sc_deleted = pg.ScatterPlotItem(
            pen=pg.mkPen(_DELETED_MARKER, width=1.5), brush=None, size=9, symbol="x",
        )
        self.addItem(self._sc_deleted)

        # Hover indicator: a vertical line at the frame a click would seek to
        # (hidden while dragging or hovering a marker -- see
        # _update_cursor_and_line). Dashed and muted so it's never confused
        # with the solid yellow playhead or the white ring highlight.
        self._hover_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen((200, 200, 200, 120), width=1, style=Qt.PenStyle.DashLine),
        )
        self._hover_line.setVisible(False)
        self.addItem(self._hover_line)

        # Connector lines linking paired orig→disp markers
        self._pair_connectors = self.plot(pen=pg.mkPen((200, 200, 200, 100), width=1))
        # Muted connector for manually-excluded pairs (see set_excluded_pairs)
        self._connector_excluded = self.plot(pen=pg.mkPen((110, 110, 110, 90), width=1))

        # Highlight overlay for the matched pair under the playhead/cursor:
        # a ring around each of its two markers, plus a brightened connector
        # segment. Layered above everything (lines=1, markers=0 by default).
        self._sc_highlight = pg.ScatterPlotItem(
            pen=pg.mkPen(_HIGHLIGHT, width=2), brush=None, size=18, symbol="o",
        )
        self._sc_highlight.setZValue(2)
        self.addItem(self._sc_highlight)
        self._connector_highlight = self.plot(pen=pg.mkPen(_HIGHLIGHT, width=3))
        self._connector_highlight.setZValue(2)

        # Ring around the marker currently being edited. Above the matched-pair
        # highlight, since the two routinely land on the same marker and the one
        # that answers "what will the nudge keys move" has to stay readable.
        self._sc_selection = pg.ScatterPlotItem(
            pen=pg.mkPen(_SELECT_RING, width=2), brush=None, size=22, symbol="o",
        )
        self._sc_selection.setZValue(3)
        self.addItem(self._sc_selection)

        # Playhead: short thick stalk + triangle anchored to the bottom of
        # the plot (mirrors TimelineWidget's own playhead shape) instead of
        # a thin full-height line that gets lost once the graph is busy.
        self._playhead_stalk = self.plot(x=[0, 0], y=[0, 0], pen=pg.mkPen(_YELLOW, width=3))
        self._playhead_marker = pg.ScatterPlotItem(
            x=[0], y=[0], pen=None, brush=pg.mkBrush(_YELLOW),
            size=_PLAYHEAD_SYMBOL_SIZE, symbol=_PLAYHEAD_SYMBOL,
        )
        self.addItem(self._playhead_marker)
        self._playhead_stalk.setVisible(False)
        self._playhead_marker.setVisible(False)
        self._playhead_y0 = 0.0
        self._playhead_apex_y = 0.0
        self._playhead_y1 = 0.0

        # Data state
        self._in_point  = 0
        self._n         = 0
        self._orig_data: np.ndarray | None = None
        self._disp_data: np.ndarray | None = None
        self._delta: float = 10.0
        self._min_spacing: int = 1
        self._max_latency: int = 0  # 0 = no limit
        self._sigma_k: float = DEFAULT_SIGMA_K
        self._ydata_min: float = 0.0
        self._ydata_max: float = 255.0

        # Zoom/pan (X axis only). Domain is the plotted-data range from the
        # last set_data call; visible is the current zoomed/panned window,
        # always clamped inside the domain.
        self._range_lo: float = 0.0
        self._range_hi: float = 0.0
        self._visible_start: float = 0.0
        self._visible_end: float = 0.0
        self._pan_drag_active = False
        self._pan_drag_start_screen_x = 0.0
        self._pan_drag_start_range = (0.0, 0.0)

        # Cached per-polarity transition frame lists. These hold ANCHOR frames
        # (steepest step) — what pairing, min-spacing and the period heuristic
        # key on. The frames drawn and navigated to come from the edges below.
        self._rise_orig_frames: list[int] = []
        self._fall_orig_frames: list[int] = []
        self._rise_disp_frames: list[int] = []
        self._fall_disp_frames: list[int] = []

        # Characterized extents, keyed by ABSOLUTE anchor frame (core.edges
        # works in array-local indices; _characterize translates on the way out
        # so the rest of this widget only ever sees absolute video frames).
        self._orig_edges: dict[int, TransitionEdge] = {}
        self._disp_edges: dict[int, TransitionEdge] = {}
        # The same edges as measured, BEFORE any manual override was applied.
        # Kept so the edit panel can show what the algorithm said beside what
        # the user chose, which is what makes Reset a meaningful offer.
        self._orig_auto_edges: dict[int, TransitionEdge] = {}
        self._disp_auto_edges: dict[int, TransitionEdge] = {}
        self._orig_quality = SignalQuality(0.0, 0.0, 0.0, False, False)
        self._disp_quality = SignalQuality(0.0, 0.0, 0.0, False, False)
        # Which marker frames belong to the original signal, so a frame looked
        # up in _frame_to_pair knows which brightness array to read its y from.
        self._orig_marker_frames: set[int] = set()

        # Pairing state
        self._rise_pairs: list[LatencyPair] = []
        self._fall_pairs: list[LatencyPair] = []
        self._rise_orig_unmatched: list[int] = []
        self._fall_orig_unmatched: list[int] = []
        self._rise_disp_unmatched: list[int] = []
        self._fall_disp_unmatched: list[int] = []

        self._polarity = "both"
        self._transition_frames: list[int] = []
        self._unmatched_frames: list[int] = []

        # Matched-pair highlight (playhead-on-marker or marker hover)
        self._frame_to_pair: dict[int, LatencyPair] = {}
        self._current_frame: int | None = None
        self._hover_matched_frame: int | None = None
        self._connector_y_level: float = 0.0
        # Broader hover state for click-to-snap: any visible marker
        # (matched or unmatched), independent of _hover_matched_frame above
        # which stays scoped to matched pairs for the ring highlight.
        self._hover_any_marker_frame: int | None = None
        # Frame under the cursor regardless of markers -- drives the hover
        # line's position when no marker is being hovered.
        self._hover_raw_frame: int | None = None
        # Captured at press time, before _hover_matched_frame gets cleared
        # below -- see mousePressEvent.
        self._click_target_frame: int | None = None
        # Playhead-only counterpart of the frame_to_pair highlight above —
        # deliberately ignores hover, unlike _resolve_highlight_pair().
        self._last_playhead_pair: LatencyPair | None = None

        # Manually-excluded pair indices (into _rise_pairs/_fall_pairs), set by
        # MainWindow via set_excluded_pairs — pure rendering hint, cleared
        # implicitly on every redetect since the pairs lists are rebuilt.
        self._rise_excluded_idx: set[int] = set()
        self._fall_excluded_idx: set[int] = set()

        # Manual editing (core.manual). MainWindow owns the authoritative list
        # — it is what gets written to the sidecar — and pushes it here with
        # set_manual_edits; this copy is what _redetect reads, and _redetect is
        # also what re-resolves it onto moved anchors.
        self._manual_edits: list[ManualEdit] = []
        self._selection: EditTarget | None = None
        # Every marker frame -> the transition ends drawn there, in a stable
        # order (original before display, rising before falling, first-light
        # before fully-lit) so a click on coincident markers is deterministic.
        self._frame_to_markers: dict[int, list[EditTarget]] = {}
        # Anchors detection found this pass, BEFORE deletions were applied.
        # A deleted transition has no edge and no anchor left downstream, so
        # this is what says whether one is genuinely deleted (draw the cross)
        # or merely dormant because detection no longer finds it at all.
        self._detected_anchors: dict[tuple[str, str], set[int]] = {}

    # ------------------------------------------------------------------ public

    def set_data(self, orig: np.ndarray, disp: np.ndarray, in_point: int) -> float:
        """Load brightness arrays and run detection. Returns the auto-computed delta."""
        self._in_point  = in_point
        self._n         = len(orig)
        self._orig_data = orig
        self._disp_data = disp

        self._range_lo = float(in_point)
        self._range_hi = float(in_point + len(orig) - 1)
        self._visible_start, self._visible_end = self._range_lo, self._range_hi

        x = np.arange(in_point, in_point + len(orig), dtype=np.float64)
        self._line_orig.setData(x=x, y=orig.astype(np.float64))
        self._line_disp.setData(x=x, y=disp.astype(np.float64))

        orig_min, orig_max = float(orig.min()), float(orig.max())
        disp_min, disp_max = float(disp.min()), float(disp.max())

        ymin = min(orig_min, disp_min)
        ymax = max(orig_max, disp_max)
        self._delta = max(5.0, 0.1 * (ymax - ymin))
        self._ydata_min = ymin
        self._ydata_max = ymax
        self.setYRange(ymin, ymax, padding=0.1)
        self.setXRange(in_point, in_point + len(orig) - 1, padding=0.01)

        rng = ymax - ymin
        view_ymin = self._vb.viewRange()[1][0]     # actual rendered lower bound
        _, py = self._vb.viewPixelSize()           # data-units per screen pixel, y

        self._playhead_y0 = view_ymin + _PLAYHEAD_EDGE_MARGIN_PX * py
        self._playhead_apex_y = self._playhead_y0 + _PLAYHEAD_TRI_H_PX * py
        stalk_reach = 0.12 * rng if rng > 0 else 1.0
        self._playhead_y1 = self._playhead_apex_y + stalk_reach
        self._playhead_stalk.setVisible(True)
        self._playhead_marker.setVisible(True)

        self._redetect()
        self.domain_changed.emit(self._range_lo, self._range_hi)
        return self._delta

    def set_delta(self, delta: float) -> None:
        self._delta = max(1.0, delta)
        if self._orig_data is not None:
            self._redetect()

    def set_min_spacing(self, spacing: int) -> None:
        self._min_spacing = max(1, spacing)
        if self._orig_data is not None:
            self._redetect()

    def set_max_latency(self, frames: int) -> None:
        self._max_latency = max(0, frames)
        if self._orig_data is not None:
            self._redetect()

    def set_sigma_k(self, k: float) -> None:
        """Edge Sensitivity: how many noise sigmas a signal must leave its
        baseline by before it counts as changing. Higher = later first-light,
        earlier fully-lit."""
        self._sigma_k = max(0.0, k)
        if self._orig_data is not None:
            self._redetect()

    def get_signal_quality(self) -> tuple[SignalQuality, SignalQuality]:
        """(original, display) whole-signal quality verdicts from the last
        detection pass."""
        return self._orig_quality, self._disp_quality

    def set_polarity(self, mode: str) -> None:
        """mode: 'both' | 'rising' | 'falling'"""
        self._polarity = mode
        self._apply_polarity()

    def set_frame(self, frame: int) -> None:
        if self._playhead_marker.isVisible():
            self._playhead_stalk.setData(x=[frame, frame], y=[self._playhead_apex_y, self._playhead_y1])
            self._playhead_marker.setData(x=[frame], y=[self._playhead_y0])
            width = self._visible_end - self._visible_start
            self.set_visible_range(*center_range(
                frame, width, self._range_lo, self._range_hi, MIN_ZOOM_FRAMES,
            ))
            self._current_frame = frame
            self._update_marker_highlight()
            self._update_playhead_pair_signal()

    def set_visible_range(self, start: float, end: float) -> None:
        """Set the graph's visible X window (zoom/pan), clamped to the
        plotted-data domain. Emits visible_range_changed on actual change —
        the zoom bar mirrors this to stay in sync with wheel-zoom and
        click-drag pan, both of which also go through this method."""
        if self._range_hi <= self._range_lo:
            return
        start, end = clamp_range(start, end, self._range_lo, self._range_hi, MIN_ZOOM_FRAMES)
        if (start, end) != (self._visible_start, self._visible_end):
            self._visible_start, self._visible_end = start, end
            self._vb.setXRange(start, end, padding=0)
            self.visible_range_changed.emit(start, end)

    def clear_data(self) -> None:
        self._orig_data = self._disp_data = None
        self._transition_frames = []
        self._unmatched_frames = []
        # Fresh list per attribute — chained `a = b = []` would alias them.
        self._rise_orig_frames, self._fall_orig_frames = [], []
        self._rise_disp_frames, self._fall_disp_frames = [], []
        self._rise_pairs, self._fall_pairs = [], []
        self._rise_orig_unmatched, self._fall_orig_unmatched = [], []
        self._rise_disp_unmatched, self._fall_disp_unmatched = [], []
        for item in (self._line_orig, self._line_disp):
            item.setData(x=[], y=[])
        self._orig_edges, self._disp_edges = {}, {}
        self._orig_auto_edges, self._disp_auto_edges = {}, {}
        self._orig_quality = SignalQuality(0.0, 0.0, 0.0, False, False)
        self._disp_quality = SignalQuality(0.0, 0.0, 0.0, False, False)
        self._orig_marker_frames = set()
        # The edit list itself deliberately survives: it belongs to the clip,
        # not to one extraction pass, so re-analyzing the same footage keeps the
        # user's review. MainWindow replaces it when a different file is opened.
        self._frame_to_markers = {}
        self._detected_anchors = {}
        for sc in (self._sc_rise_orig, self._sc_fall_orig,
                   self._sc_rise_disp, self._sc_fall_disp,
                   self._sc_mid_rise_orig, self._sc_mid_fall_orig,
                   self._sc_mid_rise_disp, self._sc_mid_fall_disp,
                   self._sc_unmatched_rise, self._sc_unmatched_fall,
                   self._sc_deleted, self._sc_selection):
            sc.setData(x=[], y=[])
        self._pair_connectors.setData(x=[], y=[])
        self._connector_excluded.setData(x=[], y=[])
        self._rise_excluded_idx, self._fall_excluded_idx = set(), set()
        self._playhead_stalk.setVisible(False)
        self._playhead_marker.setVisible(False)
        self.setYRange(0, 255, padding=0.04)
        self._n = 0
        self._range_lo = self._range_hi = 0.0
        self._visible_start = self._visible_end = 0.0
        self._frame_to_pair = {}
        self._current_frame = None
        self._hover_matched_frame = None
        self._hover_any_marker_frame = None
        self._hover_raw_frame = None
        self._click_target_frame = None
        self._sc_highlight.setData(x=[], y=[])
        self._connector_highlight.setData(x=[], y=[])
        self._hover_line.setVisible(False)
        if self._selection is not None:
            self._selection = None
            self.selection_changed.emit(None)
        self._update_playhead_pair_signal()
        self.domain_changed.emit(0.0, -1.0)
        self.pairs_updated.emit()

    def next_transition(self, after_frame: int) -> int | None:
        for f in self._transition_frames:
            if f > after_frame:
                return f
        return None

    def prev_transition(self, before_frame: int) -> int | None:
        result = None
        for f in self._transition_frames:
            if f < before_frame:
                result = f
        return result

    def next_unmatched(self, after_frame: int) -> int | None:
        for f in self._unmatched_frames:
            if f > after_frame:
                return f
        return None

    def prev_unmatched(self, before_frame: int) -> int | None:
        result = None
        for f in self._unmatched_frames:
            if f < before_frame:
                result = f
        return result

    def get_pairs(self) -> list[LatencyPair]:
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        result: list[LatencyPair] = []
        if show_r:
            result.extend(self._rise_pairs)
        if show_f:
            result.extend(self._fall_pairs)
        return sorted(result, key=lambda p: p.orig_frame)

    def get_pairs_for(self, polarity: str, *, active: str | None = None) -> list[LatencyPair]:
        """Returns this polarity's pairs ('rising' or 'falling'), or [] if
        `active` (defaults to the live polarity filter) hides that polarity."""
        gate = self._polarity if active is None else active
        if gate not in ("both", polarity):
            return []
        return self._rise_pairs if polarity == "rising" else self._fall_pairs

    def get_orig_period_frames(self, polarity: str) -> float | None:
        """Average frame-spacing between consecutive same-polarity original transitions.
        Uses rising if polarity is 'both' or 'rising', falling otherwise."""
        if polarity == "falling":
            frames = self._fall_orig_frames
        else:
            frames = self._rise_orig_frames if self._rise_orig_frames else self._fall_orig_frames
        if len(frames) < 2:
            return None
        diffs = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
        return float(sum(diffs) / len(diffs))

    def get_unmatched_counts(self) -> tuple[int, int]:
        """Returns (unmatched_orig, unmatched_disp) for the active polarity."""
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        uo = ud = 0
        if show_r:
            uo += len(self._rise_orig_unmatched)
            ud += len(self._rise_disp_unmatched)
        if show_f:
            uo += len(self._fall_orig_unmatched)
            ud += len(self._fall_disp_unmatched)
        return uo, ud

    # ------------------------------------------------------- manual editing

    def set_manual_edits(self, edits: list[ManualEdit]) -> None:
        """Adopt MainWindow's edit list and re-run the pipeline against it.

        MainWindow owns the list (it is what the sidecar stores); this is a
        copy. A redetect follows because deletions change which transitions
        pair, and overrides change what every metric reads."""
        self._manual_edits = list(edits)
        if self._orig_data is not None:
            self._redetect()

    def get_manual_edits(self) -> list[ManualEdit]:
        return list(self._manual_edits)

    def analysis_bounds(self) -> tuple[int, int]:
        """Inclusive first/last frame of the analysed range — the limits a
        marker may be nudged between."""
        return self._in_point, self._in_point + max(0, self._n - 1)

    def selection(self) -> EditTarget | None:
        return self._selection

    def set_selection(self, target: EditTarget | None) -> None:
        if target == self._selection:
            return
        self._selection = target
        self._update_selection_marker()
        self.selection_changed.emit(self._selection)

    def clear_selection(self) -> None:
        self.set_selection(None)

    def select_frame(self, frame: int, pos=None) -> bool:
        """Select a marker drawn at `frame`. Returns False if there is none.

        `pos` is an optional widget-local mouse position. X has already decided
        the frame by the time this is called, so Y is used only to choose
        between markers that sit on the SAME frame on the two traces — which are
        far apart vertically, so this asks nothing of the user's aim that they
        cannot answer by eye."""
        candidates = self._frame_to_markers.get(frame)
        if not candidates:
            return False
        if pos is None or len(candidates) == 1:
            self.set_selection(candidates[0])
            return True
        best, best_dist = candidates[0], None
        for target in candidates:
            y = self._marker_y(target)
            if y is None:
                continue
            widget_pt = self.mapFromScene(self._vb.mapViewToScene(QPointF(frame, y)))
            dist = abs(widget_pt.y() - pos.y())
            if best_dist is None or dist < best_dist:
                best, best_dist = target, dist
        self.set_selection(best)
        return True

    def _edges_for(self, roi: str) -> dict[int, TransitionEdge]:
        return self._orig_edges if roi == "original" else self._disp_edges

    def _auto_edges_for(self, roi: str) -> dict[int, TransitionEdge]:
        return self._orig_auto_edges if roi == "original" else self._disp_auto_edges

    def _data_for(self, roi: str) -> np.ndarray | None:
        return self._orig_data if roi == "original" else self._disp_data

    def edge_for(self, target: EditTarget) -> TransitionEdge | None:
        """The transition's measured extent as currently reported — overrides
        applied."""
        return self._edges_for(target.roi).get(target.anchor_frame)

    def resolved_frames(self, target: EditTarget) -> tuple[int, int] | None:
        edge = self.edge_for(target)
        return None if edge is None else (edge.first_frame, edge.full_frame)

    def auto_frames(self, target: EditTarget) -> tuple[int, int] | None:
        """What the algorithm measured, before the user moved anything. None
        when this transition could not be characterized at all."""
        edge = self._auto_edges_for(target.roi).get(target.anchor_frame)
        return None if edge is None else (edge.first_frame, edge.full_frame)

    def is_deleted(self, target: EditTarget) -> bool:
        return any(
            e.deleted and (e.roi, e.polarity, e.anchor_frame) == target.transition
            for e in self._manual_edits
        )

    def marker_frame(self, target: EditTarget) -> int | None:
        """The frame `target`'s marker is drawn at, or None if its transition is
        no longer on the graph at all."""
        if self.is_deleted(target):
            for deleted, frame in self._deleted_targets():
                if deleted.transition == target.transition:
                    return frame
            return None
        edge = self.edge_for(target)
        if edge is None:
            anchors = self._anchors_for(target.roi, target.polarity)
            return target.anchor_frame if target.anchor_frame in anchors else None
        return edge.first_frame if target.which == "first" else edge.full_frame

    def _anchors_for(self, roi: str, polarity: str) -> list[int]:
        if roi == "original":
            return self._rise_orig_frames if polarity == "rising" else self._fall_orig_frames
        return self._rise_disp_frames if polarity == "rising" else self._fall_disp_frames

    def _marker_y(self, target: EditTarget) -> float | None:
        frame = self.marker_frame(target)
        data = self._data_for(target.roi)
        if frame is None or data is None:
            return None
        i = frame - self._in_point
        if not (0 <= i < len(data)):
            return None
        return float(data[i])

    def transition_position(self, target: EditTarget) -> tuple[int, int]:
        """(1-based position, total) of `target`'s transition among the ones
        Up/Down walks, so the edit panel can say how far through a review pass
        the user is. A deleted transition returns position 0 — it is not one of
        the stops."""
        ordered = self._ordered_transitions()
        total = len(ordered)
        if self.is_deleted(target):
            return 0, total
        try:
            return ordered.index(target.transition) + 1, total
        except ValueError:
            return 0, total

    def _ordered_transitions(self) -> list[tuple[str, str, int]]:
        """Every selectable, non-deleted transition in the visible polarities,
        in the order the playhead meets them."""
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        out: list[tuple[int, tuple[str, str, int]]] = []
        groups = []
        if show_r:
            groups += [("original", "rising", self._rise_orig_frames, self._orig_edges),
                       ("display", "rising", self._rise_disp_frames, self._disp_edges)]
        if show_f:
            groups += [("original", "falling", self._fall_orig_frames, self._orig_edges),
                       ("display", "falling", self._fall_disp_frames, self._disp_edges)]
        for roi, polarity, anchors, edges in groups:
            for anchor in anchors:
                edge = edges.get(anchor)
                frame = anchor if edge is None else edge.first_frame
                out.append((frame, (roi, polarity, anchor)))
        out.sort()
        return [t for _, t in out]

    def _update_selection_marker(self) -> None:
        """Draw the selection ring, and drop a selection whose transition a
        redetect has removed — a ring around nothing would be worse than no
        ring, since the nudge keys would appear to do nothing."""
        target = self._selection
        if target is None or self._orig_data is None:
            self._sc_selection.setData(x=[], y=[])
            return
        frame = self.marker_frame(target)
        y = self._marker_y(target)
        if frame is None or y is None:
            self._sc_selection.setData(x=[], y=[])
            self._selection = None
            self.selection_changed.emit(None)
            return
        self._sc_selection.setData(x=[float(frame)], y=[y])

    # ------------------------------------------------------------------ mouse

    def wheelEvent(self, event) -> None:
        """Zoom the X axis around the cursor. Not calling super() is
        deliberate: setMouseEnabled(x=False, y=False) already suppresses the
        ViewBox's own wheel-zoom, but this replaces it outright rather than
        layering on top of it."""
        if self._range_hi <= self._range_lo:
            return
        anchor = self._scene_x_to_data(event.position())
        factor = _WHEEL_ZOOM_IN if event.angleDelta().y() > 0 else _WHEEL_ZOOM_OUT
        self._apply_zoom(factor, anchor)
        event.accept()

    def _apply_zoom(self, factor: float, anchor: float) -> None:
        self.set_visible_range(*zoom_range(
            self._visible_start, self._visible_end, factor, anchor,
            self._range_lo, self._range_hi, MIN_ZOOM_FRAMES,
        ))

    def _scene_x_to_data(self, pos) -> float:
        # Accepts QPoint or QPointF -- QGraphicsView.mapToScene only takes
        # QPoint, so round explicitly rather than relying on QPointF.toPoint()
        # (which QPoint itself doesn't have).
        scene_pos = self.mapToScene(QPoint(round(pos.x()), round(pos.y())))
        return self._vb.mapSceneToView(scene_pos).x()

    def _frame_at_pos(self, pos) -> int | None:
        """Nearest in-domain frame under a widget-local pixel position, or
        None if no data is loaded. Shared by the hover line's position and
        mouseReleaseEvent's raw-click fallback."""
        if self._range_hi <= self._range_lo:
            return None
        x = self._scene_x_to_data(pos)
        return max(int(self._range_lo), min(int(self._range_hi), round(x)))

    def _update_cursor_and_line(self) -> None:
        """Single source of truth for 'what does hovering/pressing here look
        like right now' -- every path that can change it (press, move,
        release, leave) ends by calling this, so no transition can leave a
        stale cursor or a line stuck visible."""
        if self._pan_drag_active:
            self._hover_line.setVisible(False)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._hover_any_marker_frame is not None:
            # Line stays visible, snapped to the marker's frame rather than
            # the raw mouse position -- with X-only hit-testing, two close
            # markers' hover radii can overlap, and this is what actually
            # tells you which one would be clicked, not just that "a" marker
            # is in range.
            self._hover_line.setPos(self._hover_any_marker_frame)
            self._hover_line.setVisible(True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self._hover_raw_frame is not None:
            self._hover_line.setPos(self._hover_raw_frame)
            self._hover_line.setVisible(True)
            self.setCursor(Qt.CursorShape.BlankCursor)
        else:
            self._hover_line.setVisible(False)
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._range_hi > self._range_lo:
            # Captured before the hover-clear below, which would otherwise
            # erase it before mouseReleaseEvent gets a chance to read it.
            self._click_target_frame = self._hover_any_marker_frame
            self._pan_drag_active = True
            self._pan_drag_start_screen_x = event.position().x()
            self._pan_drag_start_range = (self._visible_start, self._visible_end)
            if self._hover_matched_frame is not None:
                self._hover_matched_frame = None
                self._update_marker_highlight()
            self._hover_any_marker_frame = None
            self._hover_raw_frame = None
            self._update_cursor_and_line()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_drag_active:
            dx_px = event.position().x() - self._pan_drag_start_screen_x
            self._apply_pan_drag(dx_px)
            event.accept()
            return
        self._update_hover(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_drag_active and event.button() == Qt.MouseButton.LeftButton:
            self._pan_drag_active = False
            moved = abs(event.position().x() - self._pan_drag_start_screen_x) > _CLICK_DRAG_THRESHOLD_PX
            # Resync hover/cursor/line to the release position now, rather
            # than leaving it stale (frozen from before the drag started,
            # since mouseMoveEvent skips _update_hover while dragging) until
            # an incidental future move.
            self._update_hover(event.position())
            if not moved and self._orig_data is not None:
                frame = self._click_target_frame
                if frame is None:
                    frame = self._hover_raw_frame
                if frame is not None:
                    # Selection follows the same snap the seek does: a click
                    # that landed on a marker selects it, a click on bare graph
                    # clears the selection rather than leaving a ring somewhere
                    # the user is no longer looking.
                    if self._click_target_frame is None or not self.select_frame(
                        frame, event.position()
                    ):
                        self.clear_selection()
                    self.frame_clicked.emit(frame)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover_matched_frame is not None:
            self._hover_matched_frame = None
            self._update_marker_highlight()
        self._hover_any_marker_frame = None
        self._hover_raw_frame = None
        self._update_cursor_and_line()
        super().leaveEvent(event)

    def _update_hover(self, pos) -> None:
        frame = self._hover_hit_test(pos) if self._orig_data is not None else None
        if frame != self._hover_matched_frame:
            self._hover_matched_frame = frame
            self._update_marker_highlight()
        self._hover_any_marker_frame = self._hit_test_any_marker(pos) if self._orig_data is not None else None
        self._hover_raw_frame = self._frame_at_pos(pos)
        self._update_cursor_and_line()

    def _hover_hit_test(self, pos) -> int | None:
        """pos: widget-local QPointF (mouseMoveEvent's event.position()).
        Returns the nearest matched frame within a horizontal (X-only) pixel
        hit radius, or None. X-only rather than 2D pixel distance because
        the real cursor is hidden while the hover line is shown (see
        _update_cursor_and_line) -- the user has no way to see or aim by
        vertical position, so requiring it would make "am I on this marker"
        unanswerable by eye."""
        if not self._frame_to_pair:
            return None
        best_frame, best_dist = None, _HOVER_HIT_R_PX
        for frame in self._frame_to_pair:
            # A pair now owns up to four marker frames, so which signal a frame
            # belongs to can no longer be inferred by comparing it against the
            # pair's anchors — _orig_marker_frames records it instead.
            data = self._orig_data if frame in self._orig_marker_frames else self._disp_data
            y = float(data[frame - self._in_point])
            widget_pt = self.mapFromScene(self._vb.mapViewToScene(QPointF(frame, y)))
            dist = abs(widget_pt.x() - pos.x())
            if dist < best_dist:
                best_frame, best_dist = frame, dist
        return best_frame

    def _hit_test_any_marker(self, pos) -> int | None:
        """Like _hover_hit_test (also X-only, same reasoning), but considers
        every visible transition marker (matched AND unmatched), not just
        matched pairs. Kept independent of _hover_hit_test rather than
        derived from it: deriving one from the other would change existing
        ring-highlight hover behavior in the case where the nearest marker
        overall is unmatched but a matched one is still within radius."""
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        # Edge frames, not anchors: the anchor is no longer drawn, and snapping
        # to an invisible marker would be baffling. The midpoint dots are
        # deliberately excluded — seeking to a half-frame that no sample exists
        # at would be a lie.
        candidates: list[tuple[int, np.ndarray]] = []
        if show_r:
            candidates += [(f, self._orig_data) for f in self._edge_frames(self._rise_orig_frames, self._orig_edges)]
            candidates += [(f, self._disp_data) for f in self._edge_frames(self._rise_disp_frames, self._disp_edges)]
        if show_f:
            candidates += [(f, self._orig_data) for f in self._edge_frames(self._fall_orig_frames, self._orig_edges)]
            candidates += [(f, self._disp_data) for f in self._edge_frames(self._fall_disp_frames, self._disp_edges)]
        best_frame, best_dist = None, _HOVER_HIT_R_PX
        for frame, data in candidates:
            y = float(data[frame - self._in_point])
            widget_pt = self.mapFromScene(self._vb.mapViewToScene(QPointF(frame, y)))
            dist = abs(widget_pt.x() - pos.x())
            if dist < best_dist:
                best_frame, best_dist = frame, dist
        return best_frame

    def _resolve_highlight_pair(self) -> LatencyPair | None:
        if self._hover_matched_frame is not None:
            pair = self._frame_to_pair.get(self._hover_matched_frame)
            if pair is not None:
                return pair
        if self._current_frame is not None:
            return self._frame_to_pair.get(self._current_frame)
        return None

    def _update_marker_highlight(self) -> None:
        pair = self._resolve_highlight_pair()
        if pair is None or self._orig_data is None or self._disp_data is None:
            self._sc_highlight.setData(x=[], y=[])
            self._connector_highlight.setData(x=[], y=[])
            return
        # Ring every one of the pair's markers, so hovering the fully-lit
        # triangle highlights that triangle rather than only its first-light
        # sibling at the other end of the ramp.
        xs: list[float] = []
        ys: list[float] = []
        for orig in (True, False):
            data = self._orig_data if orig else self._disp_data
            for frame in self._pair_frames(pair, orig=orig):
                xs.append(float(frame))
                ys.append(float(data[frame - self._in_point]))
        self._sc_highlight.setData(x=xs, y=ys)
        self._connector_highlight.setData(
            x=[pair.orig_frame, pair.disp_frame],
            y=[self._connector_y_level, self._connector_y_level],
        )

    def _apply_pan_drag(self, dx_px: float) -> None:
        """dx_px: horizontal mouse movement in screen pixels since the drag
        started. Dragging right (positive dx_px) reveals earlier frames, as
        if grabbing and pulling the plotted content along with the cursor."""
        px_data, _ = self._vb.viewPixelSize()
        delta = -dx_px * px_data
        start0, end0 = self._pan_drag_start_range
        self.set_visible_range(*pan_range(start0, end0, delta, self._range_lo, self._range_hi))

    # --------------------------------------------------------------- internals

    def _redetect(self) -> None:
        sp = self._min_spacing
        rise_orig = apply_min_spacing(
            [self._in_point + i for i in find_rising(self._orig_data,  self._delta)], sp)
        fall_orig = apply_min_spacing(
            [self._in_point + i for i in find_falling(self._orig_data, self._delta)], sp)
        rise_disp = apply_min_spacing(
            [self._in_point + i for i in find_rising(self._disp_data,  self._delta)], sp)
        fall_disp = apply_min_spacing(
            [self._in_point + i for i in find_falling(self._disp_data, self._delta)], sp)
        self._detected_anchors = {
            ("original", "rising"): set(rise_orig),
            ("original", "falling"): set(fall_orig),
            ("display", "rising"): set(rise_disp),
            ("display", "falling"): set(fall_disp),
        }

        # Re-resolve the user's edits onto this pass's anchors before anything
        # keys on them, then drop the transitions they deleted. Deletion has to
        # happen HERE, upstream of characterization: characterize_signal bounds
        # each transition's search window at the midpoints to its neighbouring
        # anchors, so a noise blip deleted any later than this would still be
        # truncating the window of the real transition beside it.
        edits = rebind(self._manual_edits, rise_orig, fall_orig, "original")
        edits = rebind(edits, rise_disp, fall_disp, "display")
        if edits != self._manual_edits:
            self._manual_edits = edits
            # Before pairs_updated, so MainWindow's handlers never see a list
            # that disagrees with the pairs they are about to read.
            self.manual_edits_rebound.emit(list(edits))

        self._rise_orig_frames, self._fall_orig_frames = delete_anchors(
            edits, rise_orig, fall_orig, "original")
        self._rise_disp_frames, self._fall_disp_frames = delete_anchors(
            edits, rise_disp, fall_disp, "display")

        # Characterize each located transition. Both polarities go in together:
        # each transition's search window is bounded by its neighbours of
        # EITHER polarity, since on a square wave the plateau after a rising
        # edge is terminated by the following falling edge.
        self._orig_auto_edges, self._orig_quality = self._characterize(
            self._orig_data, self._rise_orig_frames, self._fall_orig_frames)
        self._disp_auto_edges, self._disp_quality = self._characterize(
            self._disp_data, self._rise_disp_frames, self._fall_disp_frames)

        # Overrides come last, so they cannot disturb anything upstream. The
        # anchors are untouched and pairing keys on anchors, which is what
        # guarantees that nudging a marker changes the reported milliseconds and
        # nothing else — it can never silently re-pair a transition.
        lo, hi = self._in_point, self._in_point + self._n - 1
        self._orig_edges = apply_overrides(edits, self._orig_auto_edges, lo, hi, "original")
        self._disp_edges = apply_overrides(edits, self._disp_auto_edges, lo, hi, "display")

        max_fr = self._max_latency if self._max_latency > 0 else None
        self._rise_pairs, self._rise_orig_unmatched, self._rise_disp_unmatched = \
            pair_transitions(self._rise_orig_frames, self._rise_disp_frames, "rising",  max_frames=max_fr,
                             orig_edges=self._orig_edges, disp_edges=self._disp_edges)
        self._fall_pairs, self._fall_orig_unmatched, self._fall_disp_unmatched = \
            pair_transitions(self._fall_orig_frames, self._fall_disp_frames, "falling", max_frames=max_fr,
                             orig_edges=self._orig_edges, disp_edges=self._disp_edges)

        self._populate_transitions()
        self._apply_polarity()

    def _characterize(
        self, data: np.ndarray, rise_frames: list[int], fall_frames: list[int]
    ) -> tuple[dict[int, TransitionEdge], SignalQuality]:
        """Run edge characterization in the array-local index space the
        brightness arrays live in, then translate every frame back to absolute
        video frames. Keeping the offset conversion here — the same place
        _redetect already converts detection's output — is what lets the rest of
        the widget deal only in absolute frames."""
        offset = self._in_point
        edges, quality = characterize_signal(
            data,
            [f - offset for f in rise_frames],
            [f - offset for f in fall_frames],
            self._sigma_k,
        )
        absolute = {
            anchor + offset: replace(
                edge,
                anchor_frame=edge.anchor_frame + offset,
                first_frame=edge.first_frame + offset,
                full_frame=edge.full_frame + offset,
            )
            for anchor, edge in edges.items()
        }
        return absolute, quality

    def _edge_frames(self, anchors: list[int], edges: dict[int, TransitionEdge]) -> list[int]:
        """The frames actually drawn and navigated to: first-light and fully-lit
        for each transition. They coincide on an instantaneous transition, and
        an anchor that could not be characterized falls back to itself so no
        transition ever becomes invisible or unreachable."""
        frames: set[int] = set()
        for anchor in anchors:
            edge = edges.get(anchor)
            if edge is None:
                frames.add(anchor)
            else:
                frames.add(edge.first_frame)
                frames.add(edge.full_frame)
        return sorted(frames)

    def _first_frames(self, anchors: list[int], edges: dict[int, TransitionEdge]) -> list[int]:
        """One frame per transition, at first-light. Used for Up/Down
        navigation: stopping at all three markers would triple the keypresses
        to cross a clip, and fully-lit is always a few frames further on."""
        return sorted({
            (edges[a].first_frame if a in edges else a) for a in anchors
        })

    def _flagged_frames(self, anchors: list[int], edges: dict[int, TransitionEdge]) -> set[int]:
        return {
            f
            for a in anchors
            if (edge := edges.get(a)) is not None and edge.warnings
            for f in (edge.first_frame, edge.full_frame)
        }

    def _manual_frames(self, anchors: list[int], edges: dict[int, TransitionEdge]) -> set[int]:
        return {
            f
            for a in anchors
            if (edge := edges.get(a)) is not None and edge.manual
            for f in (edge.first_frame, edge.full_frame)
        }

    def _deleted_targets(self) -> list[tuple[EditTarget, int]]:
        """(target, frame to draw at) for every transition the user has deleted
        that detection actually found this pass. An edit whose anchor isn't in
        the detected set is dormant — its transition isn't being detected at all
        right now — and drawing a cross for it would claim something was removed
        that was never there."""
        out: list[tuple[EditTarget, int]] = []
        for edit in self._manual_edits:
            if not edit.deleted:
                continue
            if edit.anchor_frame not in self._detected_anchors.get(
                (edit.roi, edit.polarity), ()
            ):
                continue
            frame = edit.anchor_frame if edit.first_frame is None else edit.first_frame
            out.append((
                EditTarget(edit.roi, edit.polarity, edit.anchor_frame, "first"),
                frame,
            ))
        return out

    def _populate_deleted(self) -> None:
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        xs: list[float] = []
        ys: list[float] = []
        for target, frame in self._deleted_targets():
            if target.polarity == "rising" and not show_r:
                continue
            if target.polarity == "falling" and not show_f:
                continue
            data = self._orig_data if target.roi == "original" else self._disp_data
            if data is None or not (0 <= frame - self._in_point < len(data)):
                continue
            xs.append(float(frame))
            ys.append(float(data[frame - self._in_point]))
        self._sc_deleted.setData(x=xs, y=ys)

    def _interp_y(self, data: np.ndarray, frame: float) -> float:
        """Brightness at a fractional frame, linearly interpolated. Only the
        midpoint dots need this — every other marker sits on a real sample."""
        i = frame - self._in_point
        lo = max(0, min(len(data) - 1, int(math.floor(i))))
        hi = max(0, min(len(data) - 1, int(math.ceil(i))))
        if lo == hi:
            return float(data[lo])
        return float(data[lo] + (data[hi] - data[lo]) * (i - lo))

    def _populate_mid(
        self,
        anchors: list[int],
        edges: dict[int, TransitionEdge],
        data: np.ndarray,
        item: pg.ScatterPlotItem,
    ) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for anchor in anchors:
            edge = edges.get(anchor)
            # Nothing to mark when the transition is instantaneous: the midpoint
            # would land exactly on the two coincident triangles.
            if edge is None or edge.full_frame == edge.first_frame:
                continue
            x = (edge.first_frame + edge.full_frame) / 2.0
            xs.append(x)
            ys.append(self._interp_y(data, x))
        item.setData(x=xs, y=ys)

    def _populate_transitions(
        self,
        rise_excluded: frozenset[int] = frozenset(),
        fall_excluded: frozenset[int] = frozenset(),
    ) -> None:
        """Draw every transition marker. Shared by _redetect and
        set_excluded_pairs so the two can never disagree about what is on
        screen.

        The two excluded sets stay separate rather than being unioned: a fast
        flash can put a rising transition's fully-lit frame on the very frame
        the following falling transition first departs from, and excluding one
        pair must not mute the other's marker."""
        for anchors, edges, data, tri, mid, brush, excluded in (
            (self._rise_orig_frames, self._orig_edges, self._orig_data,
             self._sc_rise_orig, self._sc_mid_rise_orig, _GREEN_MARKER, rise_excluded),
            (self._fall_orig_frames, self._orig_edges, self._orig_data,
             self._sc_fall_orig, self._sc_mid_fall_orig, _GREEN_MARKER, fall_excluded),
            (self._rise_disp_frames, self._disp_edges, self._disp_data,
             self._sc_rise_disp, self._sc_mid_rise_disp, _AMBER_MARKER, rise_excluded),
            (self._fall_disp_frames, self._disp_edges, self._disp_data,
             self._sc_fall_disp, self._sc_mid_fall_disp, _AMBER_MARKER, fall_excluded),
        ):
            self._populate(
                self._edge_frames(anchors, edges), data, tri, brush,
                excluded, self._flagged_frames(anchors, edges),
                self._manual_frames(anchors, edges),
            )
            self._populate_mid(anchors, edges, data, mid)
        self._populate_deleted()

    def _apply_polarity(self) -> None:
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        self._sc_rise_orig.setVisible(show_r)
        self._sc_rise_disp.setVisible(show_r)
        self._sc_fall_orig.setVisible(show_f)
        self._sc_fall_disp.setVisible(show_f)
        self._sc_mid_rise_orig.setVisible(show_r)
        self._sc_mid_rise_disp.setVisible(show_r)
        self._sc_mid_fall_orig.setVisible(show_f)
        self._sc_mid_fall_disp.setVisible(show_f)
        self._sc_unmatched_rise.setVisible(show_r)
        self._sc_unmatched_fall.setVisible(show_f)

        # Navigation lands on first-light: one stop per transition, not three.
        frames: set[int] = set()
        if show_r:
            frames.update(self._first_frames(self._rise_orig_frames, self._orig_edges))
            frames.update(self._first_frames(self._rise_disp_frames, self._disp_edges))
        if show_f:
            frames.update(self._first_frames(self._fall_orig_frames, self._orig_edges))
            frames.update(self._first_frames(self._fall_disp_frames, self._disp_edges))
        self._transition_frames = sorted(frames)

        unmatched: set[int] = set()
        if show_r:
            unmatched.update(self._first_frames(self._rise_orig_unmatched, self._orig_edges))
            unmatched.update(self._first_frames(self._rise_disp_unmatched, self._disp_edges))
        if show_f:
            unmatched.update(self._first_frames(self._fall_orig_unmatched, self._orig_edges))
            unmatched.update(self._first_frames(self._fall_disp_unmatched, self._disp_edges))
        self._unmatched_frames = sorted(unmatched)

        # Unmatched markers overlay the same edge frames the normal markers use,
        # so the red sits exactly on top of a triangle instead of beside it.
        self._populate_unmatched(
            self._edge_frames(self._rise_orig_unmatched, self._orig_edges), self._orig_data,
            self._edge_frames(self._rise_disp_unmatched, self._disp_edges), self._disp_data,
            self._sc_unmatched_rise,
        )
        self._populate_unmatched(
            self._edge_frames(self._fall_orig_unmatched, self._orig_edges), self._orig_data,
            self._edge_frames(self._fall_disp_unmatched, self._disp_edges), self._disp_data,
            self._sc_unmatched_fall,
        )

        active_pairs: list[LatencyPair] = []
        if show_r:
            active_pairs.extend(self._rise_pairs)
        if show_f:
            active_pairs.extend(self._fall_pairs)
        self._update_connectors(active_pairs)

        # Every marker frame of a matched pair maps back to it — four of them
        # once first-light and fully-lit differ — so hovering or seeking to any
        # of a pair's markers resolves to the same pair.
        self._frame_to_pair = {}
        self._orig_marker_frames = set()
        for p in active_pairs:
            for frame in self._pair_frames(p, orig=True):
                self._frame_to_pair[frame] = p
                self._orig_marker_frames.add(frame)
            for frame in self._pair_frames(p, orig=False):
                self._frame_to_pair[frame] = p
        self._rebuild_marker_index()
        self._update_marker_highlight()
        self._update_selection_marker()
        self._update_playhead_pair_signal()

        self.pairs_updated.emit()

    def _rebuild_marker_index(self) -> None:
        """frame -> the transition ends drawn there, for click-to-select.

        The visit order below fixes the tie-break when two markers land on the
        same frame: original before display, rising before falling, first-light
        before fully-lit. It has to be deterministic rather than merely
        sensible — clicking the same pixel twice must select the same thing."""
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        index: dict[int, list[EditTarget]] = {}

        def add(frame: int, target: EditTarget) -> None:
            index.setdefault(frame, []).append(target)

        groups = []
        if show_r:
            groups.append(("original", "rising", self._rise_orig_frames, self._orig_edges))
            groups.append(("display", "rising", self._rise_disp_frames, self._disp_edges))
        if show_f:
            groups.append(("original", "falling", self._fall_orig_frames, self._orig_edges))
            groups.append(("display", "falling", self._fall_disp_frames, self._disp_edges))
        # Sorted so "original before display" holds regardless of the append
        # order above, and rising before falling within a signal.
        for roi, polarity, anchors, edges in sorted(
            groups, key=lambda g: (g[0] != "original", g[1] != "rising")
        ):
            for anchor in anchors:
                edge = edges.get(anchor)
                if edge is None:
                    # Not characterizable, but still a real transition the user
                    # may want to delete — reachable at its anchor.
                    add(anchor, EditTarget(roi, polarity, anchor, "first"))
                    continue
                add(edge.first_frame, EditTarget(roi, polarity, anchor, "first"))
                if edge.full_frame != edge.first_frame:
                    add(edge.full_frame, EditTarget(roi, polarity, anchor, "full"))

        for target, frame in self._deleted_targets():
            if target.polarity == "rising" and not show_r:
                continue
            if target.polarity == "falling" and not show_f:
                continue
            add(frame, target)

        self._frame_to_markers = index

    def _pair_frames(self, pair: LatencyPair, *, orig: bool) -> list[int]:
        """One end of a pair's marker frames: first-light and fully-lit, or just
        the anchor when that end could not be characterized."""
        edge = pair.orig_edge if orig else pair.disp_edge
        if edge is None:
            return [pair.orig_frame if orig else pair.disp_frame]
        return sorted({edge.first_frame, edge.full_frame})

    def _update_playhead_pair_signal(self) -> None:
        pair = self._frame_to_pair.get(self._current_frame) if self._current_frame is not None else None
        if pair is not self._last_playhead_pair:
            self._last_playhead_pair = pair
            self.playhead_pair_changed.emit(pair)

    def _populate(
        self,
        frame_list: list[int],
        data: np.ndarray,
        item: pg.ScatterPlotItem,
        normal_brush: tuple[int, int, int] | None = None,
        excluded_frames: frozenset[int] = frozenset(),
        flagged_frames: set[int] | frozenset[int] = frozenset(),
        manual_frames: set[int] | frozenset[int] = frozenset(),
    ) -> None:
        if not frame_list:
            item.setData(x=[], y=[])
            return
        y = [float(data[f - self._in_point]) for f in frame_list]
        kwargs = {}
        if excluded_frames:
            kwargs["brush"] = [
                pg.mkBrush(_MUTED_MARKER if f in excluded_frames else normal_brush)
                for f in frame_list
            ]
        # Per-point pens, same mechanism as the per-point brushes above. Clean
        # points get an explicit empty pen rather than being left out, because
        # a list has to cover every point.
        #
        # Manual outranks flagged where a marker would carry both: the warnings
        # that survive an override describe the SIGNAL, and the outline here is
        # answering "did a person place this point", which is the more useful
        # thing to know at a glance while reviewing.
        if flagged_frames or manual_frames:
            def pen_for(f):
                if f in manual_frames:
                    return pg.mkPen(_MANUAL_OUTLINE, width=1.5)
                if f in flagged_frames:
                    return pg.mkPen(_WARN_OUTLINE, width=1.5)
                return pg.mkPen(None)

            kwargs["pen"] = [pen_for(f) for f in frame_list]
        # Omitting brush=/pen= entirely falls back to the item's own
        # construction-time defaults, which is how a redetect implicitly resets
        # excluded and flagged styling.
        item.setData(x=frame_list, y=y, **kwargs)

    def _populate_unmatched(
        self,
        orig_frames: list[int],
        orig_data: np.ndarray | None,
        disp_frames: list[int],
        disp_data: np.ndarray | None,
        item: pg.ScatterPlotItem,
    ) -> None:
        if orig_data is None or disp_data is None:
            item.setData(x=[], y=[])
            return
        xs: list[float] = []
        ys: list[float] = []
        for f in orig_frames:
            xs.append(float(f))
            ys.append(float(orig_data[f - self._in_point]))
        for f in disp_frames:
            xs.append(float(f))
            ys.append(float(disp_data[f - self._in_point]))
        if xs:
            item.setData(x=xs, y=ys)
        else:
            item.setData(x=[], y=[])

    def _update_connectors(
        self, pairs: list[LatencyPair], excluded_idx: frozenset[int] = frozenset()
    ) -> None:
        rng = self._ydata_max - self._ydata_min
        self._connector_y_level = self._ydata_max + rng * 0.06
        y_level = self._connector_y_level
        xs: list[float] = []
        ys: list[float] = []
        exc_xs: list[float] = []
        exc_ys: list[float] = []
        for i, p in enumerate(pairs):
            tx, ty = (exc_xs, exc_ys) if i in excluded_idx else (xs, ys)
            tx += [float(p.orig_frame), float(p.disp_frame), float("nan")]
            ty += [y_level, y_level, float("nan")]
        self._pair_connectors.setData(x=xs, y=ys)
        self._connector_excluded.setData(x=exc_xs, y=exc_ys)

    def set_excluded_pairs(self, rise_excluded_idx: set[int], fall_excluded_idx: set[int]) -> None:
        """Pure rendering hint from MainWindow: mute markers/connectors for
        manually-excluded pairs. Must NOT emit pairs_updated — MainWindow
        clears its exclude sets in response to that signal, so re-emitting
        here would immediately wipe the sets this call is applying."""
        self._rise_excluded_idx = set(rise_excluded_idx)
        self._fall_excluded_idx = set(fall_excluded_idx)

        # Mute every marker frame the excluded pairs own, not just their
        # anchors, or half of an excluded transition would stay bright.
        def excluded_frames(idx_set: set[int], pairs: list[LatencyPair]) -> frozenset[int]:
            return frozenset(
                f
                for i in idx_set if i < len(pairs)
                for orig in (True, False)
                for f in self._pair_frames(pairs[i], orig=orig)
            )

        self._populate_transitions(
            excluded_frames(self._rise_excluded_idx, self._rise_pairs),
            excluded_frames(self._fall_excluded_idx, self._fall_pairs),
        )

        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        active_pairs: list[LatencyPair] = []
        active_excluded: set[int] = set()
        if show_r:
            base = len(active_pairs)
            active_pairs.extend(self._rise_pairs)
            active_excluded |= {base + i for i in self._rise_excluded_idx if i < len(self._rise_pairs)}
        if show_f:
            base = len(active_pairs)
            active_pairs.extend(self._fall_pairs)
            active_excluded |= {base + i for i in self._fall_excluded_idx if i < len(self._fall_pairs)}
        self._update_connectors(active_pairs, active_excluded)
