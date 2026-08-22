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

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPainterPath

from core.detection import apply_min_spacing, find_falling, find_rising
from core.latency import LatencyPair, pair_transitions

_GREEN  = (0, 230, 0)
_AMBER  = (255, 160, 0)
_YELLOW = (255, 204, 0)
_RED    = (220, 50, 50)

# Lighter tints for transition markers, distinct from the line colors above
# so a marker doesn't disappear into a noisy/jittery line of the same hue.
_GREEN_MARKER = (64, 236, 64)
_AMBER_MARKER = (255, 184, 64)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setBackground("#1a1a1a")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()

        pi = self.getPlotItem()
        pi.hideAxis("bottom")
        pi.getAxis("left").setWidth(35)
        pi.setYRange(0, 255, padding=0.04)
        pi.disableAutoRange()
        self._vb = pi.getViewBox()

        # Signal lines
        self._line_orig = self.plot(pen=pg.mkPen(_GREEN, width=1.5))
        self._line_disp = self.plot(pen=pg.mkPen(_AMBER, width=1.5))

        # Transition scatter items: rising ▲ and falling ▼, per signal
        self._sc_rise_orig = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_GREEN_MARKER), size=9, symbol="t1")
        self._sc_fall_orig = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_GREEN_MARKER), size=9, symbol="t")
        self._sc_rise_disp = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_AMBER_MARKER), size=9, symbol="t1")
        self._sc_fall_disp = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_AMBER_MARKER), size=9, symbol="t")
        for sc in (self._sc_rise_orig, self._sc_fall_orig, self._sc_rise_disp, self._sc_fall_disp):
            self.addItem(sc)

        # Unmatched transition markers (overlaid in red)
        self._sc_unmatched_rise = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_RED), size=10, symbol="t1")
        self._sc_unmatched_fall = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(_RED), size=10, symbol="t")
        self.addItem(self._sc_unmatched_rise)
        self.addItem(self._sc_unmatched_fall)

        # Connector lines linking paired orig→disp markers
        self._pair_connectors = self.plot(pen=pg.mkPen((200, 200, 200, 100), width=1))

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
        self._ydata_min: float = 0.0
        self._ydata_max: float = 255.0

        # Cached per-polarity transition frame lists
        self._rise_orig_frames: list[int] = []
        self._fall_orig_frames: list[int] = []
        self._rise_disp_frames: list[int] = []
        self._fall_disp_frames: list[int] = []

        # Pairing state
        self._rise_pairs: list[LatencyPair] = []
        self._fall_pairs: list[LatencyPair] = []
        self._rise_orig_unmatched: list[int] = []
        self._fall_orig_unmatched: list[int] = []
        self._rise_disp_unmatched: list[int] = []
        self._fall_disp_unmatched: list[int] = []

        self._polarity = "both"
        self._transition_frames: list[int] = []

    # ------------------------------------------------------------------ public

    def set_data(self, orig: np.ndarray, disp: np.ndarray, in_point: int) -> float:
        """Load brightness arrays and run detection. Returns the auto-computed delta."""
        self._in_point  = in_point
        self._n         = len(orig)
        self._orig_data = orig
        self._disp_data = disp

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

    def set_polarity(self, mode: str) -> None:
        """mode: 'both' | 'rising' | 'falling'"""
        self._polarity = mode
        self._apply_polarity()

    def set_frame(self, frame: int) -> None:
        if self._playhead_marker.isVisible():
            self._playhead_stalk.setData(x=[frame, frame], y=[self._playhead_apex_y, self._playhead_y1])
            self._playhead_marker.setData(x=[frame], y=[self._playhead_y0])

    def clear_data(self) -> None:
        self._orig_data = self._disp_data = None
        self._transition_frames = []
        # Fresh list per attribute — chained `a = b = []` would alias them.
        self._rise_orig_frames, self._fall_orig_frames = [], []
        self._rise_disp_frames, self._fall_disp_frames = [], []
        self._rise_pairs, self._fall_pairs = [], []
        self._rise_orig_unmatched, self._fall_orig_unmatched = [], []
        self._rise_disp_unmatched, self._fall_disp_unmatched = [], []
        for item in (self._line_orig, self._line_disp):
            item.setData(x=[], y=[])
        for sc in (self._sc_rise_orig, self._sc_fall_orig,
                   self._sc_rise_disp, self._sc_fall_disp,
                   self._sc_unmatched_rise, self._sc_unmatched_fall):
            sc.setData(x=[], y=[])
        self._pair_connectors.setData(x=[], y=[])
        self._playhead_stalk.setVisible(False)
        self._playhead_marker.setVisible(False)
        self.setYRange(0, 255, padding=0.04)
        self._n = 0
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

    def get_pairs(self) -> list[LatencyPair]:
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        result: list[LatencyPair] = []
        if show_r:
            result.extend(self._rise_pairs)
        if show_f:
            result.extend(self._fall_pairs)
        return sorted(result, key=lambda p: p.orig_frame)

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

    # --------------------------------------------------------------- internals

    def _redetect(self) -> None:
        sp = self._min_spacing
        self._rise_orig_frames = apply_min_spacing(
            [self._in_point + i for i in find_rising(self._orig_data,  self._delta)], sp)
        self._fall_orig_frames = apply_min_spacing(
            [self._in_point + i for i in find_falling(self._orig_data, self._delta)], sp)
        self._rise_disp_frames = apply_min_spacing(
            [self._in_point + i for i in find_rising(self._disp_data,  self._delta)], sp)
        self._fall_disp_frames = apply_min_spacing(
            [self._in_point + i for i in find_falling(self._disp_data, self._delta)], sp)

        max_fr = self._max_latency if self._max_latency > 0 else None
        self._rise_pairs, self._rise_orig_unmatched, self._rise_disp_unmatched = \
            pair_transitions(self._rise_orig_frames, self._rise_disp_frames, "rising",  max_frames=max_fr)
        self._fall_pairs, self._fall_orig_unmatched, self._fall_disp_unmatched = \
            pair_transitions(self._fall_orig_frames, self._fall_disp_frames, "falling", max_frames=max_fr)

        self._populate(self._rise_orig_frames, self._orig_data, self._sc_rise_orig)
        self._populate(self._fall_orig_frames, self._orig_data, self._sc_fall_orig)
        self._populate(self._rise_disp_frames, self._disp_data, self._sc_rise_disp)
        self._populate(self._fall_disp_frames, self._disp_data, self._sc_fall_disp)
        self._apply_polarity()

    def _apply_polarity(self) -> None:
        show_r = self._polarity in ("both", "rising")
        show_f = self._polarity in ("both", "falling")
        self._sc_rise_orig.setVisible(show_r)
        self._sc_rise_disp.setVisible(show_r)
        self._sc_fall_orig.setVisible(show_f)
        self._sc_fall_disp.setVisible(show_f)
        self._sc_unmatched_rise.setVisible(show_r)
        self._sc_unmatched_fall.setVisible(show_f)

        frames: set[int] = set()
        if show_r:
            frames.update(self._rise_orig_frames, self._rise_disp_frames)
        if show_f:
            frames.update(self._fall_orig_frames, self._fall_disp_frames)
        self._transition_frames = sorted(frames)

        self._populate_unmatched(
            self._rise_orig_unmatched, self._orig_data,
            self._rise_disp_unmatched, self._disp_data,
            self._sc_unmatched_rise,
        )
        self._populate_unmatched(
            self._fall_orig_unmatched, self._orig_data,
            self._fall_disp_unmatched, self._disp_data,
            self._sc_unmatched_fall,
        )

        active_pairs: list[LatencyPair] = []
        if show_r:
            active_pairs.extend(self._rise_pairs)
        if show_f:
            active_pairs.extend(self._fall_pairs)
        self._update_connectors(active_pairs)

        self.pairs_updated.emit()

    def _populate(self, frame_list: list[int], data: np.ndarray, item: pg.ScatterPlotItem) -> None:
        if frame_list:
            item.setData(
                x=frame_list,
                y=[float(data[f - self._in_point]) for f in frame_list],
            )
        else:
            item.setData(x=[], y=[])

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

    def _update_connectors(self, pairs: list[LatencyPair]) -> None:
        if not pairs:
            self._pair_connectors.setData(x=[], y=[])
            return
        rng = self._ydata_max - self._ydata_min
        y_level = self._ydata_max + rng * 0.06
        xs: list[float] = []
        ys: list[float] = []
        for p in pairs:
            xs += [float(p.orig_frame), float(p.disp_frame), float("nan")]
            ys += [y_level, y_level, float("nan")]
        self._pair_connectors.setData(x=xs, y=ys)
