"""
ZoomBarWidget — Premiere-style zoom/pan control for the brightness graph.

Sits directly above TimelineWidget and shares its outer scale (the whole
loaded video, frame 0..frame_count-1), so the two widgets' pixel positions
line up. Within that outer track it draws a second, brighter marker showing
the *analysis boundary* — the frame range actually plotted on the graph
(BrightnessGraphWidget's last set_data call) — and a further highlight for
the *visible zoom window* with two draggable end-handles.

Visual layout (20 px tall widget):

  [    outer track: whole video, 0..frame_count-1    ]
  [        ▓▓▓▓▓ analysis boundary marker ▓▓▓▓▓       ]
  [           ■███ zoom window █████■                 ]
              start handle    end handle

Handles and the zoom-window fill are only drawn/interactive once an analysis
boundary exists (analysis_hi > analysis_lo) — before that (or after
clear_data) the widget shows only the flat outer track and is inert.

Dragging:
  start/end handle  – resizes the zoom window, clamped to the analysis
                       boundary and a minimum width floor.
  zoom-window fill   – pans the window (both handles move together),
                       clamped to the analysis boundary, width unchanged.
  double-click       – resets the zoom window to the full analysis boundary.

Signals:
  range_changed(float, float) – the visible zoom window changed via user
                                 interaction (handle drag, pan drag, or
                                 double-click reset). Never emitted from the
                                 programmatic setters (reset, set_range,
                                 set_analysis_bounds).
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QColor, QPainter

from core.view_range import MIN_ZOOM_FRAMES, clamp_range, pan_range

_MARGIN  = 10   # px reserved at each end so handles don't clip the edge
_TRACK_H = 10   # track bar height
_HND_W   = 6    # width of handle bar (draw)
_HND_TAB = 8    # length of the bracket's top tab
_HND_EXT = 3    # how many px the handle sticks above/below the track
_HIT_R   = 9    # hit-test radius around a handle

_COL_TRACK    = QColor("#333333")  # outer track: no data
_COL_ANALYSIS = QColor("#4d4d4d")  # analysis boundary marker: data exists here
_COL_ZOOM     = QColor("#1a55cc")  # current visible zoom window
_COL_HANDLE   = QColor("#dddddd")


class ZoomBarWidget(QWidget):
    range_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setMouseTracking(True)

        self._lo = 0.0
        self._hi = 0.0
        self._analysis_lo = 0.0
        self._analysis_hi = 0.0
        self._start = 0.0
        self._end = 0.0

        self._drag_target: str | None = None  # "start" | "end" | "middle"
        self._drag_start_x = 0.0
        self._drag_start_range = (0.0, 0.0)

    # ---------------------------------------------------------------- public API

    def reset(self, frame_count: int) -> None:
        """Call when a new file is loaded. Sets the outer (whole-video) scale."""
        self._lo = 0.0
        self._hi = float(max(0, frame_count - 1))
        self.update()

    def set_analysis_bounds(self, lo: float, hi: float) -> None:
        """Sets the analysis-boundary marker and resets the visible zoom
        window to it (new analysis results reset zoom to 100%). Silent —
        does not emit range_changed; this mirrors the graph's own
        already-reset visible range, it doesn't originate a change."""
        self._analysis_lo = lo
        self._analysis_hi = hi
        self._start = lo
        self._end = hi if hi > lo else lo
        self.update()

    def set_range(self, start: float, end: float) -> None:
        """Silent programmatic setter (same convention as
        TimelineWidget.set_frame): mirrors an externally-driven visible-range
        change (e.g. graph wheel-zoom, click-drag pan, playhead recenter)."""
        if self._analysis_hi <= self._analysis_lo:
            return
        start, end = clamp_range(start, end, self._analysis_lo, self._analysis_hi, MIN_ZOOM_FRAMES)
        if (start, end) != (self._start, self._end):
            self._start, self._end = start, end
            self.update()

    @property
    def visible_start(self) -> float:
        return self._start

    @property
    def visible_end(self) -> float:
        return self._end

    @property
    def lo(self) -> float:
        return self._lo

    @property
    def hi(self) -> float:
        return self._hi

    @property
    def analysis_lo(self) -> float:
        return self._analysis_lo

    @property
    def analysis_hi(self) -> float:
        return self._analysis_hi

    # --------------------------------------------------------------- geometry

    def _track_geom(self) -> tuple[float, float, float, float]:
        """Return (track_x, track_y, track_w, track_h)."""
        w, h = self.width(), self.height()
        ty = (h - _TRACK_H) // 2
        return _MARGIN, ty, max(1, w - 2 * _MARGIN), _TRACK_H

    def _frame_to_x(self, frame: float) -> float:
        tx, _, tw, _ = self._track_geom()
        if self._hi <= self._lo:
            return tx
        return tx + (frame - self._lo) * tw / (self._hi - self._lo)

    def _x_to_frame(self, x: float) -> float:
        tx, _, tw, _ = self._track_geom()
        if tw <= 0 or self._hi <= self._lo:
            return self._lo
        frame = self._lo + (x - tx) * (self._hi - self._lo) / tw
        return max(self._lo, min(frame, self._hi))

    def _hit_test(self, x: float) -> str | None:
        if self._analysis_hi <= self._analysis_lo:
            return None
        start_x = self._frame_to_x(self._start)
        end_x = self._frame_to_x(self._end)
        if abs(x - start_x) <= _HIT_R:
            return "start"
        if abs(x - end_x) <= _HIT_R:
            return "end"
        if start_x < x < end_x:
            return "middle"
        return None

    # ---------------------------------------------------------------- painting

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        tx, ty, tw, th = self._track_geom()
        p.fillRect(int(tx), int(ty), int(tw), int(th), _COL_TRACK)

        if self._analysis_hi > self._analysis_lo:
            a0 = self._frame_to_x(self._analysis_lo)
            a1 = self._frame_to_x(self._analysis_hi)
            p.fillRect(int(a0), int(ty), max(1, int(a1 - a0)), int(th), _COL_ANALYSIS)

            s0 = self._frame_to_x(self._start)
            s1 = self._frame_to_x(self._end)
            p.fillRect(int(s0), int(ty), max(1, int(s1 - s0)), int(th), _COL_ZOOM)

            self._draw_start_handle(p, s0, ty, th)
            self._draw_end_handle(p, s1, ty, th)

        p.end()

    def _draw_start_handle(self, p: QPainter, x: float, ty: float, th: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_COL_HANDLE)
        p.drawRect(int(x - _HND_W / 2), int(ty - _HND_EXT), _HND_W, int(th + 2 * _HND_EXT))
        p.drawRect(int(x), int(ty - _HND_EXT), _HND_TAB, _HND_EXT)  # tab extending right → [

    def _draw_end_handle(self, p: QPainter, x: float, ty: float, th: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_COL_HANDLE)
        p.drawRect(int(x - _HND_W / 2), int(ty - _HND_EXT), _HND_W, int(th + 2 * _HND_EXT))
        p.drawRect(int(x - _HND_TAB), int(ty - _HND_EXT), _HND_TAB, _HND_EXT)  # tab extending left → ]

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        hit = self._hit_test(x)
        self._drag_target = hit
        if hit == "middle":
            self._drag_start_x = x
            self._drag_start_range = (self._start, self._end)
        elif hit in ("start", "end"):
            self._apply_drag(x)
        self._update_cursor(hit)

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._drag_target:
            self._apply_drag(x)
        else:
            self._update_cursor(self._hit_test(x))

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_target = None
        self._update_cursor(self._hit_test(event.position().x()))

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._reset_to_full()

    def _reset_to_full(self) -> None:
        if self._analysis_hi <= self._analysis_lo:
            return
        self._start, self._end = self._analysis_lo, self._analysis_hi
        self.update()
        self.range_changed.emit(self._start, self._end)

    def _update_cursor(self, hit: str | None) -> None:
        if hit in ("start", "end"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif hit == "middle":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _apply_drag(self, x: float) -> None:
        if self._analysis_hi <= self._analysis_lo:
            return
        if self._drag_target == "start":
            frame = max(self._analysis_lo, min(self._x_to_frame(x), self._end - MIN_ZOOM_FRAMES))
            if frame != self._start:
                self._start = frame
                self.update()
                self.range_changed.emit(self._start, self._end)
        elif self._drag_target == "end":
            frame = min(self._analysis_hi, max(self._x_to_frame(x), self._start + MIN_ZOOM_FRAMES))
            if frame != self._end:
                self._end = frame
                self.update()
                self.range_changed.emit(self._start, self._end)
        elif self._drag_target == "middle":
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            _, _, tw, _ = self._track_geom()
            dx = x - self._drag_start_x
            delta = dx * (self._hi - self._lo) / tw if tw > 0 else 0.0
            start0, end0 = self._drag_start_range
            new_start, new_end = pan_range(start0, end0, delta, self._analysis_lo, self._analysis_hi)
            if (new_start, new_end) != (self._start, self._end):
                self._start, self._end = new_start, new_end
                self.update()
                self.range_changed.emit(self._start, self._end)
