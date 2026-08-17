"""
Custom timeline widget: playhead scrubber with draggable in/out point handles.

Visual layout (44 px tall widget):

  ▼ playhead triangle
  |
 [■══════════════════════════■]   ← track bar (in-to-out highlighted)
  in handle                  out handle

■  = in/out handle bracket (vertical bar + small extending tab)
══ = highlighted in-to-out range
▼  = yellow playhead (triangle + thin line through track)

Handles:
  In  – vertical bar + right-extending top tab  (looks like [)
  Out – vertical bar + left-extending top tab   (looks like ])

Signals:
  frame_changed(int)     – playhead was dragged or stepped
  in_point_changed(int)  – in handle was dragged
  out_point_changed(int) – out handle was dragged
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QWidget

_MARGIN    = 10   # px reserved at each end so handles don't clip the edge
_TRACK_H   = 12   # track bar height
_HND_W     = 6    # half-width of handle bar (draw)
_HND_TAB   = 10   # length of the bracket's top tab
_HND_EXT   = 4    # how many px the handle sticks above/below the track
_HIT_R     = 9    # hit-test radius around a handle or playhead

_COL_TRACK    = QColor("#333333")
_COL_RANGE    = QColor("#1a55cc")
_COL_HANDLE   = QColor("#dddddd")
_COL_PLAYHEAD = QColor("#ffcc00")


class TimelineWidget(QWidget):
    frame_changed     = pyqtSignal(int)
    in_point_changed  = pyqtSignal(int)
    out_point_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(44)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._frame_count = 0
        self._current_frame = 0
        self._in_point = 0
        self._out_point = 0

        self._drag_target: str | None = None  # "in" | "out" | "playhead"

    # ---------------------------------------------------------------- public API

    def reset(self, frame_count: int) -> None:
        """Call when a new file is loaded.  Resets playhead and in/out to full range."""
        self._frame_count = max(0, frame_count)
        self._current_frame = 0
        self._in_point = 0
        self._out_point = max(0, frame_count - 1)
        self.update()

    def set_frame(self, frame: int) -> None:
        """Move the playhead without emitting frame_changed (programmatic update)."""
        frame = self._clamp(frame)
        if frame != self._current_frame:
            self._current_frame = frame
            self.update()

    def step(self, delta: int) -> None:
        """Advance or rewind one frame, clamped to the in/out range."""
        new_frame = max(self._in_point, min(self._current_frame + delta, self._out_point))
        if new_frame != self._current_frame:
            self._current_frame = new_frame
            self.update()
            self.frame_changed.emit(new_frame)

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def in_point(self) -> int:
        return self._in_point

    @property
    def out_point(self) -> int:
        return self._out_point

    def set_in_point(self, frame: int) -> None:
        """Set in point; pushes the playhead forward if it falls before the new in point."""
        frame = max(0, min(frame, self._out_point - 1))
        if frame != self._in_point:
            self._in_point = frame
            if self._current_frame < self._in_point:
                self._current_frame = self._in_point
                self.frame_changed.emit(self._current_frame)
            self.update()
            self.in_point_changed.emit(frame)

    def set_out_point(self, frame: int) -> None:
        """Set out point; pulls the playhead back if it falls past the new out point."""
        frame = max(self._in_point + 1, min(frame, max(0, self._frame_count - 1)))
        if frame != self._out_point:
            self._out_point = frame
            if self._current_frame > self._out_point:
                self._current_frame = self._out_point
                self.frame_changed.emit(self._current_frame)
            self.update()
            self.out_point_changed.emit(frame)

    # --------------------------------------------------------------- geometry

    def _clamp(self, frame: int) -> int:
        return max(0, min(frame, max(0, self._frame_count - 1)))

    def _track_geom(self) -> tuple[int, int, int, int]:
        """Return (track_x, track_y, track_w, track_h)."""
        w, h = self.width(), self.height()
        ty = (h - _TRACK_H) // 2
        return _MARGIN, ty, max(1, w - 2 * _MARGIN), _TRACK_H

    def _frame_to_x(self, frame: int) -> int:
        tx, _, tw, _ = self._track_geom()
        if self._frame_count <= 1:
            return tx
        return tx + round(frame * tw / (self._frame_count - 1))

    def _x_to_frame(self, x: int) -> int:
        tx, _, tw, _ = self._track_geom()
        if tw <= 0 or self._frame_count <= 1:
            return 0
        return self._clamp(round((x - tx) * (self._frame_count - 1) / tw))

    def _hit_test(self, x: int) -> str | None:
        if self._frame_count == 0:
            return None
        if abs(x - self._frame_to_x(self._in_point)) <= _HIT_R:
            return "in"
        if abs(x - self._frame_to_x(self._out_point)) <= _HIT_R:
            return "out"
        if abs(x - self._frame_to_x(self._current_frame)) <= _HIT_R:
            return "playhead"
        return None

    # ---------------------------------------------------------------- painting

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        tx, ty, tw, th = self._track_geom()

        # Full track background
        p.fillRect(tx, ty, tw, th, _COL_TRACK)

        if self._frame_count > 0:
            in_x   = self._frame_to_x(self._in_point)
            out_x  = self._frame_to_x(self._out_point)
            head_x = self._frame_to_x(self._current_frame)

            # Highlighted in-to-out region
            range_w = max(1, out_x - in_x)
            p.fillRect(in_x, ty, range_w, th, _COL_RANGE)

            self._draw_in_handle(p, in_x, ty, th)
            self._draw_out_handle(p, out_x, ty, th)
            self._draw_playhead(p, head_x, ty, th)

        p.end()

    def _draw_in_handle(self, p: QPainter, x: int, ty: int, th: int) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_COL_HANDLE)
        # Vertical bar
        p.drawRect(x - _HND_W // 2, ty - _HND_EXT, _HND_W, th + 2 * _HND_EXT)
        # Top tab extending to the right  →  [
        p.drawRect(x, ty - _HND_EXT, _HND_TAB, _HND_EXT)

    def _draw_out_handle(self, p: QPainter, x: int, ty: int, th: int) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_COL_HANDLE)
        # Vertical bar
        p.drawRect(x - _HND_W // 2, ty - _HND_EXT, _HND_W, th + 2 * _HND_EXT)
        # Top tab extending to the left  →  ]
        p.drawRect(x - _HND_TAB, ty - _HND_EXT, _HND_TAB, _HND_EXT)

    def _draw_playhead(self, p: QPainter, x: int, ty: int, th: int) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_COL_PLAYHEAD)
        # Thin vertical line through track
        p.drawRect(x - 1, ty, 2, th)
        # Downward-pointing triangle above the track
        tri = QPainterPath()
        tri.moveTo(x - 5, ty - 8)
        tri.lineTo(x + 5, ty - 8)
        tri.lineTo(x,     ty - 1)
        tri.closeSubpath()
        p.drawPath(tri)

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = int(event.position().x())
        hit = self._hit_test(x)
        # Clicking anywhere that isn't an in/out handle moves the playhead
        self._drag_target = hit if hit in ("in", "out") else "playhead"
        self._apply_drag(x)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def mouseMoveEvent(self, event) -> None:
        x = int(event.position().x())
        if self._drag_target:
            self._apply_drag(x)
        else:
            # Hover cursor
            hit = self._hit_test(x)
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if hit else Qt.CursorShape.ArrowCursor
            )

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_target = None
        x = int(event.position().x())
        hit = self._hit_test(x)
        self.setCursor(
            Qt.CursorShape.SizeHorCursor if hit else Qt.CursorShape.ArrowCursor
        )

    def _apply_drag(self, x: int) -> None:
        frame = self._x_to_frame(x)
        if self._drag_target == "in":
            frame = max(0, min(frame, self._out_point - 1))
            if frame != self._in_point:
                self._in_point = frame
                self.in_point_changed.emit(frame)
            if frame != self._current_frame:
                self._current_frame = frame
                self.frame_changed.emit(frame)
            self.update()
        elif self._drag_target == "out":
            frame = max(self._in_point + 1, min(frame, self._frame_count - 1))
            if frame != self._out_point:
                self._out_point = frame
                self.out_point_changed.emit(frame)
            if frame != self._current_frame:
                self._current_frame = frame
                self.frame_changed.emit(frame)
            self.update()
        elif self._drag_target == "playhead":
            frame = max(self._in_point, min(frame, self._out_point))
            if frame != self._current_frame:
                self._current_frame = frame
                self.update()
                self.frame_changed.emit(frame)
