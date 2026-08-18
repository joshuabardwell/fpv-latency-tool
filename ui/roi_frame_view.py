"""
Interactive frame-display widget that supports click-and-drag ROI selection.

Usage:
    widget.draw_mode = "original"   # or "display" or None
    # user drags a rectangle; widget emits roi_changed(name, ROI)
    widget.draw_mode = None         # disable drawing
"""

from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from core.roi import ROI

_COLORS: dict[str, QColor] = {
    "original": QColor(0, 230, 0),    # bright green
    "display":  QColor(255, 160, 0),  # amber
}


class RoiFrameView(QLabel):
    """QLabel subclass with ROI drawing overlay."""

    # emits (name, ROI) whenever the user finishes drawing a rectangle
    roi_changed = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__("Open a video file to begin", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(500)
        self.setMinimumWidth(1)
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.setSizePolicy(sp)
        self.setStyleSheet("background-color: #222; color: #888;")
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._base_pixmap: QPixmap | None = None
        self._orig_w = 1
        self._orig_h = 1

        self._roi_original: ROI | None = None
        self._roi_display:  ROI | None = None

        self._draw_mode: str | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_end:   tuple[int, int] | None = None
        self._dragging = False

        # One level of undo: snapshot of (roi_original, roi_display) before last edit
        self._undo: tuple[ROI | None, ROI | None] | None = None

    # ------------------------------------------------------------------ public

    def set_frame(self, pixmap: QPixmap, orig_w: int, orig_h: int) -> None:
        self._base_pixmap = pixmap
        self._orig_w = orig_w
        self._orig_h = orig_h
        self._redraw()

    @property
    def draw_mode(self) -> str | None:
        return self._draw_mode

    @draw_mode.setter
    def draw_mode(self, mode: str | None) -> None:
        self._draw_mode = mode
        self._dragging = False
        self._drag_start = self._drag_end = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if mode else Qt.CursorShape.ArrowCursor
        )
        self._redraw()

    def get_roi(self, name: str) -> ROI | None:
        return self._roi_original if name == "original" else self._roi_display

    def clear_rois(self) -> None:
        self._roi_original = None
        self._roi_display = None
        self._undo = None
        self._redraw()

    def undo_roi(self) -> bool:
        """Restore the ROI state from before the last draw.  Returns True if there was something to undo."""
        if self._undo is None:
            return False
        self._roi_original, self._roi_display = self._undo
        self._undo = None
        self._redraw()
        return True

    def set_roi(self, name: str, roi: ROI) -> None:
        self._save_undo()
        if name == "original":
            self._roi_original = roi
        else:
            self._roi_display = roi
        self._redraw()
        self.roi_changed.emit(name, roi)

    def _save_undo(self) -> None:
        self._undo = (self._roi_original, self._roi_display)

    # ------------------------------------------------------- coordinate mapping

    def _displayed_size(self):
        """QSize of _base_pixmap after scaling to fit the widget (aspect-ratio preserved)."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return None
        return self._base_pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)

    def _pixmap_rect(self) -> QRect | None:
        """Rect of the displayed pixmap within the label (centered, scaled to fit)."""
        ds = self._displayed_size()
        if ds is None:
            return None
        pw, ph = ds.width(), ds.height()
        x = max(0, (self.width()  - pw) // 2)
        y = max(0, (self.height() - ph) // 2)
        return QRect(x, y, pw, ph)

    def _label_to_frame(self, lx: float, ly: float) -> tuple[int, int] | None:
        """Map a label-space point to original-frame pixel coordinates."""
        r = self._pixmap_rect()
        if r is None or r.width() == 0 or r.height() == 0:
            return None
        px = max(0, min(int(lx) - r.x(), r.width()  - 1))
        py = max(0, min(int(ly) - r.y(), r.height() - 1))
        fx = int(px * self._orig_w / r.width())
        fy = int(py * self._orig_h / r.height())
        return fx, fy

    # ----------------------------------------------------------------- drawing

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._base_pixmap is not None:
            self._redraw()

    def _redraw(self) -> None:
        if self._base_pixmap is None:
            return
        ds = self._displayed_size()
        if ds is None or ds.width() <= 0 or ds.height() <= 0:
            return
        pixmap = self._base_pixmap.scaled(
            ds, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        if pixmap.isNull():
            return
        pw, ph = pixmap.width(), pixmap.height()

        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        def scale_to_pixmap(roi: ROI) -> QRect:
            rx = int(roi.x     * pw / self._orig_w)
            ry = int(roi.y     * ph / self._orig_h)
            rw = max(1, int(roi.width  * pw / self._orig_w))
            rh = max(1, int(roi.height * ph / self._orig_h))
            return QRect(rx, ry, rw, rh)

        def draw_roi(roi: ROI | None, color: QColor, label: str) -> None:
            if roi is None:
                return
            rect = scale_to_pixmap(roi)
            fill = QColor(color)
            fill.setAlpha(40)
            p.fillRect(rect, QBrush(fill))
            p.setPen(QPen(color, 2))
            p.drawRect(rect)
            # text with a 1-pixel black drop shadow for readability
            p.setPen(QPen(Qt.GlobalColor.black))
            p.drawText(rect.x() + 5, rect.y() + 15, label)
            p.setPen(QPen(color))
            p.drawText(rect.x() + 4, rect.y() + 14, label)

        draw_roi(self._roi_original, _COLORS["original"], "Original")
        draw_roi(self._roi_display,  _COLORS["display"],  "Display")

        # In-progress drag rectangle
        if self._dragging and self._drag_start and self._drag_end and self._draw_mode:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_end
            rx = int(min(x1, x2) * pw / self._orig_w)
            ry = int(min(y1, y2) * ph / self._orig_h)
            rw = max(1, int(abs(x2 - x1) * pw / self._orig_w))
            rh = max(1, int(abs(y2 - y1) * ph / self._orig_h))
            color = _COLORS.get(self._draw_mode, QColor(255, 255, 255))
            p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            p.drawRect(rx, ry, rw, rh)

        p.end()
        self.setPixmap(pixmap)

    # ------------------------------------------------------- mouse event handlers

    def mousePressEvent(self, event) -> None:
        if self._draw_mode is None or event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._label_to_frame(event.position().x(), event.position().y())
        if pt is not None:
            self._save_undo()
            # Clear the existing ROI immediately so the drag starts clean
            if self._draw_mode == "original":
                self._roi_original = None
            else:
                self._roi_display = None
            self._drag_start = pt
            self._drag_end = pt
            self._dragging = True
            self._redraw()

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging:
            return
        pt = self._label_to_frame(event.position().x(), event.position().y())
        if pt is not None:
            self._drag_end = pt
            self._redraw()

    def mouseReleaseEvent(self, event) -> None:
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._label_to_frame(event.position().x(), event.position().y())
        if pt is not None:
            self._drag_end = pt

        start, end = self._drag_start, self._drag_end
        self._dragging = False
        self._drag_start = self._drag_end = None

        if start and end:
            x = min(start[0], end[0])
            y = min(start[1], end[1])
            w = abs(end[0] - start[0])
            h = abs(end[1] - start[1])
            if w > 4 and h > 4:
                roi = ROI(x, y, w, h)
                if self._draw_mode == "original":
                    self._roi_original = roi
                else:
                    self._roi_display = roi
                self._redraw()
                self.roi_changed.emit(self._draw_mode, roi)
            else:
                # Aborted drag (accidental click): the press cleared the
                # active ROI — restore it from the pre-press snapshot so a
                # stray click doesn't silently delete a selection.
                if self._undo is not None:
                    prev_orig, prev_disp = self._undo
                    if self._draw_mode == "original":
                        self._roi_original = prev_orig
                    else:
                        self._roi_display = prev_disp
                self._redraw()
