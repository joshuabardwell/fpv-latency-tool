from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QPixmap

from core.roi import ROI
from ui.roi_frame_view import RoiFrameView


def make_view(qtbot):
    """View sized 400x500 (500 = the widget's minimum height) showing a
    200x100 frame -> scaled to 400x200, centered with a 150 px vertical
    offset. Scale factor label->frame is 0.5x."""
    view = RoiFrameView()
    qtbot.addWidget(view)
    view.resize(400, 500)
    pm = QPixmap(200, 100)
    pm.fill(Qt.GlobalColor.darkGray)
    view.set_frame(pm, 200, 100)
    return view


def drag(qtbot, view, x1, y1, x2, y2):
    qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(x1, y1))
    qtbot.mouseMove(view, QPoint(x2, y2))
    qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(x2, y2))


class TestRoiDrawing:
    def test_drag_creates_roi_and_emits(self, qtbot):
        view = make_view(qtbot)
        view.draw_mode = "original"
        emitted = []
        view.roi_changed.connect(lambda name, roi: emitted.append((name, roi)))
        drag(qtbot, view, 100, 200, 300, 290)
        roi = view.get_roi("original")
        assert roi == ROI(50, 25, 100, 45)
        assert emitted == [("original", roi)]

    def test_aborted_click_restores_previous_roi(self, qtbot):
        """Regression: a stray click in draw mode used to silently delete
        the existing ROI without any notification."""
        view = make_view(qtbot)
        prev = ROI(10, 10, 50, 30)
        view.set_roi("original", prev)
        view.draw_mode = "original"
        emitted = []
        view.roi_changed.connect(lambda name, roi: emitted.append((name, roi)))
        drag(qtbot, view, 200, 250, 202, 251)  # 1x0 frame-px drag -> aborted
        assert view.get_roi("original") == prev
        assert emitted == []

    def test_draw_modes_are_independent(self, qtbot):
        view = make_view(qtbot)
        view.draw_mode = "display"
        drag(qtbot, view, 100, 200, 300, 290)
        assert view.get_roi("display") is not None
        assert view.get_roi("original") is None

    def test_undo_restores_previous_state(self, qtbot):
        view = make_view(qtbot)
        first = ROI(10, 10, 50, 30)
        second = ROI(20, 20, 60, 40)
        view.set_roi("original", first)
        view.set_roi("original", second)
        assert view.undo_roi() is True
        assert view.get_roi("original") == first
        assert view.undo_roi() is False  # single level

    def test_clear_rois(self, qtbot):
        view = make_view(qtbot)
        view.set_roi("original", ROI(10, 10, 50, 30))
        view.clear_rois()
        assert view.get_roi("original") is None
        assert view.get_roi("display") is None
