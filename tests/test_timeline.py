from ui.timeline import TimelineWidget


def make_timeline(qtbot, frames=100):
    tl = TimelineWidget()
    qtbot.addWidget(tl)
    tl.reset(frames)
    return tl


class TestTimeline:
    def test_reset_spans_full_range(self, qtbot):
        tl = make_timeline(qtbot)
        assert tl.current_frame == 0
        assert tl.in_point == 0
        assert tl.out_point == 99

    def test_step_clamps_to_out_point(self, qtbot):
        tl = make_timeline(qtbot)
        tl.step(500)
        assert tl.current_frame == 99

    def test_step_clamps_to_in_point(self, qtbot):
        tl = make_timeline(qtbot)
        tl.set_in_point(10)
        tl.step(-500)
        assert tl.current_frame == 10

    def test_in_point_cannot_pass_out_point(self, qtbot):
        tl = make_timeline(qtbot)
        tl.set_out_point(50)
        tl.set_in_point(80)
        assert tl.in_point == 49

    def test_out_point_cannot_pass_in_point(self, qtbot):
        tl = make_timeline(qtbot)
        tl.set_in_point(50)
        tl.set_out_point(10)
        assert tl.out_point == 51

    def test_in_point_pushes_playhead_forward(self, qtbot):
        tl = make_timeline(qtbot)
        moved = []
        tl.frame_changed.connect(moved.append)
        tl.set_in_point(20)
        assert tl.current_frame == 20
        assert moved == [20]

    def test_set_frame_is_silent(self, qtbot):
        tl = make_timeline(qtbot)
        moved = []
        tl.frame_changed.connect(moved.append)
        tl.set_frame(42)
        assert tl.current_frame == 42
        assert moved == []

    def test_single_frame_video_pins_in_out(self, qtbot):
        """Regression: set_out_point's +1 floor pushed out_point to 1 on a
        1-frame video (only frame 0 exists), making extraction request a
        frame past the end."""
        tl = make_timeline(qtbot, frames=1)
        tl.set_out_point(5)
        tl.set_in_point(3)
        assert (tl.in_point, tl.out_point) == (0, 0)
        tl._apply_drag(50)
        assert tl.current_frame == 0

    def test_set_frame_clamps(self, qtbot):
        tl = make_timeline(qtbot)
        tl.set_frame(1000)
        assert tl.current_frame == 99
        tl.set_frame(-5)
        assert tl.current_frame == 0
