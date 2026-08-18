import numpy as np

from core.detection import apply_min_spacing, find_falling, find_rising


def square_wave():
    """Dark 0-9, bright 10-24, dark 25-39. Rising at 10, falling at 25."""
    data = np.full(40, 20.0, dtype=np.float32)
    data[10:25] = 220.0
    return data


class TestFindTransitions:
    def test_rising_edge_frame(self):
        assert find_rising(square_wave(), delta=50) == [10]

    def test_falling_edge_frame(self):
        assert find_falling(square_wave(), delta=50) == [25]

    def test_no_transitions_when_delta_too_high(self):
        assert find_rising(square_wave(), delta=300) == []

    def test_short_input(self):
        assert find_rising(np.array([1.0]), delta=5) == []
        assert find_rising(np.array([], dtype=np.float32), delta=5) == []

    def test_gradual_ramp_collapses_to_steepest_step(self):
        # Multi-frame ramp where every step exceeds delta: one event,
        # at the steepest step (30 -> 130 is diff index 2 -> frame 3).
        data = np.array([10, 20, 30, 130, 200, 210, 210, 210], dtype=np.float32)
        assert find_rising(data, delta=9) == [3]

    def test_slow_fade_below_delta_missed(self):
        # Documented limitation: cumulative change 100 but per-frame step 10.
        data = np.arange(0, 110, 10, dtype=np.float32)
        assert find_rising(data, delta=50) == []

    def test_two_separate_rises(self):
        data = np.full(30, 20.0, dtype=np.float32)
        data[5:10] = 220.0
        data[20:25] = 220.0
        assert find_rising(data, delta=50) == [5, 20]
        assert find_falling(data, delta=50) == [10, 25]


class TestApplyMinSpacing:
    def test_disabled(self):
        assert apply_min_spacing([1, 2, 3], 1) == [1, 2, 3]

    def test_empty(self):
        assert apply_min_spacing([], 5) == []

    def test_drops_close_followers(self):
        assert apply_min_spacing([10, 12, 20], 5) == [10, 20]

    def test_spacing_measured_from_last_kept(self):
        # 14 dropped (4 < 5 from 10), 18 kept (8 >= 5 from 10).
        assert apply_min_spacing([10, 14, 18], 5) == [10, 18]
