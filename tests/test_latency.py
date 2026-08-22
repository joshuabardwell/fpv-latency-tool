from core.latency import LatencyPair, default_max_latency_frames, pair_transitions


def frames(pairs):
    return [(p.orig_frame, p.disp_frame) for p in pairs]


class TestLatencyPair:
    def test_delta_frames(self):
        assert LatencyPair(10, 13, "rising").delta_frames() == 3

    def test_delta_ms(self):
        assert LatencyPair(10, 13, "rising").delta_ms(30.0) == 100.0

    def test_zero_latency(self):
        assert LatencyPair(10, 10, "rising").delta_ms(30.0) == 0.0


class TestPairTransitions:
    def test_basic_pairing(self):
        pairs, uo, ud = pair_transitions([10, 25], [13, 28], "rising")
        assert frames(pairs) == [(10, 13), (25, 28)]
        assert uo == [] and ud == []

    def test_polarity_carried_through(self):
        pairs, _, _ = pair_transitions([10], [13], "falling")
        assert pairs[0].polarity == "falling"

    def test_same_frame_pairs_as_zero_latency(self):
        pairs, uo, ud = pair_transitions([10], [10], "rising")
        assert frames(pairs) == [(10, 10)]
        assert uo == [] and ud == []

    def test_disp_before_orig_never_pairs(self):
        pairs, uo, ud = pair_transitions([10], [7], "rising")
        assert pairs == []
        assert uo == [10] and ud == [7]

    def test_each_disp_used_once(self):
        # Two orig transitions compete for one disp: earliest orig wins.
        pairs, uo, ud = pair_transitions([10, 12], [13], "rising")
        assert frames(pairs) == [(10, 13)]
        assert uo == [12] and ud == []

    def test_max_frames_rejects_far_pair(self):
        pairs, uo, ud = pair_transitions([10], [30], "rising", max_frames=5)
        assert pairs == []
        assert uo == [10] and ud == [30]

    def test_max_frames_leaves_disp_for_later_orig(self):
        # disp 30 too far from orig 10 but in range of orig 28.
        pairs, uo, ud = pair_transitions([10, 28], [30], "rising", max_frames=5)
        assert frames(pairs) == [(28, 30)]
        assert uo == [10] and ud == []

    def test_unsorted_input(self):
        pairs, uo, ud = pair_transitions([25, 10], [28, 13], "rising")
        assert frames(pairs) == [(10, 13), (25, 28)]

    def test_empty_inputs(self):
        assert pair_transitions([], [], "rising") == ([], [], [])
        pairs, uo, ud = pair_transitions([5], [], "rising")
        assert (pairs, uo, ud) == ([], [5], [])
        pairs, uo, ud = pair_transitions([], [5], "rising")
        assert (pairs, uo, ud) == ([], [], [5])

    def test_greedy_takes_nearest_following(self):
        pairs, _, ud = pair_transitions([10], [12, 20], "rising")
        assert frames(pairs) == [(10, 12)]
        assert ud == [20]


class TestDefaultMaxLatencyFrames:
    def test_none_period_returns_zero_unlimited(self):
        assert default_max_latency_frames(None) == 0

    def test_half_of_even_period(self):
        assert default_max_latency_frames(20.0) == 10

    def test_rounds_to_nearest_frame(self):
        assert default_max_latency_frames(21.4) == 11
