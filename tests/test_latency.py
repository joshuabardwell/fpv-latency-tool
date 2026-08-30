from core.edges import TransitionEdge
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


def edge(anchor, first, full, polarity="rising", warnings=()):
    """A hand-built TransitionEdge. Levels and SNR are irrelevant to the
    latency arithmetic, so they get plausible fillers."""
    return TransitionEdge(
        anchor_frame=anchor, first_frame=first, full_frame=full,
        baseline=20.0, plateau=220.0, polarity=polarity,
        snr=50.0, crossings=1, warnings=warnings,
    )


class TestThreeMetrics:
    def test_each_metric_compares_like_for_like(self):
        # Source ramps 8->12 (anchor 10), display ramps 20->28 (anchor 22).
        # Both metrics share the source's first-light zero point (8):
        # first: 20-8 = 12.  full: 28-8 = 20.  avg: 16.
        p = LatencyPair(10, 22, "rising",
                        orig_edge=edge(10, 8, 12), disp_edge=edge(22, 20, 28))
        assert p.first_delta_frames() == 12.0
        assert p.full_delta_frames() == 20.0
        assert p.avg_delta_frames() == 16.0

    def test_average_can_be_a_half_frame(self):
        # first 12, full 19 (27-8) -> 15.5, a real resolution gain over integer frames.
        p = LatencyPair(10, 22, "rising",
                        orig_edge=edge(10, 8, 12), disp_edge=edge(22, 20, 27))
        assert p.avg_delta_frames() == 15.5

    def test_metrics_differ_from_the_anchor_delta(self):
        """The whole point: the anchor sits somewhere inside the ramp, so it
        matches none of the three unless the transition is instantaneous."""
        p = LatencyPair(10, 22, "rising",
                        orig_edge=edge(10, 8, 12), disp_edge=edge(22, 20, 28))
        assert p.delta_frames() == 12
        assert p.full_delta_frames() != p.delta_frames()

    def test_full_delta_cannot_read_below_first_delta(self):
        """Regression: a source that settles slower than the display used to
        make full-frame latency read BELOW first-pixel latency for the same
        pair — physically backwards. Source ramps 8->12 (4 frames), display
        ramps 20->22 (2 frames): the old formula gave full = 22-12 = 10,
        first = 20-8 = 12, so full < first. Sharing the source's first-light
        as the zero point for both ties the gap between them to the
        display's own ramp (never negative), so full is now guaranteed
        >= first: full = 22-8 = 14."""
        p = LatencyPair(10, 21, "rising",
                        orig_edge=edge(10, 8, 12), disp_edge=edge(21, 20, 22))
        assert p.full_delta_frames() == 14.0
        assert p.full_delta_frames() >= p.first_delta_frames()

    def test_instantaneous_transitions_collapse_to_the_anchor_delta(self):
        """Compatibility: on a square wave first == full == anchor, so all
        three metrics reproduce exactly what the tool reported before."""
        p = LatencyPair(10, 13, "rising",
                        orig_edge=edge(10, 10, 10), disp_edge=edge(13, 13, 13))
        assert p.first_delta_frames() == 3.0
        assert p.avg_delta_frames() == 3.0
        assert p.full_delta_frames() == 3.0

    def test_ms_conversion_matches_frames(self):
        p = LatencyPair(10, 22, "rising",
                        orig_edge=edge(10, 8, 12), disp_edge=edge(22, 20, 28))
        assert p.first_delta_ms(240.0) == 12.0 / 240.0 * 1000.0
        assert p.avg_delta_ms(240.0) == 16.0 / 240.0 * 1000.0

    def test_negative_first_latency_is_reported_not_clamped(self):
        """A display first-light before the source's means the band is too
        tight or an ROI is catching stray light. Hiding it behind a zero would
        hide a real misconfiguration."""
        p = LatencyPair(10, 11, "rising",
                        orig_edge=edge(10, 10, 14), disp_edge=edge(11, 9, 15))
        assert p.first_delta_frames() == -1.0


class TestMissingEdges:
    def test_no_edges_falls_back_to_the_anchor_delta(self):
        p = LatencyPair(10, 13, "rising")
        assert p.first_delta_frames() == 3.0
        assert p.avg_delta_frames() == 3.0
        assert p.full_delta_frames() == 3.0

    def test_one_missing_edge_still_falls_back(self):
        # Half a measurement is not a measurement; don't mix an edge frame
        # against an anchor frame.
        p = LatencyPair(10, 13, "rising", orig_edge=edge(10, 8, 12))
        assert p.first_delta_frames() == 3.0


class TestPairWarnings:
    def test_warnings_union_both_ends_without_duplicates(self):
        p = LatencyPair(10, 13, "rising",
                        orig_edge=edge(10, 10, 10, warnings=("low-snr",)),
                        disp_edge=edge(13, 13, 13, warnings=("low-snr", "slow-ramp")))
        assert p.quality_warnings() == ("low-snr", "slow-ramp")
        assert not p.is_clean()

    def test_clean_pair_has_no_warnings(self):
        p = LatencyPair(10, 13, "rising",
                        orig_edge=edge(10, 10, 10), disp_edge=edge(13, 13, 13))
        assert p.quality_warnings() == ()
        assert p.is_clean()

    def test_edgeless_pair_is_clean(self):
        assert LatencyPair(10, 13, "rising").is_clean()


class TestPairTransitionsEdgeAttachment:
    def test_edges_attach_by_anchor_frame(self):
        orig_edges = {10: edge(10, 8, 12)}
        disp_edges = {13: edge(13, 11, 15)}
        pairs, _, _ = pair_transitions([10], [13], "rising",
                                       orig_edges=orig_edges, disp_edges=disp_edges)
        assert pairs[0].orig_edge is orig_edges[10]
        assert pairs[0].disp_edge is disp_edges[13]

    def test_missing_edge_for_one_anchor_is_left_none(self):
        pairs, _, _ = pair_transitions([10], [13], "rising",
                                       orig_edges={10: edge(10, 8, 12)}, disp_edges={})
        assert pairs[0].orig_edge is not None
        assert pairs[0].disp_edge is None

    def test_matching_is_unaffected_by_edges(self):
        """Regression guard: edges are payload, never an input to the greedy
        sweep. Same frames in, same pairing out, with or without them."""
        args = ([10, 28], [30], "rising")
        bare, ubo, ubd = pair_transitions(*args, max_frames=5)
        withe, uwo, uwd = pair_transitions(
            *args, max_frames=5,
            orig_edges={10: edge(10, 9, 11), 28: edge(28, 27, 29)},
            disp_edges={30: edge(30, 29, 31)},
        )
        assert frames(bare) == frames(withe) == [(28, 30)]
        assert (ubo, ubd) == (uwo, uwd) == ([10], [])
