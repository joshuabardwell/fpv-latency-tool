"""
Transition edge characterization (core/edges.py).

Signals are built inline in the same idiom as test_detection.py: dark 20.0,
bright 220.0. Anchors are passed in explicitly rather than run through
find_rising/find_falling, so a failure here can only be edges.py's fault —
the two modules are tested independently on purpose.
"""

import numpy as np

from core.detection import find_falling, find_rising
from core.edges import (
    SNR_MIN,
    W_AMBIGUOUS_EDGE,
    W_LOW_SNR,
    W_SLOW_RAMP,
    W_UNSTEADY_LEVEL,
    characterize_signal,
    estimate_noise,
)

DARK, BRIGHT = 20.0, 220.0


def square_wave():
    """Dark 0-9, bright 10-24, dark 25-39 — the same shape test_detection.py
    uses. Instantaneous transitions: rising at 10, falling at 25."""
    data = np.full(40, DARK, dtype=np.float64)
    data[10:25] = BRIGHT
    return data


def ramped_wave():
    """Dark 0-9, a 4-frame ramp over 10-13, bright from 14. Falling mirrors it
    at 25-28, bright through 24, dark from 29."""
    data = np.full(40, DARK, dtype=np.float64)
    data[10:14] = [70.0, 120.0, 170.0, 210.0]
    data[14:25] = BRIGHT
    data[25:29] = [170.0, 120.0, 70.0, 30.0]
    data[29:] = DARK
    return data


def edges_of(data, sigma_k=3.0):
    """Characterize using the real detector's anchors, the way the app does."""
    rising = find_rising(data, delta=25)
    falling = find_falling(data, delta=25)
    return characterize_signal(data, rising, falling, sigma_k)


class TestInstantaneousTransitions:
    def test_first_and_full_collapse_onto_the_anchor(self):
        """The compatibility guarantee: with nothing to ramp through, all three
        metrics land on the same frame, so a square-wave clip measures exactly
        as it did before this module existed."""
        edges, _ = edges_of(square_wave())
        rise = edges[10]
        assert (rise.first_frame, rise.full_frame, rise.anchor_frame) == (10, 10, 10)
        fall = edges[25]
        assert (fall.first_frame, fall.full_frame, fall.anchor_frame) == (25, 25, 25)

    def test_levels_are_measured_not_assumed(self):
        edges, _ = edges_of(square_wave())
        assert edges[10].baseline == DARK and edges[10].plateau == BRIGHT
        # Falling reverses which level is which.
        assert edges[25].baseline == BRIGHT and edges[25].plateau == DARK

    def test_clean_signal_raises_no_warnings(self):
        """The false-positive guard, and the most important test here: a
        warning that fires on good footage trains the user to ignore all of
        them."""
        edges, quality = edges_of(square_wave())
        assert all(e.warnings == () for e in edges.values())
        assert quality.is_clean

    def test_zero_noise_does_not_divide_by_zero(self):
        # sigma is exactly 0 on synthetic data; MIN_BAND_ABS has to carry it.
        edges, _ = edges_of(square_wave())
        assert edges[10].snr == float("inf")
        assert not edges[10].warnings


class TestRampedTransitions:
    def test_known_ramp_boundaries(self):
        # Ramp occupies 10-13; frame 14 is the first at the settled level.
        edges, _ = edges_of(ramped_wave())
        assert edges[10].first_frame == 10
        assert edges[10].full_frame == 14

    def test_anchor_sits_inside_the_ramp(self):
        """The case that motivated this work. The steepest step (20->70 at
        frame 10) is not the midpoint, so the old single number was neither
        first-pixel nor full-frame."""
        edges, _ = edges_of(ramped_wave())
        e = edges[10]
        assert e.first_frame <= e.anchor_frame <= e.full_frame
        assert e.ramp_frames == 4

    def test_falling_ramp_mirrors_rising(self):
        edges, _ = edges_of(ramped_wave())
        fall = next(e for e in edges.values() if e.polarity == "falling")
        assert fall.first_frame == 25
        assert fall.full_frame == 29

    def test_average_lands_between_the_two(self):
        edges, _ = edges_of(ramped_wave())
        e = edges[10]
        midpoint = (e.first_frame + e.full_frame) / 2
        assert midpoint == 12.0


class TestSearchWindows:
    def test_plateau_search_stops_at_the_next_transition(self):
        """A rising edge's window must not run into the following falling edge,
        or its 'plateau' would average the bright level with the next dark one."""
        edges, _ = edges_of(square_wave())
        assert edges[10].plateau == BRIGHT  # not pulled down toward DARK

    def test_transition_too_close_to_the_start_is_dropped(self):
        # Rising at frame 1 leaves no room for a baseline run before it.
        data = np.full(30, DARK, dtype=np.float64)
        data[1:] = BRIGHT
        edges, _ = characterize_signal(data, [1], [], 3.0)
        assert edges == {}

    def test_no_anchors_yields_nothing(self):
        edges, quality = characterize_signal(np.full(20, DARK), [], [], 3.0)
        assert edges == {} and quality.is_clean


class TestNoise:
    def test_estimate_noise_ignores_the_transitions(self):
        """Noise must be measured from the flat stretches only — the 200-level
        step between them is signal, not scatter."""
        rng = np.random.default_rng(0)
        data = square_wave() + rng.normal(0.0, 2.0, 40)
        sigma = estimate_noise(data, [10, 25])
        assert 1.0 < sigma < 4.0  # recovers ~2.0, nowhere near the 200 step

    def test_noise_below_the_band_does_not_move_first_light(self):
        rng = np.random.default_rng(1)
        data = square_wave() + rng.normal(0.0, 2.0, 40)
        edges, _ = edges_of(data)
        assert edges[10].first_frame == 10

    def test_raising_sigma_k_narrows_the_measured_ramp(self):
        """Higher k = wider dead band = first-light later, fully-lit earlier.
        Monotonic, so the knob behaves predictably when tuning against footage."""
        rng = np.random.default_rng(2)
        data = ramped_wave() + rng.normal(0.0, 3.0, 40)
        loose, _ = characterize_signal(data, find_rising(data, 25), find_falling(data, 25), 1.0)
        tight, _ = characterize_signal(data, find_rising(data, 25), find_falling(data, 25), 8.0)
        assert tight[10].first_frame >= loose[10].first_frame
        assert tight[10].full_frame <= loose[10].full_frame


class TestAsymmetricNoise:
    def test_noisy_plateau_after_a_quiet_baseline_does_not_inflate_the_ramp(self):
        """Regression: the pre-transition baseline is exactly flat (a dark
        sensor floor) while the post-transition plateau carries real scatter
        (e.g. PWM flicker on a lit LED). Pooling their residuals into one MAD
        let the quiet baseline's zero residuals dilute the plateau's real
        noise to near-zero (sigma ~0.005 instead of ~12 on this data) — on
        real 240fps footage this measured a 4-frame ramp on a source that
        reaches full brightness in 1-2 frames, purely because the resulting
        too-tight band took several frames for noise to cross by chance.
        Local sigma must be the max of pre's and post's own MAD, taken
        separately, not a single MAD over both pooled together."""
        rng = np.random.default_rng(22)
        data = np.full(60, DARK, dtype=np.float64)
        data[30:] = BRIGHT
        data[30:] += rng.normal(0.0, 12.0, 30)
        edges, _ = characterize_signal(data, [30], [], 3.0)
        assert edges[30].ramp_frames <= 1


class TestPerTransitionWarnings:
    def test_diluted_contrast_flags_low_snr(self):
        """Display fills only part of an oversized ROI: the step survives but
        the amplitude collapses toward the noise.

        Anchors are given explicitly rather than detected. At this contrast a
        low enough delta to find the real step also fires on the noise, and
        that is core.detection's problem to have, not this module's."""
        rng = np.random.default_rng(3)
        data = np.full(40, 100.0)
        data[10:25] = 112.0  # 12-level step against sigma ~4
        data += rng.normal(0.0, 4.0, 40)
        edges, _ = characterize_signal(data, [10], [25], 3.0)
        rise = edges[10]
        assert rise.snr < SNR_MIN
        assert W_LOW_SNR in rise.warnings

    def test_motion_during_the_ramp_flags_ambiguous_edge(self):
        """The signal dips back below the first-light band mid-transition, so
        'the frame where light first appears' has more than one answer."""
        data = np.full(40, DARK, dtype=np.float64)
        data[10] = 120.0
        data[11] = DARK      # dropped back out — camera moved
        data[12:25] = BRIGHT
        data[25:] = DARK
        edges, _ = characterize_signal(data, [10], [25], 3.0)
        assert edges[10].crossings > 1
        assert W_AMBIGUOUS_EDGE in edges[10].warnings

    def test_drift_located_as_a_transition_is_not_reported_clean(self):
        """A long monotonic climb is drift, not a transition. It must never
        come back as a clean edge, whichever check happens to catch it — here
        either `unsteady-level` (the flat regions turn out not to be flat) or
        `low-snr` (the same climb, read as scatter about a single median,
        makes the effective noise huge relative to the amplitude — which is
        exactly what fires once local sigma is the max of pre's and post's own
        MAD rather than a single MAD pooled across both: a side that's mostly
        climb, not flat, gets an honestly large MAD instead of one diluted by
        the other side).

        With the climb filling the window, the plateau median is taken over
        the climb itself, so the signal looks like it settles early and the
        ramp measures deceptively short. The flatness of the level regions is
        what actually gives it away."""
        data = np.concatenate([
            np.full(6, DARK),
            np.linspace(DARK, BRIGHT, 28),
            np.full(6, BRIGHT),
        ])
        anchors = find_rising(data, delta=5)
        edges, _ = characterize_signal(data, anchors, [], 3.0)
        assert edges, "expected the ramp to be located at all"
        assert all(not e.is_clean for e in edges.values())
        assert all(
            W_UNSTEADY_LEVEL in e.warnings or W_LOW_SNR in e.warnings
            for e in edges.values()
        )

    def test_identical_ramps_get_the_same_verdict_either_side(self):
        """Regression: slow-ramp was measured against HALF the gap to the
        neighbouring transitions, so the same 3-frame ramp came out flagged
        when its neighbours were close and clean when they were far. A
        transition's verdict must depend on the transition, not on how much
        empty space happens to surround it."""
        data = np.full(30, DARK, dtype=np.float64)
        data[5:8] = [70.0, 120.0, 170.0]
        data[8:15] = BRIGHT
        data[15:18] = [170.0, 120.0, 70.0]
        data[18:] = DARK
        edges, _ = characterize_signal(data, [5], [15], 3.0)
        # Both are 3-frame ramps; neither is drift.
        assert edges[5].ramp_frames == edges[15].ramp_frames == 3
        assert W_SLOW_RAMP not in edges[5].warnings
        assert W_SLOW_RAMP not in edges[15].warnings

    def test_slow_ramp_fires_when_the_edge_never_settles(self):
        """A transition whose climb outruns its own window: no frame in range
        ever reaches the settled level."""
        data = np.concatenate([
            np.full(5, DARK),
            np.linspace(DARK, BRIGHT, 25),
            np.full(10, BRIGHT),
        ])
        edges, _ = characterize_signal(data, [6], [], 3.0)
        assert W_SLOW_RAMP in edges[6].warnings


class TestSignalQuality:
    def _multi_cycle(self, n_cycles=5, baseline_step=0.0, odd_amplitude=None):
        """n_cycles of a 20-frame square wave. baseline_step shifts the whole
        signal a little further up on each cycle (drift); odd_amplitude gives
        one cycle a different height."""
        out = []
        for c in range(n_cycles):
            base = DARK + c * baseline_step
            high = base + (odd_amplitude if (odd_amplitude and c == 2) else 200.0)
            out.append(np.full(10, base))
            out.append(np.full(10, high))
        return np.concatenate(out).astype(np.float64)

    def test_stable_signal_is_clean(self):
        data = self._multi_cycle()
        _, quality = edges_of(data)
        assert not quality.unstable_baseline
        assert not quality.inconsistent_amplitude

    def test_marching_baseline_is_flagged(self):
        """The motivating scenario: ROI composition changes across the clip, so
        each cycle's dark level sits higher than the last."""
        data = self._multi_cycle(baseline_step=30.0)
        _, quality = edges_of(data)
        assert quality.unstable_baseline
        assert quality.baseline_spread > 0

    def test_one_odd_cycle_flags_inconsistent_amplitude(self):
        data = self._multi_cycle(odd_amplitude=60.0)
        _, quality = edges_of(data)
        assert quality.inconsistent_amplitude

    def test_baselines_are_compared_within_polarity_only(self):
        """Regression: pooling rising and falling baselines together would
        compare the dark level against the bright level and report every clean
        square wave as unstable."""
        _, quality = edges_of(square_wave())
        assert not quality.unstable_baseline


class TestDriftImmunity:
    def test_slow_drift_does_not_move_the_measured_frames(self):
        """The headline guarantee. A background level rising steadily across the
        whole clip is absorbed by per-transition local baselines, so the
        reported frames are identical to the undrifted signal."""
        clean = ramped_wave()
        drifted = clean + np.linspace(0.0, 25.0, len(clean))

        # Fixed anchors on purpose. Drift shifts which single step is steepest,
        # so letting find_rising choose would be measuring core.detection's
        # sensitivity to drift, not this module's immunity to it.
        anchors_rise, anchors_fall = [10], [25]
        clean_edges, _ = characterize_signal(clean, anchors_rise, anchors_fall, 3.0)
        drift_edges, _ = characterize_signal(drifted, anchors_rise, anchors_fall, 3.0)

        assert set(clean_edges) == set(drift_edges) == {10, 25}
        for anchor, edge in clean_edges.items():
            # First-light is exact: it is found by walking back from the anchor
            # to where the signal leaves its own local baseline, and drift moves
            # that baseline along with it.
            assert drift_edges[anchor].first_frame == edge.first_frame
            # Fully-lit is within a frame rather than exact. Drift registers as
            # a little extra noise, which widens the band, which can let the
            # settled level be reached one frame sooner. That is the
            # conservative direction (a wider band never reports a transition
            # as faster than it was) and one frame at 240fps is ~4ms.
            assert abs(drift_edges[anchor].full_frame - edge.full_frame) <= 1

    def test_drift_shifts_the_measured_levels_but_not_the_frames(self):
        """Corollary of the above: the baselines *do* move with the drift —
        they are measured, not assumed. That is exactly why the frames don't."""
        clean = ramped_wave()
        drifted = clean + np.linspace(0.0, 25.0, len(clean))
        clean_edges, _ = characterize_signal(clean, [10], [25], 3.0)
        drift_edges, _ = characterize_signal(drifted, [10], [25], 3.0)
        assert drift_edges[10].baseline > clean_edges[10].baseline


class TestDriftAwareBand:
    """Regression tests from real 240fps footage, where a display's dark level
    crept ~0.35 levels/frame for 20+ frames before each flash — an ROI slightly
    larger than the screen inside it. Sized only for frame-to-frame scatter,
    the band let the backward walk sail through that creep and report a 0.00 ms
    latency, which is physically impossible."""

    def _creeping_signal(self):
        """Baseline creeping upward, then a genuine hard transition at frame 40.
        Shaped after the real clip: the creep is ~0.35/frame, the transition
        jumps ~90 levels in one frame."""
        data = np.empty(70, dtype=np.float64)
        data[:40] = 98.0 + np.arange(40) * 0.35   # 98 -> 111.6
        data[40] = 147.0
        data[41] = 173.0
        data[42] = 192.0
        data[43:] = 210.0
        return data

    def test_creeping_baseline_does_not_pull_first_light_early(self):
        data = self._creeping_signal()
        edges, _ = characterize_signal(data, [40], [], 3.0)
        # The transition is at 40. Anything materially earlier means the walk
        # back has traversed the creep instead of stopping at the real edge.
        assert edges[40].first_frame == 40

    def test_mild_creep_is_compensated_silently(self):
        """A creep small relative to amplitude (here ~8 levels of drift across
        the measurement window against a 103-level step) is handled by the
        adaptive band and does NOT raise a per-transition warning. The
        measurement is correct, so flagging it would be noise — and a warning
        that fires on every mild creep is one the user learns to ignore.

        Drift at this scale is still caught where it belongs: as a per-ROI
        verdict (unstable baseline / inconsistent contrast) rather than a
        per-transition one."""
        data = self._creeping_signal()
        edges, _ = characterize_signal(data, [40], [], 3.0)
        assert edges[40].warnings == ()

    def test_creep_large_against_amplitude_is_flagged(self):
        """The other side of the line: when the level moves by a serious
        fraction of the step itself, the measurement really is suspect."""
        data = np.empty(70, dtype=np.float64)
        data[:40] = 20.0 + np.arange(40) * 2.5  # 20 -> 117.5, steep
        data[40:] = 220.0
        edges, _ = characterize_signal(data, [40], [], 3.0)
        assert W_UNSTEADY_LEVEL in edges[40].warnings

    def test_clean_signal_keeps_a_tight_band(self):
        """The drift term is self-calibrating: with no tilt it contributes
        nothing, so precision on good footage is not sacrificed to robustness
        on bad. A 4-frame ramp still measures as 4 frames."""
        edges, _ = edges_of(ramped_wave())
        assert edges[10].first_frame == 10
        assert edges[10].full_frame == 14

    def test_drift_term_applies_to_first_light_only(self):
        """Fully-lit is found by scanning FORWARD and stops at the first
        qualifying frame, so it never traverses drift — widening its band would
        buy no robustness and cost real precision. Applying the term to both
        sides made this 4-frame ramp measure as 3, because the plateau window
        starts close enough to the ramp that its 'tilt' is the ramp's own tail.
        """
        data = ramped_wave()
        edges, _ = characterize_signal(data, [10], [25], 3.0)
        assert edges[10].ramp_frames == 4
