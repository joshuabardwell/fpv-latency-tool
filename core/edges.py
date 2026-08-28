"""
Transition edge characterization: how far does a located transition extend?

Pure NumPy, no Qt — usable from the GUI and from tests alike.

`core.detection` answers *which* transitions exist, returning the frame of
steepest brightness change. This module answers *how far each one extends*:
the first frame showing any light change, and the first frame at the settled
level. Those two frames are what the three reported latency metrics are built
from, each comparing the same point on the curve at both ends:

    first-pixel latency = display first_frame - source first_frame
    full-frame  latency = display full_frame  - source full_frame
    average     latency = mean of the two

Everything here is measured *locally*, per transition, against its own
baseline, plateau and noise estimate. That is what makes the measurement
immune to baseline drift across a clip — the case where an ROI is drawn
larger than the display and the display slowly moves within it. Each
transition calibrates against the levels immediately either side of itself,
so a level that changes slowly relative to the flash period never matters.

Drift fast enough to matter *within* a single transition is not corrected. It
is flagged instead (see the quality checks below). There is deliberately no
detrending: for slow drift it is redundant given local baselines, and for
fast drift it would hide genuinely corrupt data behind a plausible-looking
number. This is a measurement instrument — declining to trust a reading beats
quietly cleaning it up.
"""

from dataclasses import dataclass

import numpy as np

# --- geometry of the measurement ------------------------------------------
MIN_FLAT_FRAMES = 3    # frames needed either side to estimate a local sigma
RAMP_GUARD_FRAMES = 2  # frames each side of an anchor never counted as "flat"

# How many frames either side of a transition its levels are measured from.
# The gap between transitions can be hundreds of frames — on real 240fps
# footage the stretch before a flash ran to 276 frames, drifting 143 levels
# end to end — and a median over all of that is not the level immediately
# before the transition. Capping the window is what makes "measured locally"
# mean local in TIME rather than merely local to the neighbouring transitions.
# Long enough for a stable median and sigma, short enough that drift within it
# is negligible: 24 frames is 0.1s at 240fps.
LEVEL_WINDOW_FRAMES = 24

# --- band floors ----------------------------------------------------------
# The band is what separates "still at the baseline" from "changing". Two
# floors keep it sane when sigma is tiny or zero, which is exactly the case
# on clean synthetic data and on well-exposed footage.
MIN_BAND_ABS = 0.5        # gray levels (0-255)
MIN_BAND_FRACTION = 0.02  # of this transition's amplitude
DEFAULT_SIGMA_K = 3.0     # band = k * sigma, before the floors above

# The band must also beat the baseline's own wander. Frame-to-frame scatter is
# not the only thing that can move a level: a display drifting inside an
# oversized ROI creeps steadily, and on real footage a display's dark level
# climbed ~0.35 levels/frame for 20+ frames before a flash. Walking back from
# the anchor with a band sized only for scatter sails straight through that
# creep and reports first-light dozens of frames early — a 0 ms latency, which
# is physically impossible. Scaling the band by the measured tilt is
# self-calibrating: on clean footage the tilt is ~0 and the band stays tight,
# so precision on good clips is not sacrificed to robustness on bad ones.
DRIFT_BAND_K = 2.5

# --- quality thresholds ---------------------------------------------------
# First guesses. They must be tuned against real footage: too loose and they
# never fire, too tight and they cry wolf until the warnings get ignored.
SNR_MIN = 6.0            # amplitude / sigma below this is "low-snr"
RAMP_MAX_FRACTION = 0.5  # ramp filling more than this share of the gap between
                         # neighbouring transitions is drift, not a transition
SPREAD_MAX = 0.25        # as a fraction of amplitude: how far a level may swing
                         # within one transition, and how far levels may differ
                         # between transitions, before either is called dirty
TILT_SIGMA_K = 2.0       # a level's tilt must also beat this many sigma before
                         # it counts as drift rather than scatter

W_LOW_SNR = "low-snr"
W_AMBIGUOUS_EDGE = "ambiguous-edge"
W_SLOW_RAMP = "slow-ramp"
W_UNSTEADY_LEVEL = "unsteady-level"


@dataclass(frozen=True)
class TransitionEdge:
    """One transition's measured extent. Frames are in the same index space as
    the `data` array passed to characterize_signal (array-local, not absolute
    video frames) — callers add their own in_point offset, exactly as they
    already do for core.detection's output."""

    anchor_frame: int          # steepest step; what pairing and min-spacing key on
    first_frame: int           # first frame showing any light change
    full_frame: int            # first frame at the settled level
    baseline: float            # measured pre-transition level
    plateau: float             # measured post-transition level
    polarity: str              # "rising" | "falling"
    snr: float                 # amplitude / local sigma; inf when sigma is 0
    crossings: int             # band crossings in the window; 1 is clean
    warnings: tuple[str, ...] = ()

    @property
    def amplitude(self) -> float:
        return abs(self.plateau - self.baseline)

    @property
    def ramp_frames(self) -> int:
        return self.full_frame - self.first_frame

    @property
    def is_clean(self) -> bool:
        return not self.warnings


@dataclass(frozen=True)
class SignalQuality:
    """Whole-signal description for one ROI: on a proper square-wave test
    pattern every cycle shares a baseline and an amplitude, and these say
    whether they actually do.

    These booleans describe signal SHAPE, not correctness, and must not be
    surfaced as errors. A baseline that moves across a clip is entirely normal
    when the device under test has auto-exposure — the display ROI is showing
    that camera's image, so its AE opens up through every dark stretch — and
    the per-transition measurements can still be exactly right, as they were on
    the reference footage. Framing, changing light, a nudged camera and AE all
    produce the identical signature here, so nothing downstream may assert a
    cause. Whether any individual measurement is trustworthy is the
    per-transition warnings' question, not this record's."""

    median_amplitude: float
    baseline_spread: float   # largest departure of any one transition's
                             # baseline from the typical one, within a polarity
    amplitude_spread: float  # same, for amplitude
    unstable_baseline: bool
    inconsistent_amplitude: bool

    @property
    def is_clean(self) -> bool:
        return not (self.unstable_baseline or self.inconsistent_amplitude)


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation, scaled to estimate sigma for Gaussian noise.
    Used instead of std for noise because a single ramp frame leaking into a
    supposedly-flat window would inflate a std but barely move a median."""
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _max_deviation(values: np.ndarray) -> float:
    """Largest absolute departure from the median.

    Deliberately NOT _mad here, even though the two look interchangeable. MAD
    is robust *to outliers*, and for the quality checks an outlier is precisely
    what needs catching: one cycle at half amplitude among four good ones
    leaves MAD at exactly zero, because the majority agree.
    """
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values - np.median(values))))


def _flat_segments(n: int, anchors: list[int], guard: int) -> list[tuple[int, int]]:
    """Half-open index ranges lying between transitions, with a guard band
    around each anchor excluded so ramp frames never count as flat."""
    segments = []
    prev = 0
    for a in sorted(anchors):
        hi = a - guard
        if hi > prev:
            segments.append((prev, hi))
        prev = max(prev, a + guard + 1)
    if n > prev:
        segments.append((prev, n))
    return segments


def estimate_noise(
    data: np.ndarray, anchors: list[int], guard: int = RAMP_GUARD_FRAMES
) -> float:
    """Pooled noise estimate over every flat stretch of the signal. Residuals
    are taken about each segment's own median before pooling, so a level that
    differs between segments (dark vs bright plateaus, or drift) contributes
    nothing — only the scatter within each segment does."""
    residuals = []
    for lo, hi in _flat_segments(len(data), anchors, guard):
        segment = data[lo:hi]
        # Chunk before taking residuals. A flat stretch can run to hundreds of
        # frames, and a level that drifts across it would be counted as noise
        # if residuals were taken about a single median for the whole thing —
        # on real footage that inflated a display's sigma to 17 levels when its
        # actual frame-to-frame scatter was a fraction of that.
        for start in range(0, segment.size, LEVEL_WINDOW_FRAMES):
            chunk = segment[start : start + LEVEL_WINDOW_FRAMES]
            if chunk.size >= MIN_FLAT_FRAMES:
                residuals.append(chunk - np.median(chunk))
    if not residuals:
        return 0.0
    return _mad(np.concatenate(residuals))


def _tilt(segment: np.ndarray) -> float:
    """Systematic drift across a region: how far its last quarter's level sits
    from its first quarter's.

    Medians of quarters rather than peak-to-peak, because ptp grows with noise
    and would call a merely noisy level a drifting one — on real footage a
    perfectly flat but noisy display region had a 45-level ptp against a
    4-level actual tilt."""
    if segment.size < 4:
        return 0.0
    k = max(2, segment.size // 4)
    return abs(float(np.median(segment[-k:]) - np.median(segment[:k])))


def _first_departure(
    data: np.ndarray, anchor: int, lo: int, hi: int, threshold: float, rising: bool
) -> int:
    """First frame that has left the baseline band and stays out of it through
    the anchor.

    Walks *backward* from the anchor rather than forward from `lo`, which
    matters: over a long baseline run a single noise sample crossing k-sigma is
    likely, and a forward scan would latch onto it and report first-light many
    frames early. Walking back from the anchor finds the contiguous departure
    that actually belongs to this transition."""
    beyond = (lambda v: v > threshold) if rising else (lambda v: v < threshold)

    if not beyond(data[anchor]):
        # The steepest step landed before the signal cleared the band — happens
        # on a slow ramp. Scan forward for the real crossing instead.
        i = anchor
        while i + 1 < hi and not beyond(data[i]):
            i += 1
        return i

    i = anchor
    while i > lo and beyond(data[i - 1]):
        i -= 1
    return i


def _first_settled(
    data: np.ndarray, start: int, hi: int, threshold: float, rising: bool
) -> int | None:
    reached = (lambda v: v >= threshold) if rising else (lambda v: v <= threshold)
    for i in range(start, hi):
        if reached(data[i]):
            return i
    return None


def _characterize_one(
    data: np.ndarray,
    anchor: int,
    polarity: str,
    lo: int,
    hi: int,
    sigma_k: float,
    global_sigma: float,
) -> TransitionEdge | None:
    """Measure one transition inside the half-open window [lo, hi).
    Returns None when the window is too tight to measure a level either side —
    better no reading than a fabricated one."""
    rising = polarity == "rising"

    # Only the frames nearest the transition on each side: the tail of the run
    # before it, and the head of the run after it. See LEVEL_WINDOW_FRAMES.
    pre = data[lo : max(lo, anchor - RAMP_GUARD_FRAMES)][-LEVEL_WINDOW_FRAMES:]
    post = data[min(hi, anchor + RAMP_GUARD_FRAMES + 1) : hi][:LEVEL_WINDOW_FRAMES]
    if pre.size == 0 or post.size == 0:
        return None

    baseline = float(np.median(pre))
    plateau = float(np.median(post))
    signed_amplitude = plateau - baseline
    # A transition that doesn't actually go the direction it was detected as
    # means the search window is wrong; don't invent an edge for it.
    if (rising and signed_amplitude <= 0) or (not rising and signed_amplitude >= 0):
        return None
    amplitude = abs(signed_amplitude)

    # Local estimate first, so drift across the clip never inflates this
    # transition's band. But a MAD over the handful of frames either side of
    # one transition has high variance and tends to come out low, and an
    # under-estimated sigma is the dangerous direction: it narrows the band and
    # reports first-light early. Taking the larger of the local and pooled
    # estimates keeps the drift immunity (the pooled figure is scatter about
    # each segment's own median, not about a global level) while refusing to
    # trust a suspiciously quiet local sample.
    if pre.size >= MIN_FLAT_FRAMES and post.size >= MIN_FLAT_FRAMES:
        local_sigma = _mad(
            np.concatenate([pre - np.median(pre), post - np.median(post)])
        )
        sigma = max(local_sigma, global_sigma)
    else:
        sigma = global_sigma

    level_tilt = max(_tilt(pre), _tilt(post))
    band = max(sigma_k * sigma, MIN_BAND_FRACTION * amplitude, MIN_BAND_ABS)

    # The drift term applies to the first-light band only, and deliberately so.
    # First-light is found by walking BACKWARD from the anchor, so it traverses
    # the baseline and will run the whole length of any creep it can't
    # distinguish from signal. Fully-lit is found by scanning FORWARD to the
    # first frame that reaches the settled level; it stops at the first
    # qualifying frame and never traverses drift, so widening its band buys no
    # robustness and costs real precision. Applying it to both sides made a
    # 4-frame synthetic ramp measure as 3, because the plateau window starts
    # close enough to the ramp that its "tilt" is the tail of the ramp itself.
    first_band = max(band, DRIFT_BAND_K * _tilt(pre))

    first_threshold = baseline + first_band if rising else baseline - first_band
    full_threshold = plateau - band if rising else plateau + band

    first_frame = _first_departure(data, anchor, lo, hi, first_threshold, rising)
    settled = _first_settled(data, first_frame, hi, full_threshold, rising)

    warnings: list[str] = []
    if settled is None:
        # Never reached the settled level inside its own window: this is drift
        # being read as a transition, not a transition.
        settled = hi - 1
        warnings.append(W_SLOW_RAMP)

    # The anchor is the steepest step, so it lies on the ramp by construction.
    # Clamp rather than trust arithmetic on pathological data.
    first_frame = min(first_frame, anchor)
    full_frame = max(settled, anchor)

    window = data[lo:hi]
    above = window > first_threshold if rising else window < first_threshold
    crossings = int(np.count_nonzero(np.diff(above.astype(np.int8)) != 0))

    snr = float("inf") if sigma <= 0 else amplitude / sigma

    if snr < SNR_MIN:
        warnings.append(W_LOW_SNR)
    if crossings > 1:
        warnings.append(W_AMBIGUOUS_EDGE)
    # The two levels have to actually BE levels. This is the check that catches
    # a long drift located as a transition, and the ramp-length test below
    # cannot: when the climb fills the window, the plateau median is taken over
    # the climb itself, so the signal appears to settle early and the ramp
    # measures short. Whether the "flat" regions are flat gives it away.
    #
    # The sigma term matters as much as the amplitude one. Without it, a noisy
    # but perfectly level region trips this purely because scatter is a large
    # fraction of a small amplitude — on real footage that flagged every pair
    # of a good clip, which is exactly how a warning gets trained into being
    # ignored.
    if level_tilt > max(SPREAD_MAX * amplitude, TILT_SIGMA_K * sigma):
        warnings.append(W_UNSTEADY_LEVEL)
    # Measured against the WHOLE gap to the neighbouring transitions, not half
    # of it. Half made the verdict depend on how much empty space happened to
    # surround a transition rather than on the transition itself: the same
    # 3-frame ramp came out flagged when its neighbours were close and clean
    # when they were far. A ramp filling most of the gap between transitions is
    # unambiguously pathological; a few frames of LCD settling is not.
    span = max(1.0, float(hi - lo))
    if W_SLOW_RAMP not in warnings and (full_frame - first_frame) > RAMP_MAX_FRACTION * span:
        warnings.append(W_SLOW_RAMP)

    return TransitionEdge(
        anchor_frame=anchor,
        first_frame=first_frame,
        full_frame=full_frame,
        baseline=baseline,
        plateau=plateau,
        polarity=polarity,
        snr=snr,
        crossings=crossings,
        warnings=tuple(warnings),
    )


def _signal_quality(edges: dict[int, TransitionEdge]) -> SignalQuality:
    values = list(edges.values())
    if not values:
        return SignalQuality(0.0, 0.0, 0.0, False, False)

    amplitudes = np.array([e.amplitude for e in values], dtype=np.float64)
    median_amplitude = float(np.median(amplitudes))

    # Baselines are compared *within* a polarity, never across it: on a square
    # wave a rising transition's baseline is the dark level and a falling one's
    # is the bright level, so pooling them would report every clean signal as
    # unstable.
    baseline_spread = 0.0
    for polarity in ("rising", "falling"):
        group = [e.baseline for e in values if e.polarity == polarity]
        if len(group) >= 3:  # a spread over one or two samples is not an estimate
            baseline_spread = max(baseline_spread, _max_deviation(np.array(group)))

    amplitude_spread = _max_deviation(amplitudes) if len(values) >= 3 else 0.0
    limit = SPREAD_MAX * median_amplitude

    return SignalQuality(
        median_amplitude=median_amplitude,
        baseline_spread=baseline_spread,
        amplitude_spread=amplitude_spread,
        unstable_baseline=median_amplitude > 0 and baseline_spread > limit,
        inconsistent_amplitude=median_amplitude > 0 and amplitude_spread > limit,
    )


def characterize_signal(
    data: np.ndarray,
    rising_anchors: list[int],
    falling_anchors: list[int],
    sigma_k: float = DEFAULT_SIGMA_K,
) -> tuple[dict[int, TransitionEdge], SignalQuality]:
    """Measure every located transition on one signal.

    Both polarities are taken together because each transition's search window
    is bounded by its neighbours *of either polarity*: on a square wave the
    plateau after a rising edge is terminated by the following falling edge, so
    characterizing rising transitions alone would let their plateau search run
    into the next cycle.

    Returns (edges keyed by anchor frame, whole-signal quality verdict).
    Anchors that could not be measured are simply absent from the dict.
    """
    if data is None or len(data) == 0:
        return {}, SignalQuality(0.0, 0.0, 0.0, False, False)

    polarity_of = {a: "rising" for a in rising_anchors}
    polarity_of.update({a: "falling" for a in falling_anchors})
    anchors = sorted(polarity_of)
    if not anchors:
        return {}, SignalQuality(0.0, 0.0, 0.0, False, False)

    global_sigma = estimate_noise(data, anchors)
    n = len(data)
    edges: dict[int, TransitionEdge] = {}

    for idx, anchor in enumerate(anchors):
        prev_anchor = anchors[idx - 1] if idx > 0 else None
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else None
        lo = 0 if prev_anchor is None else (prev_anchor + anchor) // 2 + 1
        hi = n if next_anchor is None else (anchor + next_anchor) // 2 + 1
        edge = _characterize_one(
            data, anchor, polarity_of[anchor], lo, hi, sigma_k, global_sigma
        )
        if edge is not None:
            edges[anchor] = edge

    return edges, _signal_quality(edges)
