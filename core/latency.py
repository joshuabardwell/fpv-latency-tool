from dataclasses import dataclass


@dataclass
class LatencyPair:
    orig_frame: int
    disp_frame: int
    polarity: str  # "rising" | "falling"

    def delta_frames(self) -> int:
        return self.disp_frame - self.orig_frame

    def delta_ms(self, fps: float) -> float:
        return self.delta_frames() / fps * 1000.0


DEFAULT_MAX_LATENCY_FRACTION = 0.5  # fraction of Original Period used as the Max Latency default


def default_max_latency_frames(orig_period_frames: float | None) -> int:
    """Suggested Max Latency cap, in frames: half the mean Original Period.
    Bounds pairing to roughly one half-cycle so a source transition can't be
    mismatched to the wrong cycle's display transition. Returns 0 (unlimited)
    when the period is unknown (fewer than 2 original transitions detected)."""
    if orig_period_frames is None:
        return 0
    return round(DEFAULT_MAX_LATENCY_FRACTION * orig_period_frames)


def pair_transitions(
    orig_frames: list[int],
    disp_frames: list[int],
    polarity: str,
    max_frames: int | None = None,
) -> tuple[list[LatencyPair], list[int], list[int]]:
    """Greedy nearest-following match. Each disp frame used at most once.
    A same-frame transition (df == of) pairs as zero latency. Frame lists are
    assumed duplicate-free (detection emits strictly increasing frames).
    max_frames: if set, only pair disp transitions within this many frames of orig.
    Returns (pairs, unmatched_orig_frames, unmatched_disp_frames)."""
    pairs: list[LatencyPair] = []
    unmatched_orig: list[int] = []
    sorted_disp = sorted(disp_frames)

    # Two-pointer sweep: both lists ascend, so the nearest unused disp
    # candidate for each orig frame only ever moves forward.
    j = 0
    for of in sorted(orig_frames):
        while j < len(sorted_disp) and sorted_disp[j] < of:
            j += 1
        if j < len(sorted_disp) and (
            max_frames is None or sorted_disp[j] - of <= max_frames
        ):
            pairs.append(LatencyPair(of, sorted_disp[j], polarity))
            j += 1
        else:
            unmatched_orig.append(of)

    paired_disp = {p.disp_frame for p in pairs}
    unmatched_disp = [df for df in disp_frames if df not in paired_disp]
    return pairs, unmatched_orig, unmatched_disp
