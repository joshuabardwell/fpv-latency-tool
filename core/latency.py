from dataclasses import dataclass

from core.edges import TransitionEdge


@dataclass
class LatencyPair:
    orig_frame: int
    disp_frame: int
    polarity: str  # "rising" | "falling"
    # Measured extent of each end's transition, when it could be characterized.
    # Optional with defaults so positional construction — LatencyPair(10, 13,
    # "rising") — keeps working everywhere it already appears.
    orig_edge: TransitionEdge | None = None
    disp_edge: TransitionEdge | None = None

    def delta_frames(self) -> int:
        """Anchor to anchor: the steepest-step difference, which is what
        pairing keys on. Deliberately unchanged and still an int — the three
        reported metrics below are what the UI shows."""
        return self.disp_frame - self.orig_frame

    def delta_ms(self, fps: float) -> float:
        return self.delta_frames() / fps * 1000.0

    # --- the three reported metrics ---------------------------------------
    # Each compares the SAME point on the transition curve at both ends, so a
    # slow rise on the source cannot inflate the result. Where either end could
    # not be characterized, all three fall back to the anchor delta rather than
    # reporting nothing: a transition at the very edge of the analysis range
    # still deserves a usable row.

    def _edge_delta(self, attr: str) -> float | None:
        if self.orig_edge is None or self.disp_edge is None:
            return None
        return float(getattr(self.disp_edge, attr) - getattr(self.orig_edge, attr))

    def first_delta_frames(self) -> float:
        """Display first-light minus source first-light."""
        delta = self._edge_delta("first_frame")
        return float(self.delta_frames()) if delta is None else delta

    def full_delta_frames(self) -> float:
        """Display fully-lit minus source fully-lit."""
        delta = self._edge_delta("full_frame")
        return float(self.delta_frames()) if delta is None else delta

    def avg_delta_frames(self) -> float:
        """Mean of the two. Commonly a half-frame value, which is a genuine
        gain in resolution rather than a rounding artifact."""
        return (self.first_delta_frames() + self.full_delta_frames()) / 2.0

    def first_delta_ms(self, fps: float) -> float:
        return self.first_delta_frames() / fps * 1000.0

    def full_delta_ms(self, fps: float) -> float:
        return self.full_delta_frames() / fps * 1000.0

    def avg_delta_ms(self, fps: float) -> float:
        return self.avg_delta_frames() / fps * 1000.0

    def orig_first_frame(self) -> int:
        """The frame the source first shows any light — the frame the UI
        reports and seeks to. Falls back to the anchor when this end could not
        be characterized, matching the delta accessors above.

        The anchor (`orig_frame`) is the steepest single-frame step and is an
        internal matching detail: it is not drawn on the graph and is not one
        of the three reported metrics."""
        return self.orig_edge.first_frame if self.orig_edge else self.orig_frame

    def disp_first_frame(self) -> int:
        """Display counterpart of orig_first_frame."""
        return self.disp_edge.first_frame if self.disp_edge else self.disp_frame

    # --- quality ----------------------------------------------------------

    def quality_warnings(self) -> tuple[str, ...]:
        """Both ends' quality flags, deduplicated, source end first. A pair is
        only as trustworthy as its worse end, so these are unioned rather than
        reported separately."""
        seen: list[str] = []
        for edge in (self.orig_edge, self.disp_edge):
            if edge is None:
                continue
            for warning in edge.warnings:
                if warning not in seen:
                    seen.append(warning)
        return tuple(seen)

    def is_clean(self) -> bool:
        return not self.quality_warnings()


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
    orig_edges: dict[int, TransitionEdge] | None = None,
    disp_edges: dict[int, TransitionEdge] | None = None,
) -> tuple[list[LatencyPair], list[int], list[int]]:
    """Greedy nearest-following match. Each disp frame used at most once.
    A same-frame transition (df == of) pairs as zero latency. Frame lists are
    assumed duplicate-free (detection emits strictly increasing frames).
    max_frames: if set, only pair disp transitions within this many frames of orig.
    orig_edges/disp_edges: characterized extents keyed by anchor frame, in the
    same frame space as the lists above; attached to each pair when supplied.
    Matching itself is unaffected by them.
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
            df = sorted_disp[j]
            pairs.append(LatencyPair(
                of, df, polarity,
                orig_edge=None if orig_edges is None else orig_edges.get(of),
                disp_edge=None if disp_edges is None else disp_edges.get(df),
            ))
            j += 1
        else:
            unmatched_orig.append(of)

    paired_disp = {p.disp_frame for p in pairs}
    unmatched_disp = [df for df in disp_frames if df not in paired_disp]
    return pairs, unmatched_orig, unmatched_disp
