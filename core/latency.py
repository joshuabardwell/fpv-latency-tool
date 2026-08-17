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


def pair_transitions(
    orig_frames: list[int],
    disp_frames: list[int],
    polarity: str,
    max_frames: int | None = None,
) -> tuple[list[LatencyPair], list[int], list[int]]:
    """Greedy nearest-following match. Each disp frame used at most once.
    max_frames: if set, only pair disp transitions within this many frames of orig.
    Returns (pairs, unmatched_orig_frames, unmatched_disp_frames)."""
    pairs: list[LatencyPair] = []
    used_disp: set[int] = set()
    unmatched_orig: list[int] = []

    for of in sorted(orig_frames):
        for df in sorted(disp_frames):
            if df > of and df not in used_disp:
                if max_frames is None or (df - of) <= max_frames:
                    pairs.append(LatencyPair(of, df, polarity))
                    used_disp.add(df)
                    break
        else:
            unmatched_orig.append(of)

    unmatched_disp = [df for df in disp_frames if df not in used_disp]
    return pairs, unmatched_orig, unmatched_disp
