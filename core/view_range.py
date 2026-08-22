"""
Pure frame-range math for the brightness graph's zoom/pan.  No Qt.

A "range" is a `(start, end)` float tuple describing the visible window in
frame-index units.  `clamp_range` is the single place a range is forced to
fit inside a `[lo, hi]` domain and never fall below `min_width` — every other
function here ends by delegating to it, so panning, zooming, and recentering
can never disagree about the limits.
"""

MIN_ZOOM_FRAMES = 10  # floor on visible-window width, in frames


def clamp_range(
    start: float, end: float, lo: float, hi: float, min_width: float,
) -> tuple[float, float]:
    """Fit [start, end] inside [lo, hi], preserving its width where possible.

    Width is shrunk to the full domain if it's wider than [lo, hi], or grown
    to min_width if narrower (both capped by the domain itself, for a
    degenerate case where min_width > hi - lo). Position is then clamped so
    the window never extends past either edge, sliding it inward — pinning
    an edge to the domain boundary — rather than shrinking it further.
    """
    full = hi - lo
    if full <= 0:
        return (lo, lo)

    width = end - start
    width = min(width, full)
    width = max(width, min(min_width, full))

    center = (start + end) / 2.0
    new_start = center - width / 2.0
    new_end = new_start + width

    if new_start < lo:
        new_start, new_end = lo, lo + width
    if new_end > hi:
        new_start, new_end = hi - width, hi

    return (new_start, new_end)


def center_range(
    center: float, width: float, lo: float, hi: float, min_width: float,
) -> tuple[float, float]:
    """Same width, re-centered on `center`, then clamped.

    Near a domain edge, clamping pins the window's edge to the boundary
    instead of shrinking it — so the requested center can't always be
    reached exactly, but no dead space past the domain is ever shown.
    """
    return clamp_range(center - width / 2.0, center + width / 2.0, lo, hi, min_width)


def zoom_range(
    start: float, end: float, factor: float, anchor: float,
    lo: float, hi: float, min_width: float,
) -> tuple[float, float]:
    """Scale [start, end] by `factor` (<1 zooms in, >1 zooms out) around
    `anchor` — the data-space point that stays fixed (e.g. under the cursor).
    """
    width = end - start
    frac = (anchor - start) / width if width > 0 else 0.5
    new_width = width * factor
    new_start = anchor - frac * new_width
    new_end = new_start + new_width
    return clamp_range(new_start, new_end, lo, hi, min_width)


def pan_range(start: float, end: float, delta: float, lo: float, hi: float) -> tuple[float, float]:
    """Shift [start, end] by `delta`. Width is never changed by clamping —
    a pan starts from an already-valid range, so only position needs to slide
    back inside [lo, hi] when it would otherwise overshoot an edge."""
    width = end - start
    return clamp_range(start + delta, end + delta, lo, hi, width)
