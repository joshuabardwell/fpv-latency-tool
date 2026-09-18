"""
Manual transition editing: the user's placement overrides auto-detection.

Pure Python, no NumPy and no Qt — usable from the GUI and from tests alike.

`core.detection` answers *which* transitions exist and `core.edges` answers *how
far each extends*. Both are heuristics, and on noisy footage neither can be made
reliable by tuning alone: parameters that read a clean clip correctly produce
false edges on a dirty one. Rather than growing more knobs, the tool makes a
first pass and the user corrects it by eye. This module is what "corrected by
eye" means to the rest of the code.

A user decision is authoritative. It outranks any automatic value and survives
every parameter change, because an edit is bound to a TRANSITION rather than to
a position in a list (see `rebind`).

Two kinds of edit exist, and they are applied at deliberately different points
in the pipeline:

    find_rising/find_falling -> apply_min_spacing
      -> rebind -> delete_anchors        <- deletions, BEFORE characterization
      -> characterize_signal
      -> apply_overrides                 <- moved frames, AFTER characterization
      -> pair_transitions

Deletions have to come first because `characterize_signal` bounds each
transition's search window at the midpoints to its NEIGHBOURING anchors. A noise
blip located as a transition truncates the window of the real transition beside
it, so deleting the blip has to happen early enough for the survivor to be
re-measured against the wider window it should have had all along.

Overrides come last because they must not disturb anything upstream of them.
`anchor_frame` is never touched, and pairing keys on anchors, so moving a marker
changes the reported milliseconds and nothing else — it can never re-pair a
transition behind the user's back. Deleting one removes an anchor and therefore
does re-pair, which is the entire point of offering it.
"""

from dataclasses import dataclass, replace

from core.edges import (
    W_AMBIGUOUS_EDGE,
    W_SLOW_RAMP,
    TransitionEdge,
)

# How far an anchor may move between detection runs and still be recognised as
# the same physical transition. Anchors do not move at all when Edge Sensitivity
# or Max Latency change (neither reaches core.detection); a Min DeltaBrightness
# or Min Spacing change shifts them by a frame or two, since the steepest step of
# a real transition stays where it is and only the run collapsed around it grows.
# Wide enough to track that, narrow enough that an edit can never migrate onto a
# genuinely different transition.
REBIND_WINDOW_FRAMES = 5

# Warnings dropped from an edge the user has placed. Both describe how confident
# the AUTOMATIC reading was about WHERE the edge lay — `slow-ramp` that the ramp
# filled too much of the window, `ambiguous-edge` that the signal crossed the
# band more than once — and that reading no longer exists once a human has said
# where the edge is. `low-snr` and `unsteady-level` describe the signal itself
# and are just as true afterwards, so they stay.
POSITION_WARNINGS = (W_SLOW_RAMP, W_AMBIGUOUS_EDGE)


@dataclass(frozen=True)
class EditTarget:
    """One END of one transition — what the UI's selection points at, and the
    finest thing the user can move.

    Deliberately addresses a transition by `anchor_frame` rather than by the
    frame its marker currently sits on: nudging changes where the marker is
    drawn, and a selection that lost track of its transition the moment it moved
    would be unusable.
    """

    roi: str          # "original" | "display"
    polarity: str     # "rising" | "falling"
    anchor_frame: int
    which: str        # "first" | "full"

    @property
    def other_end(self) -> "EditTarget":
        return replace(self, which="full" if self.which == "first" else "first")

    @property
    def transition(self) -> tuple[str, str, int]:
        """Identity of the transition, ignoring which end is selected."""
        return (self.roi, self.polarity, self.anchor_frame)


@dataclass(frozen=True)
class ManualEdit:
    """One user decision about one transition.

    `anchor_frame` is in absolute video frames and is the edit's identity: it is
    re-resolved against each detection run by `rebind`, not stored once and
    trusted forever.

    `first_frame`/`full_frame` are None until the user moves something, at which
    point BOTH are pinned — see `set_frame`.
    """

    roi: str                          # "original" | "display"
    polarity: str                     # "rising" | "falling"
    anchor_frame: int
    first_frame: int | None = None
    full_frame: int | None = None
    deleted: bool = False

    @property
    def is_noop(self) -> bool:
        """An edit that no longer says anything, and can be dropped."""
        return self.first_frame is None and self.full_frame is None and not self.deleted

    @property
    def moves_frames(self) -> bool:
        return self.first_frame is not None or self.full_frame is not None


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def rebind(
    edits: list[ManualEdit],
    rise_anchors: list[int],
    fall_anchors: list[int],
    roi: str,
    window: int = REBIND_WINDOW_FRAMES,
) -> list[ManualEdit]:
    """Re-resolve one ROI's edits against a fresh set of anchors.

    Each edit takes the nearest same-polarity anchor within `window`, one-to-one;
    nearest wins and ties break to the lower anchor frame, so the result does not
    depend on the order the user happened to make the edits in. The edit's
    recorded `anchor_frame` is updated to whatever it bound to, so a series of
    small parameter changes tracks the transition instead of accumulating
    distance from it until it falls out of the window.

    An edit that finds no anchor is left **dormant, not discarded**: its
    transition is not currently being detected, but raising Min Delta and
    lowering it again must bring the user's decision back rather than silently
    losing it. Edits belonging to the other ROI pass through untouched.
    """
    result = list(edits)
    for polarity, anchors in (("rising", rise_anchors), ("falling", fall_anchors)):
        indices = [
            i for i, e in enumerate(edits) if e.roi == roi and e.polarity == polarity
        ]
        if not indices:
            continue
        # (distance, anchor, edit index) sorts into exactly the priority above.
        candidates = sorted(
            (abs(a - edits[i].anchor_frame), a, i)
            for i in indices
            for a in anchors
            if abs(a - edits[i].anchor_frame) <= window
        )
        claimed_anchors: set[int] = set()
        claimed_edits: set[int] = set()
        for _, anchor, i in candidates:
            if anchor in claimed_anchors or i in claimed_edits:
                continue
            claimed_anchors.add(anchor)
            claimed_edits.add(i)
            if anchor != edits[i].anchor_frame:
                result[i] = replace(edits[i], anchor_frame=anchor)
    return result


def delete_anchors(
    edits: list[ManualEdit],
    rise_anchors: list[int],
    fall_anchors: list[int],
    roi: str,
) -> tuple[list[int], list[int]]:
    """Drop the anchors the user has deleted, for one ROI.

    Call this AFTER `rebind` and BEFORE `characterize_signal` — see the module
    docstring for why the order is not negotiable.
    """
    dead = {(e.polarity, e.anchor_frame) for e in edits if e.roi == roi and e.deleted}
    return (
        [a for a in rise_anchors if ("rising", a) not in dead],
        [a for a in fall_anchors if ("falling", a) not in dead],
    )


def resolve(
    edge: TransitionEdge, edit: ManualEdit | None, lo: int, hi: int
) -> tuple[int, int]:
    """The (first_frame, full_frame) an edit produces for an edge: the pinned
    values where it has them, the automatic ones elsewhere, clamped to the
    analysis range [lo, hi] and ordered."""
    first = edge.first_frame if edit is None or edit.first_frame is None else edit.first_frame
    full = edge.full_frame if edit is None or edit.full_frame is None else edit.full_frame
    first = _clamp(first, lo, hi)
    full = _clamp(full, lo, hi)
    # `set_frame` already guarantees the order for anything the user pinned. This
    # only catches an automatic value that a re-characterization moved across a
    # pinned one; the pinned frame is the one that stands.
    if first > full:
        if edit is not None and edit.first_frame is not None:
            full = first
        else:
            first = full
    return first, full


def set_frame(
    edit: ManualEdit,
    edge: TransitionEdge,
    which: str,
    value: int,
    lo: int,
    hi: int,
) -> ManualEdit:
    """Place one end of a transition at `value`. The single mutator behind both
    the nudge keys and Set-to-playhead.

    `which` is "first" or "full". The other end is **pushed** rather than
    blocked when the two would cross: raising first-light past fully-lit carries
    fully-lit along, and lowering fully-lit past first-light carries first-light
    along.

    Pushing is required, not a nicety. `_characterize_one` clamps both frames to
    the anchor, so on an instantaneous transition — the normal shape for an LED
    source — first-light and fully-lit sit on the SAME frame and the graph draws
    one triangle. Blocking at the boundary would make "nudge first-light right"
    and "nudge fully-lit left" dead keys on the most ordinary transition there
    is; pushing makes them move the whole transition, which is what someone
    looking at a single triangle means by nudging it.

    Both ends are pinned once either is touched. Leaving the untouched end
    tracking the algorithm would let it move on its own the next time a
    parameter changed, which is not what "the user's decision is authoritative"
    can be allowed to mean. Reset restores both together.
    """
    first, full = resolve(edge, edit, lo, hi)
    value = _clamp(value, lo, hi)
    if which == "first":
        first = value
        full = max(full, value)
    else:
        full = value
        first = min(first, value)
    return replace(edit, first_frame=first, full_frame=full, deleted=False)


def apply_overrides(
    edits: list[ManualEdit],
    edges: dict[int, TransitionEdge],
    lo: int,
    hi: int,
    roi: str,
) -> dict[int, TransitionEdge]:
    """Replace the measured extent of every edge the user has moved, for one ROI.

    `edges` is keyed by anchor frame in absolute video frames, exactly as
    `BrightnessGraphWidget._characterize` returns it. An edit whose anchor is not
    present is dormant and contributes nothing.
    """
    out = dict(edges)
    for edit in edits:
        if edit.roi != roi or edit.deleted or not edit.moves_frames:
            continue
        edge = out.get(edit.anchor_frame)
        if edge is None:
            continue
        first, full = resolve(edge, edit, lo, hi)
        out[edit.anchor_frame] = replace(
            edge,
            first_frame=first,
            full_frame=full,
            manual=True,
            warnings=tuple(w for w in edge.warnings if w not in POSITION_WARNINGS),
        )
    return out


def find_edit(
    edits: list[ManualEdit], roi: str, polarity: str, anchor_frame: int
) -> ManualEdit | None:
    for edit in edits:
        if (
            edit.roi == roi
            and edit.polarity == polarity
            and edit.anchor_frame == anchor_frame
        ):
            return edit
    return None


def upsert(edits: list[ManualEdit], edit: ManualEdit) -> list[ManualEdit]:
    """Replace the edit for `edit`'s transition, or append it. An edit that has
    become a no-op is dropped rather than stored, so "is this transition edited"
    stays a question about the list's contents."""
    out = [
        e
        for e in edits
        if not (
            e.roi == edit.roi
            and e.polarity == edit.polarity
            and e.anchor_frame == edit.anchor_frame
        )
    ]
    if not edit.is_noop:
        out.append(edit)
    return out
