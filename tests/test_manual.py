import numpy as np
import pytest

from core.edges import (
    W_AMBIGUOUS_EDGE,
    W_LOW_SNR,
    W_SLOW_RAMP,
    W_UNSTEADY_LEVEL,
    TransitionEdge,
    characterize_signal,
)
from core.manual import (
    REBIND_WINDOW_FRAMES,
    ManualEdit,
    apply_overrides,
    delete_anchors,
    find_edit,
    rebind,
    resolve,
    set_frame,
    upsert,
)


def edge(anchor, first=None, full=None, polarity="rising", warnings=()):
    return TransitionEdge(
        anchor_frame=anchor,
        first_frame=anchor if first is None else first,
        full_frame=anchor if full is None else full,
        baseline=10.0,
        plateau=200.0,
        polarity=polarity,
        snr=50.0,
        crossings=1,
        warnings=warnings,
    )


# ----------------------------------------------------------------- rebinding


def test_rebind_keeps_an_anchor_that_did_not_move():
    edits = [ManualEdit("display", "rising", 100, first_frame=98, full_frame=104)]
    assert rebind(edits, [100, 200], [], "display") == edits


def test_rebind_follows_an_anchor_that_shifted_a_frame():
    edits = [ManualEdit("display", "rising", 100, first_frame=98, full_frame=104)]
    out = rebind(edits, [102, 200], [], "display")
    assert out[0].anchor_frame == 102
    # The user's frames are untouched: only the identity moved.
    assert (out[0].first_frame, out[0].full_frame) == (98, 104)


def test_rebind_tracks_across_successive_moves_rather_than_accumulating():
    """Each rebind rewrites anchor_frame, so drift of a frame at a time never
    adds up to more than the window and never loses the edit."""
    edits = [ManualEdit("display", "rising", 100)]
    for anchor in (104, 108, 112, 116):
        edits = rebind(edits, [anchor], [], "display")
        assert edits[0].anchor_frame == anchor


def test_rebind_leaves_an_edit_dormant_when_no_anchor_is_near():
    edits = [ManualEdit("display", "rising", 100, first_frame=98)]
    out = rebind(edits, [500], [], "display")
    assert out[0].anchor_frame == 100  # kept, not discarded


def test_a_dormant_edit_reapplies_when_its_anchor_returns():
    edits = [ManualEdit("display", "rising", 100, first_frame=98, full_frame=104)]
    dormant = rebind(edits, [], [], "display")
    revived = rebind(dormant, [100], [], "display")
    edges = apply_overrides(revived, {100: edge(100)}, 0, 999, "display")
    assert (edges[100].first_frame, edges[100].full_frame) == (98, 104)


def test_rebind_will_not_reach_past_the_window():
    far = REBIND_WINDOW_FRAMES + 1
    edits = [ManualEdit("display", "rising", 100)]
    assert rebind(edits, [100 + far], [], "display")[0].anchor_frame == 100


def test_rebind_matches_nearest_first_not_list_order():
    """Two edits competing for two anchors must settle by distance, so the
    result cannot depend on which one the user happened to make first."""
    edits = [
        ManualEdit("display", "rising", 104),
        ManualEdit("display", "rising", 100),
    ]
    out = rebind(edits, [101, 105], [], "display")
    assert [e.anchor_frame for e in out] == [105, 101]


def test_rebind_never_gives_one_anchor_to_two_edits():
    edits = [
        ManualEdit("display", "rising", 100),
        ManualEdit("display", "rising", 101),
    ]
    out = rebind(edits, [100], [], "display")
    assert out[0].anchor_frame == 100
    assert out[1].anchor_frame == 101  # unbound, left dormant


def test_rebind_does_not_mix_polarities_or_rois():
    edits = [
        ManualEdit("display", "falling", 100),
        ManualEdit("original", "rising", 100),
    ]
    out = rebind(edits, [102], [], "display")
    assert [e.anchor_frame for e in out] == [100, 100]


# ------------------------------------------------------------------ deletion


def test_delete_anchors_removes_only_the_marked_transition():
    edits = [ManualEdit("display", "rising", 100, deleted=True)]
    rise, fall = delete_anchors(edits, [100, 200], [150], "display")
    assert (rise, fall) == ([200], [150])


def test_delete_anchors_ignores_the_other_roi():
    edits = [ManualEdit("original", "rising", 100, deleted=True)]
    rise, _ = delete_anchors(edits, [100, 200], [], "display")
    assert rise == [100, 200]


def test_deleting_a_blip_widens_its_neighbours_characterization_window():
    """The reason deletions must run before characterize_signal: a spurious
    anchor bounds the real transition's search window at the midpoint to it."""
    data = np.concatenate([
        np.full(40, 10.0),
        np.linspace(10.0, 200.0, 30),   # a real, slow, multi-frame ramp
        np.full(60, 200.0),
    ]).astype(np.float64)
    real_anchor = int(np.argmax(np.diff(data))) + 1
    blip = real_anchor + 20  # sits inside the ramp, truncating its window

    with_blip, _ = characterize_signal(data, [real_anchor, blip], [])
    kept_rise, kept_fall = delete_anchors(
        [ManualEdit("display", "rising", blip, deleted=True)],
        [real_anchor, blip], [], "display",
    )
    without_blip, _ = characterize_signal(data, kept_rise, kept_fall)

    assert blip in with_blip
    assert blip not in without_blip
    # The survivor now sees the whole ramp instead of being cut off mid-climb.
    assert without_blip[real_anchor].full_frame > with_blip[real_anchor].full_frame


# ------------------------------------------------------------ moving a frame


def test_set_frame_pins_both_ends():
    e = edge(100, first=98, full=104)
    out = set_frame(ManualEdit("display", "rising", 100), e, "first", 97, 0, 999)
    assert (out.first_frame, out.full_frame) == (97, 104)


def test_nudging_first_light_right_past_fully_lit_pushes_it():
    e = edge(100, first=98, full=104)
    out = set_frame(ManualEdit("display", "rising", 100), e, "first", 106, 0, 999)
    assert (out.first_frame, out.full_frame) == (106, 106)


def test_nudging_fully_lit_left_past_first_light_pushes_it():
    e = edge(100, first=98, full=104)
    out = set_frame(ManualEdit("display", "rising", 100), e, "full", 95, 0, 999)
    assert (out.first_frame, out.full_frame) == (95, 95)


@pytest.mark.parametrize(
    "which,value,expected",
    [("first", 101, (101, 101)), ("full", 99, (99, 99))],
)
def test_a_coincident_transition_moves_as_one(which, value, expected):
    """first == full is the normal shape for an LED source, and the graph draws
    a single triangle there. Nudging either end must move the whole thing rather
    than doing nothing."""
    e = edge(100)  # first == full == 100
    out = set_frame(ManualEdit("display", "rising", 100), e, which, value, 0, 999)
    assert (out.first_frame, out.full_frame) == expected


def test_set_frame_clamps_to_the_analysis_range():
    e = edge(100, first=98, full=104)
    lo_edit = set_frame(ManualEdit("display", "rising", 100), e, "first", -50, 50, 150)
    hi_edit = set_frame(ManualEdit("display", "rising", 100), e, "full", 9999, 50, 150)
    assert lo_edit.first_frame == 50
    assert hi_edit.full_frame == 150


def test_set_frame_undeletes():
    e = edge(100)
    out = set_frame(ManualEdit("display", "rising", 100, deleted=True), e, "first", 99, 0, 999)
    assert out.deleted is False


def test_resolve_keeps_an_automatic_end_when_only_one_is_pinned():
    e = edge(100, first=98, full=104)
    assert resolve(e, ManualEdit("display", "rising", 100, first_frame=95), 0, 999) == (95, 104)


def test_resolve_lets_a_pinned_frame_win_over_an_automatic_one_that_crossed_it():
    e = edge(100, first=98, full=99)
    assert resolve(e, ManualEdit("display", "rising", 100, first_frame=110), 0, 999) == (110, 110)


# ----------------------------------------------------------------- overrides


def test_apply_overrides_replaces_frames_and_marks_the_edge():
    edges = {100: edge(100, first=98, full=104)}
    out = apply_overrides(
        [ManualEdit("display", "rising", 100, first_frame=96, full_frame=102)],
        edges, 0, 999, "display",
    )
    assert (out[100].first_frame, out[100].full_frame) == (96, 102)
    assert out[100].manual is True
    assert edges[100].manual is False  # input untouched


def test_apply_overrides_drops_position_warnings_but_keeps_signal_ones():
    edges = {100: edge(100, warnings=(W_LOW_SNR, W_SLOW_RAMP, W_AMBIGUOUS_EDGE, W_UNSTEADY_LEVEL))}
    out = apply_overrides(
        [ManualEdit("display", "rising", 100, first_frame=98, full_frame=104)],
        edges, 0, 999, "display",
    )
    assert out[100].warnings == (W_LOW_SNR, W_UNSTEADY_LEVEL)


def test_apply_overrides_ignores_a_dormant_edit():
    edges = {100: edge(100)}
    out = apply_overrides(
        [ManualEdit("display", "rising", 555, first_frame=1)], edges, 0, 999, "display"
    )
    assert out[100] == edges[100]


def test_apply_overrides_ignores_a_deletion():
    """Deletions are handled upstream, on the anchor list, not here."""
    edges = {100: edge(100)}
    out = apply_overrides(
        [ManualEdit("display", "rising", 100, deleted=True)], edges, 0, 999, "display"
    )
    assert out[100].manual is False


def test_apply_overrides_ignores_the_other_roi():
    edges = {100: edge(100)}
    out = apply_overrides(
        [ManualEdit("original", "rising", 100, first_frame=90)], edges, 0, 999, "display"
    )
    assert out[100].first_frame == 100


def test_no_edits_leaves_the_edges_exactly_as_measured():
    """The whole feature is inert when nobody has touched anything — identical
    footage and settings must keep producing identical numbers."""
    edges = {100: edge(100, first=98, full=104), 200: edge(200, polarity="falling")}
    assert apply_overrides([], edges, 0, 999, "display") == edges


# ------------------------------------------------------------ list plumbing


def test_upsert_replaces_the_edit_for_the_same_transition():
    edits = [ManualEdit("display", "rising", 100, first_frame=98)]
    out = upsert(edits, ManualEdit("display", "rising", 100, first_frame=95))
    assert len(out) == 1 and out[0].first_frame == 95


def test_upsert_drops_an_edit_that_has_become_a_noop():
    edits = [ManualEdit("display", "rising", 100, first_frame=98)]
    assert upsert(edits, ManualEdit("display", "rising", 100)) == []


def test_upsert_keeps_edits_for_other_transitions():
    edits = [ManualEdit("display", "rising", 100), ManualEdit("display", "falling", 100)]
    out = upsert(edits, ManualEdit("display", "rising", 100, deleted=True))
    assert len(out) == 2


def test_find_edit_matches_on_roi_polarity_and_anchor():
    edits = [ManualEdit("display", "rising", 100), ManualEdit("original", "rising", 100)]
    assert find_edit(edits, "original", "rising", 100) is edits[1]
    assert find_edit(edits, "display", "falling", 100) is None
