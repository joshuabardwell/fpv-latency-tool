import json

from core.manual import ManualEdit
from core.session import (
    SIDECAR_VERSION,
    SessionState,
    load_session,
    save_session,
    sidecar_path_for,
)


def video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return path


def test_sidecar_sits_beside_the_clip_and_keeps_its_extension(tmp_path):
    """clip.mp4 and clip.mov in one folder must not share a settings file."""
    assert sidecar_path_for(tmp_path / "clip.mp4").name == "clip.mp4.latency.json"
    assert sidecar_path_for(tmp_path / "clip.mov").name == "clip.mov.latency.json"


def test_round_trip(tmp_path):
    path = video(tmp_path)
    state = SessionState(
        fps=239.76,
        roi_original=(10, 20, 30, 40),
        roi_display=(50, 60, 70, 80),
        direction="rising",
        min_delta=22,
        min_spacing=6,
        edge_sigma=2.5,
        max_latency=120,
        in_point=5,
        out_point=1799,
        manual_edits=[
            ManualEdit("display", "rising", 1162, first_frame=1160, full_frame=1166),
            ManualEdit("original", "falling", 400, deleted=True),
        ],
    )
    save_session(path, state)
    assert load_session(path) == state


def test_round_trip_of_an_empty_state(tmp_path):
    path = video(tmp_path)
    save_session(path, SessionState())
    assert load_session(path) == SessionState()


def test_no_sidecar_reads_as_none(tmp_path):
    assert load_session(video(tmp_path)) is None


def test_malformed_json_reads_as_none(tmp_path):
    path = video(tmp_path)
    sidecar_path_for(path).write_text("{ not json", encoding="utf-8")
    assert load_session(path) is None


def test_a_future_version_reads_as_none(tmp_path):
    """A reader that does not know the format must not pick out the fields it
    recognises — their meaning may have changed underneath it."""
    path = video(tmp_path)
    sidecar_path_for(path).write_text(
        json.dumps({"version": SIDECAR_VERSION + 1, "min_delta": 20}), encoding="utf-8"
    )
    assert load_session(path) is None


def test_a_json_document_that_is_not_an_object_reads_as_none(tmp_path):
    path = video(tmp_path)
    sidecar_path_for(path).write_text("[1, 2, 3]", encoding="utf-8")
    assert load_session(path) is None


def test_absent_keys_come_back_as_none_not_defaults(tmp_path):
    """None means "the sidecar says nothing about this", which is what lets a
    CLI flag and a restored value be told apart."""
    path = video(tmp_path)
    sidecar_path_for(path).write_text(
        json.dumps({"version": SIDECAR_VERSION, "min_delta": 20}), encoding="utf-8"
    )
    state = load_session(path)
    assert state.min_delta == 20
    assert state.min_spacing is None and state.roi_original is None
    assert state.manual_edits == []


def test_one_unreadable_edit_does_not_cost_the_user_the_rest(tmp_path):
    path = video(tmp_path)
    sidecar_path_for(path).write_text(
        json.dumps({
            "version": SIDECAR_VERSION,
            "roi_original": [1, 2, 3, 4],
            "manual_edits": [
                {"roi": "nonsense", "polarity": "rising", "anchor_frame": 1},
                {"roi": "display", "polarity": "rising", "anchor_frame": 100},
                "not even a dict",
                {"roi": "display"},  # missing anchor_frame
            ],
        }),
        encoding="utf-8",
    )
    state = load_session(path)
    assert state.roi_original == (1, 2, 3, 4)
    assert state.manual_edits == [ManualEdit("display", "rising", 100)]


def test_a_bad_scalar_is_dropped_rather_than_failing_the_read(tmp_path):
    path = video(tmp_path)
    sidecar_path_for(path).write_text(
        json.dumps({
            "version": SIDECAR_VERSION,
            "min_delta": "twenty",
            "direction": "sideways",
            "roi_display": [1, 2, 3],
            "min_spacing": 7,
        }),
        encoding="utf-8",
    )
    state = load_session(path)
    assert state.min_spacing == 7
    assert state.min_delta is None
    assert state.direction is None
    assert state.roi_display is None


def test_save_replaces_an_existing_sidecar(tmp_path):
    path = video(tmp_path)
    save_session(path, SessionState(min_delta=10))
    save_session(path, SessionState(min_delta=20))
    assert load_session(path).min_delta == 20
    assert not sidecar_path_for(path).with_suffix(".json.tmp").exists()


def test_save_returns_the_path_it_wrote(tmp_path):
    path = video(tmp_path)
    assert save_session(path, SessionState()) == sidecar_path_for(path)
