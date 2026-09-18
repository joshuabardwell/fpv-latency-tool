"""
Per-clip settings sidecar: `<video>.latency.json`.

A review pass over a clip's transitions is real work — potentially dozens of
hand-placed markers (see core.manual) — and it is worthless if reopening the file
throws it away. The sidecar makes a measurement session resumable: every analysis
parameter plus every manual edit, written beside the footage.

`SessionState` deliberately carries the SAME field names as the argparse
namespace in `ui.main_window.main`, so a restored session and a command line are
applied by one code path (`MainWindow._apply_settings`) and cannot drift apart in
what they mean. That is also what makes the precedence rule trivial to state and
to implement: the sidecar is applied first and the CLI second, so any flag the
user actually typed wins and every flag they omitted comes from the sidecar.

Reading is total: a missing, unreadable, truncated, malformed or
future-versioned file yields None and the app opens the clip with its normal
defaults. A settings file is a convenience, and no convenience is allowed to
stop a measurement tool from opening footage.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.manual import ManualEdit

# Bumped only for a change old readers must not guess at. A reader seeing a
# version it does not know returns None rather than picking fields it recognises
# out of a format whose meaning may have changed underneath them.
SIDECAR_VERSION = 1

SIDECAR_SUFFIX = ".latency.json"


@dataclass
class SessionState:
    """Everything worth restoring. Every field optional: an absent value means
    "the sidecar says nothing about this", which is different from a value that
    happens to equal the default."""

    fps: float | None = None
    roi_original: tuple[int, int, int, int] | None = None
    roi_display: tuple[int, int, int, int] | None = None
    direction: str | None = None
    min_delta: int | None = None
    min_spacing: int | None = None
    edge_sigma: float | None = None
    max_latency: int | None = None
    in_point: int | None = None
    out_point: int | None = None
    manual_edits: list[ManualEdit] = field(default_factory=list)


def sidecar_path_for(video_path: str | Path) -> Path:
    """`clip.mp4` -> `clip.mp4.latency.json`. The full video filename is kept
    rather than replacing its extension, so `clip.mp4` and `clip.mov` in one
    directory get separate sidecars."""
    p = Path(video_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def _as_roi(value) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(int(v) for v in value)


def _as_edit(value) -> ManualEdit | None:
    if not isinstance(value, dict):
        return None
    roi, polarity = value.get("roi"), value.get("polarity")
    if roi not in ("original", "display") or polarity not in ("rising", "falling"):
        return None
    first, full = value.get("first_frame"), value.get("full_frame")
    return ManualEdit(
        roi=roi,
        polarity=polarity,
        anchor_frame=int(value["anchor_frame"]),
        first_frame=None if first is None else int(first),
        full_frame=None if full is None else int(full),
        deleted=bool(value.get("deleted", False)),
    )


def load_session(video_path: str | Path) -> SessionState | None:
    """Read the sidecar beside `video_path`, or None if there isn't a usable one.

    Individual malformed entries are skipped rather than failing the whole read:
    one unreadable manual edit should not cost the user their ROIs.
    """
    path = sidecar_path_for(video_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("version") != SIDECAR_VERSION:
        return None

    def number(key, cast):
        value = raw.get(key)
        try:
            return None if value is None else cast(value)
        except (TypeError, ValueError):
            return None

    edits: list[ManualEdit] = []
    for item in raw.get("manual_edits") or []:
        try:
            edit = _as_edit(item)
        except (TypeError, ValueError, KeyError):
            continue
        if edit is not None:
            edits.append(edit)

    direction = raw.get("direction")
    try:
        return SessionState(
            fps=number("fps", float),
            roi_original=_as_roi(raw.get("roi_original")),
            roi_display=_as_roi(raw.get("roi_display")),
            direction=direction if direction in ("both", "rising", "falling") else None,
            min_delta=number("min_delta", int),
            min_spacing=number("min_spacing", int),
            edge_sigma=number("edge_sigma", float),
            max_latency=number("max_latency", int),
            in_point=number("in_point", int),
            out_point=number("out_point", int),
            manual_edits=edits,
        )
    except (TypeError, ValueError):
        return None


def save_session(video_path: str | Path, state: SessionState) -> Path:
    """Write the sidecar beside `video_path` and return its path.

    Written via a temporary file and replaced atomically: this runs on every
    nudge, so an interrupted write must not be able to leave a half-written file
    where a valid one was.
    """
    path = sidecar_path_for(video_path)
    payload = {
        "version": SIDECAR_VERSION,
        "fps": state.fps,
        "roi_original": None if state.roi_original is None else list(state.roi_original),
        "roi_display": None if state.roi_display is None else list(state.roi_display),
        "direction": state.direction,
        "min_delta": state.min_delta,
        "min_spacing": state.min_spacing,
        "edge_sigma": state.edge_sigma,
        "max_latency": state.max_latency,
        "in_point": state.in_point,
        "out_point": state.out_point,
        "manual_edits": [asdict(e) for e in state.manual_edits],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
