# DESIGN — FPV Latency Tool

Architecture reference for contributors. Describes how the pieces fit together,
the invariants each module relies on, and the known limitations of the method.

## What the tool measures

A camera films two screens at once: the **original** signal source and the
**display** under test (e.g. FPV goggles showing the same feed through the radio
link). A blinking test pattern produces light/dark transitions on both screens.
The tool detects each transition in both regions and pairs them; the frame
offset between an original transition and its display counterpart, divided by
the capture frame rate, is the glass-to-glass latency.

Accuracy is bounded by the capture frame rate: at 240 fps each frame is ~4.2 ms,
so that is the measurement granularity. The capture camera's fps must be known
precisely — container metadata often lies (slow-motion phone/GoPro footage),
which is why the FPS is user-confirmable and cross-checkable (see below).

## Module map

```
main.py                   entry point (delegates to ui.main_window.main)
core/
  video_io.py             VideoReader: frame-accurate random access + metadata
  roi.py                  ROI dataclass: pixel rect, clipping, mean brightness
  extractor.py            BrightnessExtractor: QThread, sequential brightness pass
  detection.py            derivative-based transition detection (pure NumPy)
  latency.py              LatencyPair + pair_transitions (greedy matching)
  export.py               CSV export of latency pairs (stdlib csv)
  view_range.py            pure clamp/center/zoom/pan math for graph zoom
ui/
  main_window.py          MainWindow: layout, wiring, CLI args, CSV export
  roi_frame_view.py       RoiFrameView: frame display + click-drag ROI overlay
  brightness_graph.py     BrightnessGraphWidget: traces, detection, pair markers
  timeline.py             TimelineWidget: playhead + draggable in/out handles
  zoom_bar.py              ZoomBarWidget: graph zoom/pan bar above the timeline
```

`core/` has no Qt-widget dependencies beyond `extractor.py`'s QThread and is
importable without a GUI. `ui/` depends on `core/`, never the other way around.

## Data flow

1. **Load** — `MainWindow.open_file` builds a `VideoReader`. Scrubbing calls
   `read_frame(index)`; seeks use `CAP_PROP_POS_FRAMES` (frame-index, not
   time-based — time seeks are not reliably frame-accurate in OpenCV/ffmpeg).
2. **ROI selection** — user drags rectangles on `RoiFrameView`; it emits
   `roi_changed(name, ROI)` in *original-frame pixel coordinates*. A live
   mean-brightness readout is recomputed per scrubbed frame.
3. **Extraction** — Analyze starts a `BrightnessExtractor` thread. It opens its
   *own* `cv2.VideoCapture` (never shares the GUI reader across threads), seeks
   to the in point once and reads sequentially — one forward pass, no per-frame
   seeking. Output: two float32 arrays (mean gray brightness per frame per ROI).
   If the file ends before the metadata-reported frame count (common), the
   frames extracted so far are delivered and the status line says so.
4. **Detection** — `BrightnessGraphWidget.set_data` stores the arrays and runs
   `core.detection`: `np.diff` against a delta threshold, consecutive
   over-threshold frames collapsed to the steepest step. Detection re-runs live
   when the user changes Min ΔBrightness, Min Spacing, or Max Latency.
5. **Pairing** — `pair_transitions` greedily matches each original transition to
   the nearest display transition at or after it (same polarity: rising with
   rising, falling with falling), one-to-one, optionally capped by Max Latency.
   A same-frame match counts as zero latency. Unmatched transitions render red.
6. **Results** — pairs feed the rise/fall results tables, the mean/min/max
   summary, CSV export, and the FPS verification row (measured
   original-pattern period vs. user-entered known period → computed true
   fps).

## Threading model

Exactly two threads matter:

- **GUI thread** — everything except extraction.
- **Extractor thread** — `BrightnessExtractor.run`. Communicates only via queued
  signals (`progress`, `extraction_done`, `error`). Cancellation is a plain
  boolean flag polled once per frame; a Python bool store/load is atomic, no
  lock needed. The result signal is deliberately *not* named `finished` — that
  would shadow the built-in `QThread.finished`, which is the only signal that
  fires on every exit path (completed, cancelled, errored) and is what the GUI
  uses to re-enable controls and drop the worker reference.

The GUI-side `VideoReader` and the extractor's capture are separate
`cv2.VideoCapture` instances by design; OpenCV captures are not thread-safe.

## Coordinate spaces

Three spaces exist and `roi_frame_view.py` is the only translator:

1. **Frame pixels** — the video's native resolution. ROIs are stored here.
2. **Label pixels** — the widget area; the scaled frame is centered inside it.
3. **Scaled-pixmap pixels** — the frame after aspect-preserving scale-to-fit.

Mouse events map label → frame via `_label_to_frame` (subtract the centering
offset, divide by the scale). Drawing maps frame → scaled-pixmap in
`_redraw`. Everything outside this file works purely in frame pixels.

## Graph zoom/pan (X axis only)

`BrightnessGraphWidget` is the canonical owner of its own visible X range
(`_visible_start`/`_visible_end`, always inside `_range_lo`/`_range_hi` — the
plotted-data domain from the last `set_data` call). `ZoomBarWidget` mirrors
it rather than owning it: it has its own *outer* domain (the whole loaded
video, set once via `reset(frame_count)`, matching `TimelineWidget`'s scale
so the two widgets' pixel positions line up), plus the graph's domain drawn
inside that as an analysis-boundary marker. The bar's draggable handles are
clamped to that marker, not the outer domain — there's no plotted data
outside it to zoom into.

All range math (clamping to bounds, minimum zoom width, centering, scaling
around an anchor, panning) lives in `core/view_range.py`, pure and
Qt-free — the graph's wheel-zoom and click-drag pan, and the bar's handle
and middle-bar drag, all route through it so they can't disagree about
limits.

Sync is signal-driven, one direction per concept:
- `zoom_bar.range_changed → graph.set_visible_range` (bar drag drives the graph)
- `graph.visible_range_changed → zoom_bar.set_range` (graph-side zoom/pan — wheel,
  click-drag — mirrors back to the bar)
- `graph.domain_changed → zoom_bar.set_analysis_bounds` (new/cleared analysis
  moves the marker and resets zoom to 100%)

The playhead recenters the visible window on every frame change (keyboard
step, timeline drag, transition jump, playback tick) as part of
`BrightnessGraphWidget.set_frame` — the only call site for playhead
movement, so no other wiring is needed. Recentering clamps the *window*, not
just its center, so near a domain edge the window's own edge pins to the
boundary instead of overhanging past it — the playhead drifts off-center
and reaches the graph's edge exactly when it reaches the last analyzed
frame, with no dead space ever shown past the data.

## Detection algorithm and its limits

Detection is a **per-frame derivative threshold**: a transition exists where a
single frame-to-frame brightness step exceeds delta. Consequences:

- A slow multi-frame fade (LCD pixel response, exposure blending) where no
  single step crosses delta is **missed entirely**, even if the cumulative
  change is large. Lower delta or a faster test pattern edge is the workaround.
- Delta is auto-computed on new data (10 % of the combined brightness range,
  min 5) only while the spinbox is untouched — a user-set or CLI threshold
  survives re-analysis.
- Min Spacing suppresses double-triggers on noisy edges; Max Latency bounds the
  pairing search window so a missed display transition doesn't chain-shift all
  later pairs. Max Latency auto-computes to half the mean Original Period
  after analysis (0/"unlimited" if fewer than 2 original transitions were
  detected), following the same auto/user/CLI-override pattern as delta. An
  "Auto" button next to the spinbox recomputes and re-applies that value
  on demand; the click itself counts as a user edit, so it's a one-time
  snap rather than a standing auto mode.

## Frame-accuracy caveats

- `CAP_PROP_FRAME_COUNT` and container fps are metadata, not ground truth.
  The FPS verification row exists precisely because of this: measure the test
  pattern's period in frames, enter its known period in ms, and the true
  capture fps falls out.
- `CAP_PROP_POS_FRAMES` seeks can be approximate on long-GOP codecs. Latency
  *deltas* are immune (both ROIs come from the same frames), but absolute frame
  labels can be offset if the initial seek lands wrong.

## Known cosmetic limitations (accepted)

- Playback uses a fixed `1000/fps` ms timer and ignores decode time, so
  high-fps footage plays slower than real time. Scrubbing accuracy — the thing
  measurements depend on — is unaffected.
