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
  latency.py              LatencyPair + pair_transitions (greedy matching)
ui/
  main_window.py          MainWindow: layout, wiring, CLI args, CSV export
  roi_frame_view.py       RoiFrameView: frame display + click-drag ROI overlay
  brightness_graph.py     BrightnessGraphWidget: traces, detection, pair markers
  timeline.py             TimelineWidget: playhead + draggable in/out handles
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
4. **Detection** — `BrightnessGraphWidget.set_data` stores the arrays and runs
   derivative-based detection: `np.diff` against a delta threshold, consecutive
   over-threshold frames collapsed to the steepest step. Detection re-runs live
   when the user changes Min ΔBrightness, Min Spacing, or Max Latency.
5. **Pairing** — `pair_transitions` greedily matches each original transition to
   the nearest following display transition of the same polarity (rising with
   rising, falling with falling), one-to-one, optionally capped by Max Latency.
   Unmatched transitions render red.
6. **Results** — pairs feed the table, the mean/min/max summary, CSV export, and
   the FPS verification row (measured original-pattern period vs. user-entered
   known period → computed true fps).

## Threading model

Exactly two threads matter:

- **GUI thread** — everything except extraction.
- **Extractor thread** — `BrightnessExtractor.run`. Communicates only via queued
  signals (`progress`, results, `error`). Cancellation is a plain boolean flag
  polled once per frame; a Python bool store/load is atomic, no lock needed.

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

## Detection algorithm and its limits

Detection is a **per-frame derivative threshold**: a transition exists where a
single frame-to-frame brightness step exceeds delta. Consequences:

- A slow multi-frame fade (LCD pixel response, exposure blending) where no
  single step crosses delta is **missed entirely**, even if the cumulative
  change is large. Lower delta or a faster test pattern edge is the workaround.
- Delta is auto-set to 10 % of the combined brightness range (min 5) whenever
  new data is loaded, and can be overridden.
- Min Spacing suppresses double-triggers on noisy edges; Max Latency bounds the
  pairing search window so a missed display transition doesn't chain-shift all
  later pairs.

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
