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
  edges.py                TransitionEdge: measures each transition's extent
                          (first-light / fully-lit) + quality checks
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
   fps). Each results row has a manual Exclude checkbox that drops that pair
   from the Mean/Min/Max/Median summary and from the CSV's `Excluded` column
   flag, without hiding the row from the normal table view. Exclude state,
   the per-direction "Show Excluded" filter, and "Clear All" are all
   independent between the rise and fall panels. Exclusion has no identity
   across a redetect — pairs are keyed only by their position in the current
   list, not a stable id — so any threshold change or new Analyze run clears
   all exclusions for both directions. `MainWindow` owns the exclude sets;
   `BrightnessGraphWidget.set_excluded_pairs()` is a pure rendering hint
   (mutes the matched marker/connector color for excluded pairs) and must
   never emit `pairs_updated`, or it would immediately clear the sets it was
   just given. The results tables sit behind a `QSortFilterProxyModel`
   (first use of a proxy model in this codebase) so the "Show Excluded"
   filter doesn't require rebuilding the underlying `QStandardItemModel` —
   avoids reentering the model's own `itemChanged` signal from its handler.
   The playhead landing on a matched frame also tints that pair's row (not
   the Exclude column, to avoid the same reentrance) and scrolls it into
   view, via a second, playhead-only signal (`playhead_pair_changed`) kept
   deliberately separate from the existing hover-aware ring highlight
   (`_resolve_highlight_pair`) — hovering a marker highlights it on the graph
   but does not touch the results table. Auto-scroll is suppressed while
   `_playback_timer` is running, so continuous playback doesn't yank the
   table view on every transition it sweeps past; the row still tints.
   `MainWindow._apply_playhead_highlight` is idempotent and re-resolves the
   highlighted pair fresh against the current table on every call (from
   either the signal or the end of `_update_results_table`), so it
   self-heals regardless of which fires first around a redetect.

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

Left-click on the graph is disambiguated between click-to-seek and
click-drag-pan at `mouseReleaseEvent` time: `_pan_drag_active` still
activates unconditionally on press (as it always has, so panning itself is
unchanged), but a release whose net horizontal displacement since press is
below `_CLICK_DRAG_THRESHOLD_PX` is additionally treated as a click and
emits `frame_clicked(frame)` — wired straight to `MainWindow.show_frame`,
the same way `TimelineWidget.frame_changed` is, since the graph doesn't own
"the current frame" itself. A click snaps to the nearest transition marker
(matched *or* unmatched) within `_HOVER_HIT_R_PX` horizontal pixels if one
was hovered at press time, else seeks to the raw clicked frame. Both marker
hit-tests (`_hover_hit_test`, `_hit_test_any_marker`) compare X only, not 2D
pixel distance — the real cursor is blanked while the hover line shows (see
`_update_cursor_and_line` below), so the user has no way to see or aim by
vertical position; requiring it would make "am I on this marker" a question
they can't answer by eye. That hover state
(`_hover_any_marker_frame`) is deliberately a *second*, broader hit-test
kept independent of `_hover_matched_frame`/`_hover_hit_test` (which stays
matched-pairs-only, feeding the existing ring highlight) — merging them
would let a nearer unmatched marker silently steal the ring highlight away
from a matched one still in-radius. It must also be captured at the very
start of `mousePressEvent`, before the existing press-time hover-clear runs
(there to stop the ring highlight sticking through a pan) — reading it at
release time would always see it already cleared.

`_update_cursor_and_line` is the single place that decides the cursor shape
and the `_hover_line` (a `pg.InfiniteLine`, the same primitive once used for
the old playhead before it was swapped for the current triangle+stalk —
dashed and muted here so the two are never confused): dragging wins
(closed-hand, line hidden), then hovering any marker (pointing-hand, line
stays visible but snapped to the marker's frame rather than the raw mouse
position — with X-only hit-testing two close markers' hover radii can
overlap, and the line is what actually shows which one would be clicked),
then a plain hover (the line at `_hover_raw_frame`, real cursor blanked so
there's one indicator, not two overlapping ones), else neither. Every path
that can change what's under the mouse — press, move, release, `leaveEvent`
— ends by calling it, including `mouseReleaseEvent`, which resyncs hover
state to the release position first: `mouseMoveEvent` skips hover updates
entirely while
`_pan_drag_active` (panning takes the whole event), so without that resync
the cursor/line would stay frozen at wherever the mouse was *before* the
drag started until an incidental future move.

## Detection algorithm and its limits

Detection runs in **two stages**, deliberately kept in separate modules:

1. **Locate** (`core/detection.py`) — *which* transitions exist. A per-frame
   derivative threshold; a run of consecutive over-threshold steps collapses to
   the frame of steepest change, the **anchor**. Pairing, Min Spacing and the
   Original Period heuristic all key on anchors.
2. **Characterize** (`core/edges.py`) — *how far each one extends*. Given the
   anchor, walk outward to find first-light and fully-lit.

The anchor is not drawn and is not reported. It is the most robust point on the
curve for matching, but it is not any of the three metrics: on an asymmetric
ramp the steepest step sits early, which is what made the single number this
tool used to report neither first-pixel nor full-frame.

### The source light is not always an LED

Test setups vary: an LED bulb, some other kind of bulb, or a computer screen
playing a test pattern. **Never tune detection to one of them.** Each has a
different transition shape:

- **LED** — near-instant; first-light and fully-lit are one or two frames apart.
- **Incandescent/other bulb** — ramps slowly and *uniformly* across the whole
  lit area.
- **Screen** — lights progressively by scanline, so a small area is fully bright
  while the rest is still dark.

ROI **mean** brightness is the source-agnostic statistic, and that is why it
stays. A high percentile or a lit-pixel fraction is tempting — it detects a
screen's first scanline far more sensitively than a mean does, since a mean
barely moves when 5% of the area lights. But on a bulb that brightens uniformly
the same statistic fires the instant any single pixel crosses a threshold, which
is noise, not signal. It would optimise for one source at the others' expense.

The same applies to ramp length: a multi-frame rise is a property of the source,
not a defect, and must not be treated as bad data on its own.

### The three metrics

    first-pixel latency = display first_frame - source first_frame
    full-frame  latency = display full_frame  - source first_frame
    average     latency = mean of the two

Both metrics are anchored at the **same zero point**: the source's own
first-light frame. First-pixel compares it against the display's first-light;
full-frame against the display's fully-lit. Average is a genuine half-frame
value when the two differ, which is a resolution gain, not a rounding
artifact. Where either end could not be characterized, all three fall back to
the anchor delta rather than mixing an edge frame against an anchor frame.

Full-frame deliberately does **not** subtract the source's own `full_frame`.
An earlier version did, on the reasoning that each metric should compare "the
same point on the curve at both ends" — but two edges each individually
measured with `first_frame <= full_frame` give no guarantee about the
relationship between *one edge's* `full_frame` and the *other's*. On footage
of a flickering LED source whose plateau bounced for several frames after its
steepest step, the source's own `full_frame` landed later than its
`first_frame` purely from that bounce, and because the display settled cleanly
and fast, full-frame latency came out *below* first-pixel latency for the same
pair (29.2 ms vs 37.5 ms) — physically backwards, since the display can't
finish showing a change earlier than its own reported first-light delay would
suggest.

Anchoring both metrics at the source's first-light instead fixes this
structurally, not just for that footage: `full_delta - first_delta` is now
always exactly `display.full_frame - display.first_frame`, the display's own
ramp length, which is `>= 0` for every edge by construction (see
`TransitionEdge` below). Full-frame latency can therefore never again read
below first-pixel latency, regardless of how a given source settles.

The tradeoff is explicit, not hidden: for a source with a real, non-trivial
rise time — a computer-monitor test pattern filling by scanline is the
documented case above — that rise time now flows into full-frame and average
latency rather than canceling out against the source's own measurement. This
is intentional. A known-fast source (an LED reaching full brightness in 1-2
frames) contributes next to nothing either way; a source with a genuinely
comparable rise time to the display contributes a real, honest amount, rather
than a number whose sign depended on how precisely two independent
measurements of the source's own settling happened to agree.

For an instantaneous transition first = full = anchor, so a square-wave clip
measures exactly as it did before this split existed.

### Baseline drift, and why there is no detrending

Baseline, plateau and noise are all measured **locally**, per transition, from
the flat runs immediately either side of it. Nothing is computed once globally
and reused. Each transition calibrates against its own local levels.

"Local" means local in **time**, not merely local to the neighbouring
transitions, and the distinction is not academic. On real 240fps footage the
gap between flashes ran to 276 frames, and that stretch drifted 143 levels end
to end; a median over all of it is nothing like the level immediately before
the transition. `LEVEL_WINDOW_FRAMES` caps each level's measurement window to
the frames nearest the transition. `estimate_noise` chunks flat runs by the
same window before taking residuals, for the same reason — measured in one
pass over a whole drifting run, a display's sigma came out at 17 levels when
its actual frame-to-frame scatter was well under one.

**The band widens with the baseline's own wander** (`DRIFT_BAND_K` × the
measured tilt), and this applies to the **first-light band only**. First-light
is found by walking *backward* from the anchor, so it traverses the baseline
and will run the entire length of any creep it cannot distinguish from signal.
Real footage had a display's dark level climbing ~0.35 levels/frame for 20+
frames before each flash; with a band sized only for frame-to-frame scatter the
walk sailed straight through it and reported **0.00 ms latency**, which is
physically impossible. Fully-lit is found by scanning *forward* and stops at
the first qualifying frame, so it never traverses drift — widening its band
buys no robustness and costs real precision, and doing so made a 4-frame
synthetic ramp measure as 3.

The drift term is self-calibrating: with no tilt it contributes nothing, so
precision on clean footage is not sacrificed to robustness on drifting footage.

Sigma takes the **larger** of the local and pooled estimates. Local alone keeps
drift out, but a MAD over the dozen frames either side of one transition has
high variance and tends to come out low, and under-estimating sigma is the
dangerous direction — it narrows the band and reports first-light early. The
pooled figure is scatter about each segment's own median, so using it as a
floor costs no drift immunity.

**Local sigma itself takes the larger of pre's and post's own MAD, computed
separately — never one MAD pooled over both concatenated.** Pre (baseline) and
post (plateau) can have genuinely different noise floors: on real 240fps
footage of a flickering LED source, the dark baseline sat rock-steady while
the lit plateau carried a few levels of real scatter (PWM flicker). Pooling
let the quiet baseline's near-zero residuals drag the combined MAD down to
~0.08 despite the plateau visibly bouncing several levels — sigma this
underestimated let a too-tight band go uncrossed for several frames, delaying
that transition's `full_frame` until noise happened to cross it by chance,
which measured a 4-frame ramp on a source that reaches full brightness in 1-2
frames. Taking each side's MAD separately fixed this without touching the
pooled *global* floor above, which stays pooled across the whole signal — a
tempting whole-signal "take the max across every chunk" alternative was tried
and rejected, since one genuinely anomalous frame elsewhere in the clip (a
spurious brightness spike mid-transition, unrelated to this fix) blew that
estimate up to ~48 and made every band absurd. The dilution problem is local
to one transition's own pre/post windows; fix it there.

There is deliberately **no detrending**. For slow drift it is redundant given
local baselines; for fast drift the data is genuinely corrupt and subtracting a
trend would hide that behind a plausible-looking number. Drift fast enough to
matter within one transition is flagged instead.

### The dirty-data guard

Per transition (`TransitionEdge.warnings`): `low-snr`, `ambiguous-edge`,
`slow-ramp`, `unsteady-level`. Per signal (`SignalQuality`):
`unstable_baseline`, `inconsistent_amplitude`, each comparing transitions
against each other — baselines **within a polarity only**, since on a square
wave a rising transition's baseline is the dark level and a falling one's is
the bright level.

### The device under test moves its own baseline

The Display ROI shows the *device under test's* image. If that device has
auto-exposure — most FPV cameras do — its AE opens up through every dark
stretch of the test pattern and closes when the light fires. Measured on real
footage across one LED-off period, the display's level went
34 → 43 → 90 → 126 → 91 → 104 before the next flash: AE opening, then hunting.

That is the device working normally, not a defective recording, and the
measurements came out **correct anyway** — the drift-aware band exists exactly
for this. It is why `unstable_baseline` and `inconsistent_amplitude` describe
signal *shape* and are never presented as errors. Framing, changing light, a
nudged camera and AE on the device under test all produce an identical
signature, so **nothing may assert a cause**; the tool reports what it measured
and the person who set the shot up supplies the why.

Two retractions worth recording, because both are easy to arrive at again:

- **Do not judge "is this ROI framed correctly" with an absolute brightness
  threshold.** Counting pixels above `dark + 40` said the reference clip's
  Display ROI was 29% permanently bright and 25% permanently dark, implying
  bad framing. It was an artifact — exposure and screen brightness vary, so an
  absolute cutoff measures the wrong thing. Redone per-pixel, comparing each
  pixel only against itself at two times, **89% of that ROI participates fully**
  and the framing is fine.
- **The recording camera's exposure was locked, and background patches away
  from both screens swing *with* the light** (2.3 → 39/87/175), which is the
  light illuminating the room — the opposite sign from AE compensation. Do not
  re-diagnose the recording camera.

### Which UI each level drives

The two levels drive **different UI**, and the distinction matters: the ⚠
column and Exclude Flagged key on per-transition flags; the banner keys on the
per-signal verdict, and renders it as information rather than a warning unless
some measurement is flagged too. A steadily drifting baseline flags the *signal* but flags
no individual pair, because under pure drift every transition is still locally
well-measured and its latency is genuinely fine. Drift mild relative to the
step is absorbed by the adaptive band and raises no per-transition warning at
all — the measurement is correct, so flagging it would be noise, and a warning
that fires on every mild creep is one the user learns to ignore.

`unsteady-level` compares the **tilt** of a level region (medians of its first
and last quarters) against both a fraction of amplitude and a multiple of
sigma. Peak-to-peak was the obvious first choice and is wrong: ptp grows with
noise, so on real footage it called a perfectly flat but noisy display region
drifting — flagging every pair of a good clip, which is precisely how a warning
gets trained into being ignored. Both terms are needed; the sigma term is what
separates a noisy level from a moving one.

The guard warns and never suppresses. A flag is a prompt to look, not a
verdict; Exclude Flagged is a button the user presses, and it reuses the
existing exclusion sets rather than introducing a parallel mechanism.

Two estimator subtleties worth not re-deriving:

- The quality checks use **max-deviation-from-median, not MAD**, even though
  MAD is used for noise. MAD is robust *to outliers*, and an outlier is exactly
  what those checks exist to catch: one cycle at half amplitude among four good
  ones leaves MAD at exactly zero.
- `slow-ramp` measures the ramp against the **whole** gap to the neighbouring
  transitions, not half of it. Half made the verdict depend on how much empty
  space happened to surround a transition rather than on the transition itself.

All the thresholds are named constants in `core/edges.py`. They have been
calibrated against one real 240fps clip (4 transitions, LED source) plus
synthetic cases, which is a start, not a validation — footage from a bulb or a
screen source has not been tested at all, and those have very different
transition shapes (see above). They are still
first guesses that still need tuning against real footage.

### Limits of the locate stage

These are properties of stage 1 and are unchanged:

- A slow multi-frame fade (LCD pixel response, exposure blending) where no
  single step crosses delta is **missed entirely**, even if the cumulative
  change is large. Lower delta or a faster test pattern edge is the workaround.
  Characterization cannot rescue this: it only measures transitions the locate
  stage already found.
- A step change in ROI composition — the display snapping into frame — is
  detected as a genuine transition. Characterization now flags it, but does not
  suppress it; rejecting non-transition steps would mean changing the locate
  stage, which would move existing measurements.
- **Lowering Min ΔBrightness far enough can make auto-exposure look like a
  transition.** On the reference footage the device under test's AE moved the
  display 25–39 levels over ten frames. That is only ~3 levels per frame, so a
  per-frame derivative at the Min Δ of 20 in use never saw it — but someone
  dropping Min Δ to chase a dim transition could start detecting AE hunting as
  transitions. The tell is transitions appearing at implausibly regular
  intervals through a stretch where nothing flashed.
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
