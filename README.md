# FPV Latency Tool — Glass-to-Glass Latency Analyzer

Measures glass-to-glass latency by analyzing a video that captures two screens side by side: the original signal source and the delayed display (e.g. FPV goggles). The tool detects light/dark transitions in each screen's region, pairs them, and reports the timing offset.

![A camera films the signal source and the FPV goggles at once; the tool finds the flash transition in each screen region and converts the frame offset into milliseconds](assets/measurement-concept.svg)

## Features

- **Video scrubbing** — frame-indexed seek with playhead, in/out point markers, play/pause
- **ROI selection** — click-and-drag rectangles over each screen; live mean-brightness readout updates as you scrub
- **Brightness extraction** — runs in a background thread; progress bar with cancel
- **Derivative-based transition detection** — finds rising and falling edges via per-frame brightness change; configurable min ΔBrightness, min spacing, and max latency window
- **Three latency metrics** — first-pixel (the frame light first appears), full-frame (the frame the screen is fully lit) and average, the mean of the two. First-pixel and full-frame both measure from the source's own first-light frame, so full-frame latency can never read below first-pixel latency
- **Transition pairing** — greedy nearest-following match; unmatched transitions highlighted red on the graph
- **Measurement-quality warnings** — flags transitions whose measurement can't be trusted (low contrast, motion mid-transition, a baseline that isn't level); warns rather than hiding, with an "Exclude Flagged" button when you agree. It also reports whether an ROI's baseline or contrast varies across the clip, but as information rather than a warning: that's normal when the device under test has auto-exposure (its AE opens up through each dark stretch), and the measurements compensate for it. The tool never guesses *why* a baseline moved — framing, changing light, a nudged camera and AE all look identical in the signal
- **Results table** — original frame, display frame, and first-pixel / average / full-frame latency in ms; click a row to jump to that frame; CSV export carries all three in frames and ms, plus any warnings
- **FPS verification** — measure the test pattern's periodicity and cross-check against the known period to compute the true frame rate
- **Manual transition editing** — the first pass is a starting point, not the last word. Walk the transitions with Up/Down, check each against the video, and nudge a marker a frame at a time (or drop it onto the playhead) when it's wrong. A false read can be deleted outright so the real transition beside it gets matched instead. Your corrections outrank the algorithm and survive every parameter change
- **Per-clip settings file** — after the first analysis, every parameter, both ROIs, the in/out points and all your manual corrections are saved to `<video>.latency.json` beside the footage, and restored when you reopen the clip. Any CLI flag you pass overrides the saved value; the ones you leave out come from the file. Turn it off with `--no-sidecar`
- **CLI parameters** — pre-fill settings from the command line for reproducible runs; "Show CLI Options" dialog copies the full command

> **Detection limitation:** transitions are found where a *single*
> frame-to-frame brightness step exceeds the Min ΔBrightness threshold. A slow
> fade spread over several frames (e.g. LCD pixel response) can be missed even
> though the total change is large — lower the threshold or use a test pattern
> with a hard edge. See DESIGN.md for details.
>
> More generally, no threshold reads every clip correctly: settings tuned
> for clean footage produce false edges on noisy footage and vice versa.
> That is why the markers are editable — see *Reviewing and correcting
> transitions* below.

## Download

Prebuilt binaries (Windows exe, Linux, macOS) are produced by the CI workflow:
grab them from the *Actions* tab of any run, or from *Releases* for tagged
versions. See [BUILDING.md](BUILDING.md) to build one yourself.

## Running from source

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
uv run main.py
```

(Works the same on Windows, Linux, and macOS. Without uv: install the
dependencies listed in `pyproject.toml` with pip, then `python main.py`.)

Run the tests with:

```bash
uv run pytest
```

## Recording good footage

The measurement is only as good as the clip. What works:

**Test pattern.** A full-screen, hard-edged black↔white flash with a known
period — 1 Hz is a good default. Enter that period in the *Known period* field
and the tool cross-checks your camera's real frame rate. Record 10+ flash
cycles so the mean/min/max latency is statistics, not a single sample. A hard
edge matters: detection triggers on a single frame-to-frame brightness step,
so a pattern that fades misses (see the note above).

**Camera.** Any phone with a slow-motion mode (120/240 fps) or an action cam
(GoPro: 120/240 fps modes) works. Each frame is `1000 / fps` ms of measurement
resolution — at 30 fps that is a coarse ±33 ms, at 240 fps ±4 ms. Settings
that matter:

- **Lock exposure, focus, and white balance** (tap-and-hold AE/AF lock on
  phones, "lock" in GoPro Protune). Auto-exposure drifts ruin the brightness
  traces.
- Steady the camera — tripod or propped, not handheld.
- Do **not** trust the file's reported fps for slow-motion clips; containers
  frequently store the playback rate, not the capture rate. That is exactly
  what the FPS verification row is for.

**Framing.** Both screens fully visible in one shot, side by side
*horizontally* rather than stacked — rolling-shutter cameras scan the frame
top to bottom, so vertical separation adds a scan-time offset between the two
regions. Film the goggle screen straight through the eyecup lens, focused,
without glare.

**The display you flash on.** Panel response time shapes the transition edge:

| Panel | Suitability | Notes |
|---|---|---|
| OLED | best | near-instant pixel response, clean hard edges |
| fast IPS | good | a few ms response, still a clear step |
| VA | usable | slow response smears the edge over frames — lower *Min ΔBrightness* if transitions go undetected |
| LCD/QLED with local dimming | usable | **disable local dimming** — backlight zones fade slowly and blur the edge |

Set the monitor bright enough that the dark/light states are clearly separated
in the footage, but not so bright the white clips or blooms. Avoid running the
screen very dim: many backlights dim with PWM flicker, which beats against the
camera frame rate and pollutes the brightness trace.

## Usage

```
python main.py [file]
               [--fps FLOAT]
               [--roi-original x,y,w,h]
               [--roi-display  x,y,w,h]
               [--direction    both|rising|falling]
               [--min-delta    INT]
               [--min-spacing  INT]
               [--max-latency  INT]
               [--edge-sigma   FLOAT]
               [--in-point     INT]
               [--out-point    INT]
               [--no-sidecar]
```

Settings are read from `<video>.latency.json` first, then any flag you
actually typed is applied on top — so the command line overrides the saved
session one setting at a time. `--no-sidecar` ignores the file entirely and
writes nothing.

## Reviewing and correcting transitions

Automatic detection gets most transitions right and some of them wrong, and
which ones depends on the footage. Rather than hiding that behind more
thresholds, the tool lets you check its work and fix it.

After Analyze, walk the clip:

1. **Down** jumps to the next transition and selects its marker. The
   *Transition Editing* panel under the graph shows which transition it is, both
   of its measured frames, and how far through the clip you are.
2. **Left / Right** step one frame at a time. Look at the video: is the frame
   before the first-light marker really dark? These keys never move the marker,
   only the playhead, so you can step around freely.
3. Fix what's wrong:

| The marker is… | Do this |
|---|---|
| right | **Down** — on to the next one |
| off by a frame or two | **Shift+←** / **Shift+→** |
| off by a lot | scrub to the frame that's actually right, then **M** |
| the wrong end | **Shift+↓** for fully-lit, **Shift+↑** for first-light |
| not a real transition | **Delete** — it drops out of pairing and the real transition beside it gets matched instead. **Delete** again to bring it back |

Every one of those has a button in the panel if you'd rather use the mouse, and
clicking a marker on the graph (or a row in the results table) selects it.

A marker you've placed is outlined in white on the graph, its pair gets a ✎ in
the results table, and the panel shows the automatic value beside yours so
**Reset** means something. A deleted transition stays visible as a grey ×.

Your corrections outrank the algorithm: changing Edge Sensitivity, Min
ΔBrightness or anything else re-measures every transition *except* the ones
you've placed by hand. They're saved beside the clip, so closing the app and
reopening the file picks the review back up where you left it.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Left / Right | Step one frame |
| Up / Down | Previous / next transition (lands on first-pixel, and selects it) |
| PgUp / PgDn | Jump ~1 second |
| Space | Play / pause |
| I / O | Set in / out point at playhead |
| Home / End | Jump to in / out point |
| Shift+← / Shift+→ | Move the selected marker one frame |
| Shift+↑ / Shift+↓ | Select first-light / fully-lit |
| M | Move the selected marker to the playhead |
| Delete | Delete the selected transition / restore it |
| Esc | Clear the selection (cancels analysis while one is running) |
| Ctrl+Z | Undo last ROI change |
| F1 / ? | Show this help |

## Project layout

```
fpv-latency-tool/
├── main.py                   # entry point
├── pyproject.toml            # dependencies (managed with uv)
├── main.spec                 # PyInstaller build spec
├── DESIGN.md                 # architecture reference
├── BUILDING.md               # binary build instructions
├── CHANGELOG.md
├── CLAUDE.md                 # AI-agent instructions
├── scripts/                  # local build scripts (Linux/macOS, Windows)
├── assets/                   # README diagrams (SVG)
├── .github/workflows/        # CI: tests + binaries on all 3 OSes
├── core/
│   ├── detection.py          # derivative-based transition detection (which)
│   ├── edges.py              # transition extent + quality checks (how far)
│   ├── export.py             # CSV export
│   ├── extractor.py          # QThread brightness extraction worker
│   ├── latency.py            # LatencyPair dataclass + pairing algorithm
│   ├── manual.py             # user-placed transitions, overriding detection
│   ├── roi.py                # ROI dataclass: pixel coords + mean_brightness()
│   ├── session.py            # <video>.latency.json per-clip settings file
│   └── video_io.py           # VideoReader: frame-accurate seeking, metadata
├── ui/
│   ├── brightness_graph.py   # pyqtgraph brightness traces + transition markers
│   ├── edit_panel.py         # selected-marker readout + editing buttons
│   ├── main_window.py        # main window: controls, layout, wiring
│   ├── roi_frame_view.py     # click-drag ROI overlay on the video frame
│   └── timeline.py           # playhead + in/out handle widget
└── tests/                    # pytest suite (runs headless, see conftest.py)
```

Data flow through the modules (see [DESIGN.md](DESIGN.md) for the full picture):

![Data flow: video file through VideoReader, ROI selection, brightness extraction, transition detection and pairing, to the results table and CSV export](assets/architecture.svg)
