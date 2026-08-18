# FPV Latency Tool — Glass-to-Glass Latency Analyzer

Measures glass-to-glass latency by analyzing a video that captures two screens side by side: the original signal source and the delayed display (e.g. FPV goggles). The tool detects light/dark transitions in each screen's region, pairs them, and reports the timing offset.

## Features

- **Video scrubbing** — frame-accurate seek with playhead, in/out point markers, play/pause
- **ROI selection** — click-and-drag rectangles over each screen; live mean-brightness readout updates as you scrub
- **Brightness extraction** — runs in a background thread; progress bar with cancel
- **Derivative-based transition detection** — finds rising and falling edges via per-frame brightness change; configurable min ΔBrightness, min spacing, and max latency window
- **Transition pairing** — greedy nearest-following match; unmatched transitions highlighted red on the graph
- **Results table** — transition #, display frame, output frame, direction, latency in frames and ms; click a row to jump to that frame; CSV export
- **FPS verification** — measure the test pattern's periodicity and cross-check against the known period to compute the true frame rate
- **CLI parameters** — pre-fill any setting from the command line for reproducible runs; "Show CLI Options" dialog copies the full command

> **Detection limitation:** transitions are found where a *single*
> frame-to-frame brightness step exceeds the Min ΔBrightness threshold. A slow
> fade spread over several frames (e.g. LCD pixel response) can be missed even
> though the total change is large — lower the threshold or use a test pattern
> with a hard edge. See DESIGN.md for details.

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

(Works the same on Windows, Linux, and macOS. Plain pip also works:
`pip install .` then `python main.py`.)

Run the tests with:

```bash
uv run pytest
```

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
               [--in-point     INT]
               [--out-point    INT]
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Left / Right | Step one frame |
| Up / Down | Previous / next transition |
| PgUp / PgDn | Jump ~1 second |
| Space | Play / pause |
| I / O | Set in / out point at playhead |
| Home / End | Jump to in / out point |
| Ctrl+Z | Undo last ROI change |
| F1 / ? | Show this help |

## Project layout

```
fpv-latency-tool/
├── main.py                   # entry point
├── pyproject.toml            # dependencies (managed with uv)
├── main.spec                 # PyInstaller build spec (see BUILDING.md)
├── DESIGN.md                 # architecture reference
├── core/
│   ├── detection.py          # derivative-based transition detection
│   ├── export.py             # CSV export
│   ├── extractor.py          # QThread brightness extraction worker
│   ├── latency.py            # LatencyPair dataclass + pairing algorithm
│   ├── roi.py                # ROI dataclass: pixel coords + mean_brightness()
│   └── video_io.py           # VideoReader: frame-accurate seeking, metadata
├── ui/
│   ├── brightness_graph.py   # pyqtgraph brightness traces + transition markers
│   ├── main_window.py        # main window: controls, layout, wiring
│   ├── roi_frame_view.py     # click-drag ROI overlay on the video frame
│   └── timeline.py           # playhead + in/out handle widget
└── tests/                    # pytest suite (runs headless, see conftest.py)
```
