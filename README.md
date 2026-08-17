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

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python main.py
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
latency_app/
├── main.py                   # entry point
├── requirements.txt
├── core/
│   ├── extractor.py          # QThread brightness extraction worker
│   ├── latency.py            # LatencyPair dataclass + pairing algorithm
│   ├── roi.py                # ROI dataclass: pixel coords + mean_brightness()
│   └── video_io.py           # VideoReader: frame-accurate seeking, metadata
└── ui/
    ├── brightness_graph.py   # pyqtgraph brightness traces + transition markers
    ├── main_window.py        # main window: controls, layout, wiring
    ├── roi_frame_view.py     # click-drag ROI overlay on the video frame
    └── timeline.py           # playhead + in/out handle widget
```
