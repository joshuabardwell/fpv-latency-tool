# Changelog

## Unreleased

Findings from a fourth audit round, re-verified against this branch before fixing, plus the three-metric measurement rework.

> **The reported latency number has changed.** The tool now reports first-pixel, average and full-frame latency, where average is exactly `(first + full) / 2`. Previously it reported a single number taken at the frame of steepest brightness change, which is neither of the two endpoints — and on an asymmetric transition (fast onset, slow settle, typical of LCDs) sits early in the ramp. **Re-measuring footage with a multi-frame transition will give a different number than earlier versions.** Footage whose transitions are instantaneous is unaffected: first, full and the old number all coincide there.

### Added
- Zoom/pan on the brightness graph's X axis: a Premiere-style zoom bar above the timeline, with draggable end-handles and a draggable middle bar, plus scroll-wheel zoom and click-drag pan directly on the graph. The zoomed window recenters on the playhead as it moves (including during playback) and resets to 100% on double-click or a new analysis.
- Max Latency now defaults to half the measured Original Period after analysis (was always "unlimited"), reducing the chance of wraparound mis-pairing without requiring manual tuning; an explicit `--max-latency` CLI value or a user edit still takes precedence.
- "Auto" button next to Max Latency snaps it back to half the measured Original Period on demand, once a CLI value or manual edit has overridden the default.
- Luminance graph's playhead is now an upward-pointing triangle+stalk marker (matching the size/shape of the timeline's own playhead triangle, mirrored to point up) anchored to the bottom of the plot, replacing the thin `InfiniteLine` that was easy to lose once the graph got busy.
- Esc now cancels a running analysis, same as clicking the Cancel button.
- "<< Unmatched" / "Unmatched >>" buttons jump the playhead between unmatched transitions (shown red on the graph) without stepping through every matched pair in between.
- Matched transition pairs highlight (a white ring around each marker plus a brightened connector segment) when the playhead lands on one of the pair's frames, or when the mouse hovers a matched marker directly; unmatched markers never highlight.
- Results tables gained a per-row "Exclude" checkbox to manually drop an outlier/mismatched pair from the Mean/Min/Max/Median summary without deleting it from the table, plus per-direction "Clear All" and "Show Excluded" (filters the table to only excluded rows, without changing the summary) controls. Excluded pairs render with a muted marker/connector color on the brightness graph. Any threshold change or new Analyze run clears all exclusions, since pairs have no stable identity across a redetect. CSV export now always includes an `Excluded` Y/N column.
- Dragging a video file onto the main window opens it, same as the Open Video button/dialog.
- The results table row for the playhead's current matched pair now tints and auto-scrolls into view as the timeline is scrubbed, tracking the graph's existing ring highlight (but playhead only, not mouse hover). Auto-scroll is suppressed during active playback so the table doesn't jump around as the playhead sweeps past every transition; the row still tints.
- Three latency metrics per pair instead of one: **First pixel** (display first-light minus source first-light), **Full frame** (display fully-lit minus source fully-lit) and **Average**, the mean of the two. Each compares the same point on the transition curve at both ends, so a slow rise on the source can't inflate the result. Results tables gained First/Avg/Full (ms) columns and each direction's summary grew from one row to three, one per metric. CSV export carries all three in frames and ms.
- Brightness graph draws what was measured: a triangle at first-light, a triangle at fully-lit, and a small dim achromatic dot at the derived midpoint the average is taken at. The steepest-change point that pairing keys on is no longer drawn. Click-to-seek snaps to either triangle but never to the midpoint, which is a half-frame position no brightness sample exists at.
- Measurement-quality warnings. Per transition: `low-snr` (display fills too little of the ROI), `ambiguous-edge` (the signal re-crosses the first-light threshold, i.e. motion mid-transition), `slow-ramp` and `unsteady-level` (the region the baseline is measured from isn't level). Per ROI: unstable baseline and inconsistent contrast across the clip, which is what catches an ROI drawn larger than its screen with the screen drifting inside it. Flagged rows get a ⚠ with a tooltip naming the specific checks that failed; a banner names which ROI is at fault. Note the two are driven by different things — a steadily drifting baseline flags the ROI but flags no individual pair, because under pure drift each transition is still locally well-measured. Warnings never suppress or auto-exclude anything.
- "Exclude Flagged" button per results panel, ticking Exclude on every pair carrying a warning. It populates the existing exclusion sets, so graph muting, Clear All, Show Excluded and the CSV `Excluded` column all work unchanged.
- "Edge Sensitivity" detection parameter (and `--edge-sigma`), default 3.0σ: how far above the noise floor a signal must move before it counts as changing. Higher means first-light later and fully-lit earlier. It has no effect on noise-free footage.
- Clicking the brightness graph now seeks the playhead to the clicked frame, snapping to the nearest transition marker (matched or unmatched) if one was hovered, distinguished from click-drag-pan by movement since press. Hovering the graph now previews the click target with a dashed vertical line (real cursor hidden), which switches to a closed-hand cursor while dragging and, over any marker, a pointing-hand cursor with the line staying visible snapped to that marker's exact frame — needed since marker hover is horizontal-only (vertical position no longer matters, since the real cursor is hidden while the line shows), so two close markers' hover zones can overlap and the line is what shows which one actually wins.

### Fixed
- Quality warnings now read as English rather than as the internal check names. Hovering a flagged row's ⚠ gave "Measurement quality: unsteady-level", which names the code that fired and tells the user nothing; it now explains what was seen and why it matters — e.g. "Drifting levels — the brightness either side of this transition was still drifting rather than holding steady, so the levels this measurement is compared against are approximate. Common when the device under test has auto-exposure." The banner uses the short labels too. CSV export deliberately keeps the raw slugs, which stay greppable and stable.
- Results table's frame columns showed the internal steepest-step anchor rather than first-pixel, and were labelled only "Original Frame"/"Display Frame" so there was no way to tell. On real footage the column read 1153 where first light was at 1152. They now report first-pixel — matching the graph markers, Up/Down navigation and row-click seek — and are labelled "Original 1st Pixel"/"Display 1st Pixel" over two header lines. CSV export follows, so the two can't disagree about the same number.
- Results table column headers no longer render bold on a populated panel and plain on an empty one. Qt bolds the header holding the current item by default (`highlightSections`), which read as a deliberate distinction and wasn't one.
- Data-quality banner no longer raises a warning, or blames ROI framing, for a signal whose measurements are all correct. A baseline that moves across the clip is normal when the device under test has auto-exposure — its AE opens up through each dark stretch, and the Display ROI is showing that camera's image — so the per-ROI checks now read as information unless some measurement is also flagged. The tool no longer asserts a cause at all: framing, changing light, a nudged camera and auto-exposure produce an identical signature, and only `low-snr` still suggests checking an ROI, since that is the one flag where framing genuinely is implicated. The banner is also now recomputed on the same refresh as the results tables, so its flagged count can't disagree with them.
- Frame/transition keyboard navigation (arrows, Page Up/Down, Home/End, I/O, Space) no longer stops working after clicking the brightness graph or the left column. Both are `QAbstractScrollArea` subclasses — the graph via pyqtgraph's `PlotWidget`/`QGraphicsView`, plus the `QScrollArea` added to stop the left column overflowing — and that class handles the arrow keys itself to scroll its viewport while defaulting to `StrongFocus`, so taking click focus swallowed every navigation key before `MainWindow.keyPressEvent` saw it. Both are now `NoFocus`, matching every other widget in that column. Click-to-seek is what exposed it: before that there was no reason to click the graph.
- CSV export now follows the same pinned polarity as the results tables (`self._results_polarity`, set at the last Analyze click) instead of the live Direction pulldown, so it can no longer silently diverge from what's on screen.
- Frame/transition keyboard navigation also stopped working after dragging the timeline's in/out/playhead handles, or clicking the video preview — a variant of the click-focus bug above that the `QAbstractScrollArea` fix missed. `TimelineWidget` and `RoiFrameView` are plain widgets, not scroll areas, but both took `ClickFocus` on click/drag and don't handle keys themselves, so an ignored arrow keypress propagated up into `left_scroll` (the scroll area wrapping the whole left column), which swallowed it to scroll its viewport before `MainWindow.keyPressEvent` ever saw it. Both are now `NoFocus`; neither needs keyboard focus since dragging/drawing only needs mouse capture.
- Results table now refreshes live when Min ΔBrightness, Min Spacing, or Max Latency is adjusted after analysis, instead of staying stuck on the previous (possibly empty) matched-pairs state.
- CLI startup with a nonexistent video filename now refuses to launch instead of opening an empty GUI with only a status-bar error.
- Opening a file that fails to load (bad path, unreadable/corrupt video) now also raises a modal dialog — previously the only sign was a status-bar line that was easy to miss. Applies to every path that reaches `open_file()`: the Open Video button/dialog, drag-and-drop, and CLI startup.
- Luminance graph's rise/fall transition markers now use the correct pyqtgraph triangle symbols (rise up, fall down), fixing the fall marker rendering as a sideways triangle.
- `--out-point`/`--in-point` CLI args that conflict now warn instead of silently clamping to a different range than requested.
- Startup window now sets its geometry explicitly from the screen's available area instead of relying on `showMaximized()`'s automatic calculation, fixing a multi-monitor Windows quirk where the window reported itself maximized without actually filling the screen.
- Results table no longer steals keyboard focus on click, which was swallowing arrow-key/Home/End/I/O navigation shortcuts.
- Left column (video preview, controls, brightness graph, scrub bar) now scrolls instead of forcing the main window taller than the screen when its content's minimum height grows past the available space (e.g. once the FPS-verification row and "Analysis complete" status appear after Analyze) — previously this pushed the window behind the taskbar and could leave Detection Parameters overlapping the video preview after a subsequent Restore/Maximize.

### Changed
- **Reported latency is now `(first + full) / 2`** rather than the frame of steepest brightness change. See the note at the top of this release: numbers move on footage with a multi-frame transition, and stay identical on footage whose transitions are instantaneous.
- Up/Down transition navigation lands on each transition's first-pixel frame — one stop per transition rather than one per marker. Fully-lit is always a few frames further on and trivial to reach with the arrow keys, so giving it its own stop would triple the keypresses to cross a clip. The unmatched-transition jump buttons follow the same rule.
- Results tables no longer show latency in frames; with three metrics it would triple the width for a number that's redundant once ms is present, and the average is a half-frame value that reads badly as an integer. All three are still exported in frames as well as ms.
- CSV export replaces its `Latency (frames)`/`Latency (ms)` columns with `First`/`Avg`/`Full` in both units, plus a `Warnings` column. A single `Latency` column would now have to mean the steepest-change delta, which is no longer what the tool reports on screen, and a column that quietly disagrees with the UI is worse than a renamed one.
- Frame/Transition/Unmatched navigation buttons now use a uniform `<< Word` / `Word >>` label convention and are stacked vertically beside the timeline (Transition nearest the graph, Frame nearest the timeline) instead of laid out in a single horizontal row, shrinking the scrub bar's width.
- `BrightnessGraphWidget.set_data` computes each array's min/max once instead of redundantly rescanning for the threshold, delta, and Y-range calculations.
- Brightness graph transition markers use a darker shade than their line, and the line now paints solid through each marker instead of being interrupted by it, so markers stay visible when the line gets noisy near a transition.
- Results panel now shows two fixed 50/50 tables ("Dark To Light Transitions" / "Light To Dark Transitions"), each with its own Mean/Min/Max/Median summary row, instead of one combined table with a Direction column and a single mean/min/max text line. Drops the "#" row-number column in favor of "Original Frame" as the natural index. Both panels are always visible; a direction with no matching results just shows blank rows and "--.- ms" placeholders. The panel content is pinned to whichever direction Analyze was last run with — changing the Direction pulldown afterward stages the next run but doesn't reshuffle the loaded results until Analyze is clicked again. CSV export is unchanged (still one interleaved file with Direction and #, and still follows the pulldown live); the Export CSV button now sits below both tables instead of above them.

### Removed
- The dashed horizontal lines on the brightness graph (drawn at each signal's min/max midpoint). They were never tied to the actual delta-based detection and could be mistaken for the real detection threshold.

## v0.2.2

Findings from a third audit round (fresh code pass over previously uncovered paths, plus a full docs-vs-code consistency check).

### Fixed
- Extraction re-clips ROIs against the actually decoded frame size — container metadata lying about dimensions (rotation side-data on phone footage) could yield an empty ROI slice whose NaN mean silently poisoned all results.
- README corrections: results-table column wording, false `pip install .` claim, incomplete project-layout tree, "frame-accurate"/"any setting" overstatements; DESIGN.md delta-threshold nuance; stale "Step 2 UI" module docstring.

## v0.2.1

Findings from two adversarial audit rounds over v0.2.0 (three parallel fresh-context reviews, each finding re-verified before fixing; a follow-up round then reviewed the fixes themselves and caught one gap they introduced).

### Fixed
- Queued results/errors from an invalidated analysis session (file re-opened, ROI changed, analysis restarted) are dropped instead of overwriting the current session.
- Opening an invalid path no longer destroys the current session — the new file is validated before the old one is torn down.
- Editing or clearing an ROI mid-extraction cancels the in-flight run instead of showing old-ROI results under the new overlay.
- An aborted ROI click no longer leaves a phantom undo that wiped results as a no-op.
- Timeline in/out points stay pinned to frame 0 on single-frame videos.
- "Show CLI Options" omits `--min-delta` until a threshold has actually been applied, and a warning appears when a CLI ROI lies outside the video frame.
- Windows CI smoke test now waits for the GUI-subsystem exe and checks its exit code (it previously passed unconditionally).
- The Cancel button also invalidates the session, so a result finishing in the same instant can never populate a cancelled analysis.

### Added
- Project `CLAUDE.md` (agent instructions: stability contract, dependency direction, fix conventions).

## v0.2.0

### Fixed
- Results table and CSV headers: original/display frame columns were swapped in name.
- Analyze button no longer stays disabled after cancelling an analysis.
- Extractor result signal renamed (`finished` → `extraction_done`); it shadowed `QThread.finished`.
- An accidental click in ROI-draw mode no longer silently deletes the existing ROI.
- Ctrl+Z (undo ROI) now clears results computed from the pre-undo ROI.
- CLI `--min-delta` applies to the first analysis only instead of overriding every re-analysis.
- Arrow/Home/End keys work again inside focused spinboxes (navigation moved off window-level shortcuts).
- A user-set ΔBrightness threshold survives re-analysis; auto value shown equals value in effect.
- Same-frame transitions can pair (zero-frame latency was previously unmeasurable).
- Extraction of a file shorter than its metadata claims keeps the extracted frames instead of discarding everything.
- Unknown CLI options now error instead of being silently ignored.
- CSV export appends `.csv` when the filename doesn't already end with it.

### Changed
- Packaging: uv + `pyproject.toml` replace `requirements.txt`; pandas dependency dropped (stdlib csv writes identical output).
- Transition detection extracted to `core/detection.py`; pairing rewritten as an O(n+m) sweep.
- Live brightness readout converts only the ROI region to grayscale.

### Added
- Test suite (pytest, headless Qt) covering core logic and UI regressions.
- PyInstaller spec, build scripts, and GitHub Actions CI building Windows/Linux/macOS binaries.
- `DESIGN.md` architecture reference and `BUILDING.md`.
