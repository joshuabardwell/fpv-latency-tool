# Changelog

## Unreleased

Findings from a fourth audit round, re-verified against this branch before fixing.

### Added
- Luminance graph's playhead is now a triangle+stalk marker anchored to the bottom of the plot, replacing the thin `InfiniteLine` that was easy to lose once the graph got busy.

### Fixed
- `--out-point`/`--in-point` CLI args that conflict now warn instead of silently clamping to a different range than requested.
- Startup window now sets its geometry explicitly from the screen's available area instead of relying on `showMaximized()`'s automatic calculation, fixing a multi-monitor Windows quirk where the window reported itself maximized without actually filling the screen.

### Changed
- `BrightnessGraphWidget.set_data` computes each array's min/max once instead of redundantly rescanning for the threshold, delta, and Y-range calculations.

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
