# Changelog

## Unreleased

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
- CSV export appends `.csv` when no extension is given.

### Changed
- Packaging: uv + `pyproject.toml` replace `requirements.txt`; pandas dependency dropped (stdlib csv writes identical output).
- Transition detection extracted to `core/detection.py`; pairing rewritten as an O(n+m) sweep.
- Live brightness readout converts only the ROI region to grayscale.

### Added
- Test suite (pytest, headless Qt) covering core logic and UI regressions.
- PyInstaller spec, build scripts, and GitHub Actions CI building Windows/Linux/macOS binaries.
- `DESIGN.md` architecture reference and `BUILDING.md`.
