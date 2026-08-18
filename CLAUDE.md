# fpv-latency-tool

PyQt6 desktop app measuring glass-to-glass FPV latency: a camera films the
signal source and the goggle screen in one shot; the tool detects brightness
transitions in each screen's ROI, pairs them, and reports the frame offset in
ms. Architecture lives in DESIGN.md — read it before structural changes and
update it with them.

## Commands

- `uv sync` — install deps (Python managed by uv)
- `uv run main.py` — run the app
- `uv run pytest` — test suite; runs headless (offscreen Qt), must pass before commit
- `./scripts/build.sh` — one-file binary for this OS; Windows exe comes from CI only (PyInstaller cannot cross-compile)

## Rules

- Do not change UI layout or user-visible behavior without an explicit ask — measurement tool, users rely on stable workflow and outputs.
- `core/` stays Qt-widget-free (extractor's QThread is the only Qt dependency); `ui/` depends on `core/`, never the reverse.
- Every bug fix gets a regression test and a one-line CHANGELOG.md entry; detailed reasoning goes in the commit message body.
- Extractor result signal is `extraction_done` on purpose — never name a QThread signal `finished` (shadows the built-in).
