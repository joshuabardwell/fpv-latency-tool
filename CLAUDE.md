# fpv-latency-tool

PyQt6 desktop app measuring glass-to-glass FPV latency: a camera films the signal source and the goggle screen in one shot; the tool detects brightness transitions in each screen's ROI, pairs them, and reports the frame offset in ms. Architecture lives in DESIGN.md — read it before structural changes and update it with them. Setup, commands, and build steps are in README.md and BUILDING.md.

## Rules

- This is a measurement instrument: identical footage and settings must keep producing identical numbers, and the workflow must stay familiar to existing users. Change behavior or UI only on explicit request.
- `core/` stays Qt-widget-free (extractor's QThread is the only Qt dependency); `ui/` depends on `core/`, never the reverse.
- Every bug fix gets a regression test and a one-line CHANGELOG.md entry; detailed reasoning goes in the commit message body.
- Extractor result signal is `extraction_done` on purpose — never name a QThread signal `finished` (shadows the built-in).
- `uv run pytest` must pass before any commit.
