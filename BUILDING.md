# Building

## Prerequisite: uv

All builds go through [uv](https://docs.astral.sh/uv/getting-started/installation/),
which installs the right Python and all dependencies automatically.

## Windows (main target)

The **recommended path is GitHub Actions** — every push builds
`fpv-latency-tool.exe` on a real Windows runner:

1. Push to GitHub (the `CI` workflow runs automatically, or start it from
   the *Actions* tab via *Run workflow*).
2. Open the run → *Artifacts* → download `fpv-latency-tool-windows`.
3. Tagging a commit `vX.Y.Z` publishes the Windows/Linux/macOS binaries as
   a GitHub release.

To build locally on a Windows machine instead:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
# -> dist\fpv-latency-tool.exe
```

## Linux

```bash
./scripts/build.sh
# -> dist/fpv-latency-tool
```

## macOS

Same as Linux (`./scripts/build.sh`). Built binaries are unsigned; Gatekeeper
will warn on first launch (right-click → Open).

## Running from source (any OS)

```bash
uv sync
uv run main.py
```

## Notes

- **PyInstaller cannot cross-compile.** A Windows exe must be built on
  Windows (that is what the CI workflow is for). Building the exe under Wine
  on Linux is technically possible (install Windows Python + PyInstaller
  inside a Wine prefix) but fragile and unsupported here — use CI.
- The spec (`main.spec`) builds a one-file windowed binary and excludes Qt
  modules the app does not use. First launch of a one-file build is slower
  (it unpacks to a temp dir); that is normal.
- Antivirus false positives on unsigned PyInstaller exes are common; code
  signing or a onedir build usually helps if it becomes a problem.
