#!/usr/bin/env bash
# Build a single-file binary for the current OS (Linux/macOS).
# Output: dist/fpv-latency-tool
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
uv run pyinstaller --clean --noconfirm main.spec
echo "Built: $(ls -lh dist/fpv-latency-tool* | awk '{print $9, "("$5")"}')"
