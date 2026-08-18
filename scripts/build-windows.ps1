# Build fpv-latency-tool.exe on Windows.
# Requires uv (https://docs.astral.sh/uv/getting-started/installation/).
# Output: dist\fpv-latency-tool.exe
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
uv sync
uv run pyinstaller --clean --noconfirm main.spec
Write-Host "Built: dist\fpv-latency-tool.exe"
