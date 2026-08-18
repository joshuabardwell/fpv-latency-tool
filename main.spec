# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build spec. One-file, windowed (no console window on Windows).
# Build with:  uv run pyinstaller --clean --noconfirm main.spec
# PyInstaller cannot cross-compile — run this on the target OS (CI does).

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Qt modules the app never touches — keeps the binary smaller.
        'PyQt6.QtNetwork',
        'PyQt6.QtQml',
        'PyQt6.QtQuick',
        'PyQt6.QtPdf',
        'PyQt6.QtMultimedia',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fpv-latency-tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
