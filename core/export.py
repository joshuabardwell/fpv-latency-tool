"""CSV export of latency pairs. Stdlib csv (UTF-8, '\n' line ends, minimal
quoting) — the format this module produces is pinned byte-for-byte by
tests/test_export.py."""

import csv
from pathlib import Path

from core.latency import LatencyPair

# The single "Latency" pair of columns is gone, replaced by the three reported
# metrics. Keeping a lone "Latency" here would have to mean the anchor delta,
# which is no longer what the tool reports on screen — a column that quietly
# disagrees with the UI is worse than a renamed one.
CSV_COLUMNS = [
    "#",
    "Original 1st Pixel",
    "Display 1st Pixel",
    "Direction",
    "First (frames)",
    "First (ms)",
    "Avg (frames)",
    "Avg (ms)",
    "Full (frames)",
    "Full (ms)",
    "Warnings",
]


def pairs_to_rows(
    pairs: list[LatencyPair], fps: float, excluded_flags: list[bool] | None = None
) -> list[dict]:
    assert excluded_flags is None or len(excluded_flags) == len(pairs)
    rows = [
        {
            "#": i,
            # First-pixel, matching the results table. The anchor these used
            # to carry is an internal matching detail, and having the CSV and
            # the table disagree about "the frame" would be a trap.
            "Original 1st Pixel": p.orig_first_frame(),
            "Display 1st Pixel": p.disp_first_frame(),
            "Direction": "Dark→Light" if p.polarity == "rising" else "Light→Dark",
            "First (frames)": round(p.first_delta_frames(), 2),
            "First (ms)": round(p.first_delta_ms(fps), 2),
            "Avg (frames)": round(p.avg_delta_frames(), 2),
            "Avg (ms)": round(p.avg_delta_ms(fps), 2),
            "Full (frames)": round(p.full_delta_frames(), 2),
            "Full (ms)": round(p.full_delta_ms(fps), 2),
            # An exported measurement should carry its own caveats rather than
            # leaving them behind in the GUI.
            "Warnings": ";".join(p.quality_warnings()),
        }
        for i, p in enumerate(pairs, 1)
    ]
    if excluded_flags is not None:
        for row, excluded in zip(rows, excluded_flags):
            row["Excluded"] = "Y" if excluded else "N"
    return rows


def write_pairs_csv(
    path: str | Path,
    pairs: list[LatencyPair],
    fps: float,
    excluded_flags: list[bool] | None = None,
) -> None:
    columns = CSV_COLUMNS + (["Excluded"] if excluded_flags is not None else [])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs_to_rows(pairs, fps, excluded_flags))
