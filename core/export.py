"""CSV export of latency pairs. Stdlib csv — output byte-identical to the
pandas DataFrame.to_csv() this replaced (UTF-8, '\n' line ends, minimal
quoting)."""

import csv
from pathlib import Path

from core.latency import LatencyPair

CSV_COLUMNS = [
    "#",
    "Original Frame",
    "Display Frame",
    "Direction",
    "Latency (frames)",
    "Latency (ms)",
]


def pairs_to_rows(
    pairs: list[LatencyPair], fps: float, excluded_flags: list[bool] | None = None
) -> list[dict]:
    assert excluded_flags is None or len(excluded_flags) == len(pairs)
    rows = [
        {
            "#": i,
            "Original Frame": p.orig_frame,
            "Display Frame": p.disp_frame,
            "Direction": "Dark→Light" if p.polarity == "rising" else "Light→Dark",
            "Latency (frames)": p.delta_frames(),
            "Latency (ms)": round(p.delta_ms(fps), 2),
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
