import pytest

from core.edges import TransitionEdge
from core.export import write_pairs_csv
from core.latency import LatencyPair

# Pins the exact output format, byte for byte. This used to be captured from
# the pandas implementation the stdlib writer replaced; that equivalence stopped
# being the point once the single Latency pair of columns was replaced by the
# three reported metrics, so it now pins the current format on its own terms.
#
# The pairs below carry no edges, so all three metrics fall back to the anchor
# delta and report the same number — which is also exactly what an
# instantaneous transition produces.
FORMAT_GOLDEN = (
    "#,Original 1st Pixel,Display 1st Pixel,Direction,"
    "First (frames),First (ms),Avg (frames),Avg (ms),Full (frames),Full (ms),Warnings\n"
    "1,10,13,Dark→Light,3.0,100.0,3.0,100.0,3.0,100.0,\n"
    "2,25,28,Light→Dark,3.0,100.0,3.0,100.0,3.0,100.0,\n"
)


def edge(anchor, first, full, warnings=()):
    return TransitionEdge(
        anchor_frame=anchor, first_frame=first, full_frame=full,
        baseline=20.0, plateau=220.0, polarity="rising",
        snr=50.0, crossings=1, warnings=warnings,
    )


class TestCsvExport:
    def test_matches_pinned_format(self, tmp_path):
        pairs = [
            LatencyPair(10, 13, "rising"),
            LatencyPair(25, 28, "falling"),
        ]
        path = tmp_path / "out.csv"
        write_pairs_csv(path, pairs, fps=30.0)
        assert path.read_text(encoding="utf-8") == FORMAT_GOLDEN

    def test_empty_pairs_writes_header_only(self, tmp_path):
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [], fps=30.0)
        assert path.read_text(encoding="utf-8") == FORMAT_GOLDEN.splitlines()[0] + "\n"

    def test_ms_rounded_to_two_decimals(self, tmp_path):
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [LatencyPair(0, 1, "rising")], fps=3.0)
        line = path.read_text(encoding="utf-8").splitlines()[1]
        assert line.endswith(",333.33,")  # trailing empty Warnings field

    def test_three_metrics_land_in_their_own_columns(self, tmp_path):
        """Source ramps 8->12, display 20->28: first 12, full 16, avg 14."""
        pair = LatencyPair(10, 22, "rising",
                           orig_edge=edge(10, 8, 12), disp_edge=edge(22, 20, 28))
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [pair], fps=1000.0)
        fields = path.read_text(encoding="utf-8").splitlines()[1].split(",")
        header = FORMAT_GOLDEN.splitlines()[0].split(",")
        row = dict(zip(header, fields))
        assert row["Original 1st Pixel"] == "8"   # first-pixel, not the anchor 10
        assert row["Display 1st Pixel"] == "20"   # first-pixel, not the anchor 22
        assert row["First (frames)"] == "12.0"
        assert row["Avg (frames)"] == "14.0"
        assert row["Full (frames)"] == "16.0"

    def test_half_frame_average_survives_export(self, tmp_path):
        pair = LatencyPair(10, 22, "rising",
                           orig_edge=edge(10, 8, 12), disp_edge=edge(22, 20, 27))
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [pair], fps=1000.0)
        assert ",13.5," in path.read_text(encoding="utf-8").splitlines()[1]

    def test_warnings_are_exported_semicolon_joined(self, tmp_path):
        """An exported measurement carries its own caveats; leaving them behind
        in the GUI would let a flagged number travel as if it were clean."""
        pair = LatencyPair(10, 13, "rising",
                           orig_edge=edge(10, 10, 10, warnings=("low-snr",)),
                           disp_edge=edge(13, 13, 13, warnings=("slow-ramp",)))
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [pair], fps=30.0)
        assert path.read_text(encoding="utf-8").splitlines()[1].endswith("low-snr;slow-ramp")

    def test_excluded_column_appended_when_flags_provided(self, tmp_path):
        pairs = [
            LatencyPair(10, 13, "rising"),
            LatencyPair(25, 28, "falling"),
        ]
        path = tmp_path / "out.csv"
        write_pairs_csv(path, pairs, fps=30.0, excluded_flags=[False, True])
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == FORMAT_GOLDEN.splitlines()[0] + ",Excluded"
        assert lines[1].endswith(",N")
        assert lines[2].endswith(",Y")

    def test_excluded_flags_length_mismatch_raises(self, tmp_path):
        pairs = [LatencyPair(10, 13, "rising")]
        path = tmp_path / "out.csv"
        with pytest.raises(AssertionError):
            write_pairs_csv(path, pairs, fps=30.0, excluded_flags=[False, True])
