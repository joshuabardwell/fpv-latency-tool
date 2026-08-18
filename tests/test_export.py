from core.export import write_pairs_csv
from core.latency import LatencyPair

# Captured verbatim from the pandas implementation this replaced
# (pd.DataFrame(rows).to_csv(index=False)) — the stdlib writer must stay
# byte-identical.
PANDAS_GOLDEN = (
    "#,Original Frame,Display Frame,Direction,Latency (frames),Latency (ms)\n"
    "1,10,13,Dark→Light,3,100.0\n"
    "2,25,28,Light→Dark,3,100.0\n"
)


class TestCsvExport:
    def test_matches_pandas_output(self, tmp_path):
        pairs = [
            LatencyPair(10, 13, "rising"),
            LatencyPair(25, 28, "falling"),
        ]
        path = tmp_path / "out.csv"
        write_pairs_csv(path, pairs, fps=30.0)
        assert path.read_text(encoding="utf-8") == PANDAS_GOLDEN

    def test_empty_pairs_writes_header_only(self, tmp_path):
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [], fps=30.0)
        assert path.read_text(encoding="utf-8") == PANDAS_GOLDEN.splitlines()[0] + "\n"

    def test_ms_rounded_to_two_decimals(self, tmp_path):
        path = tmp_path / "out.csv"
        write_pairs_csv(path, [LatencyPair(0, 1, "rising")], fps=3.0)
        line = path.read_text(encoding="utf-8").splitlines()[1]
        assert line.endswith(",333.33")
