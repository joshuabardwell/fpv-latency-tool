from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols

from ui.brightness_graph import BrightnessGraphWidget


def make_graph(qtbot):
    g = BrightnessGraphWidget()
    qtbot.addWidget(g)
    return g


def _apex_is_up(symbol_key):
    """True if the named pyqtgraph triangle symbol's lone vertex is at the
    top (min y) -- i.e. it renders pointing up."""
    path = Symbols[symbol_key]
    ys = [path.elementAt(i).y for i in range(path.elementCount() - 1)]
    counts = {y: ys.count(y) for y in set(ys)}
    apex_y = min(counts, key=counts.get)
    return apex_y == min(ys)


class TestBrightnessGraphTransitionSymbols:
    def test_rise_markers_point_up_and_fall_markers_point_down(self, qtbot):
        """Locks in the rise/fall symbol swap (was rise="t" rendering down,
        fall="t2" rendering right) against the module docstring's
        triangle-up = rising / triangle-down = falling contract."""
        g = make_graph(qtbot)
        rise_items = (g._sc_rise_orig, g._sc_rise_disp, g._sc_unmatched_rise)
        fall_items = (g._sc_fall_orig, g._sc_fall_disp, g._sc_unmatched_fall)
        for item in rise_items:
            assert _apex_is_up(item.opts["symbol"])
        for item in fall_items:
            assert not _apex_is_up(item.opts["symbol"])
