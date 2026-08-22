from core.view_range import clamp_range, center_range, pan_range, zoom_range


class TestClampRange:
    def test_within_bounds_is_unchanged(self):
        assert clamp_range(10, 20, 0, 100, 5) == (10, 20)

    def test_wider_than_domain_clamps_to_full_range(self):
        assert clamp_range(-50, 500, 0, 100, 5) == (0, 100)

    def test_narrower_than_min_width_grows_to_min_width(self):
        start, end = clamp_range(10, 11, 0, 100, 5)
        assert end - start == 5
        assert start == 8.0 and end == 13.0  # grown symmetrically around center 10.5

    def test_min_width_wider_than_domain_falls_back_to_full_range(self):
        assert clamp_range(4, 6, 0, 10, 999) == (0, 10)

    def test_overhang_right_pins_to_right_edge(self):
        assert clamp_range(90, 110, 0, 100, 5) == (80, 100)

    def test_overhang_left_pins_to_left_edge(self):
        assert clamp_range(-10, 10, 0, 100, 5) == (0, 20)

    def test_degenerate_domain_returns_lo_lo(self):
        assert clamp_range(0, 5, 10, 10, 5) == (10, 10)
        assert clamp_range(0, 5, 10, 5, 5) == (10, 10)


class TestCenterRange:
    def test_centers_window_of_given_width(self):
        assert center_range(50, 20, 0, 100, 5) == (40, 60)

    def test_no_dead_space_at_right_edge(self):
        """Window pinned at the domain's right edge, width unchanged — the
        thing that lets the playhead drift to the graph's edge instead of
        the window overhanging past the last analyzed frame."""
        start, end = center_range(100, 20, 0, 100, 5)
        assert end == 100
        assert end - start == 20

    def test_no_dead_space_at_left_edge(self):
        start, end = center_range(0, 20, 0, 100, 5)
        assert start == 0
        assert end - start == 20

    def test_full_zoom_pins_window_for_entire_domain(self):
        """At 100% zoom (width == full domain) centering anywhere is a no-op:
        the window is already pinned to both edges."""
        assert center_range(0, 100, 0, 100, 5) == (0, 100)
        assert center_range(100, 100, 0, 100, 5) == (0, 100)
        assert center_range(50, 100, 0, 100, 5) == (0, 100)


class TestZoomRange:
    def test_zoom_in_shrinks_around_anchor(self):
        # anchor at the window's midpoint: shrinks symmetrically
        assert zoom_range(0, 100, 0.5, 50, 0, 1000, 5) == (25, 75)

    def test_zoom_in_keeps_anchor_fixed_off_center(self):
        # anchor at 1/4 into the window stays at 1/4 into the new window
        start, end = zoom_range(0, 100, 0.5, 25, 0, 1000, 5)
        width = end - start
        assert width == 50
        assert (25 - start) / width == 0.25

    def test_zoom_out_grows_around_anchor(self):
        assert zoom_range(40, 60, 2.0, 50, 0, 1000, 5) == (30, 70)

    def test_zoom_in_floors_at_min_width(self):
        start, end = zoom_range(0, 6, 0.1, 3, 0, 1000, 5)
        assert end - start == 5

    def test_zoom_out_caps_at_full_domain(self):
        assert zoom_range(40, 60, 100.0, 50, 0, 100, 5) == (0, 100)


class TestPanRange:
    def test_shifts_by_delta(self):
        assert pan_range(10, 20, 5, 0, 100) == (15, 25)

    def test_negative_delta_shifts_left(self):
        assert pan_range(10, 20, -5, 0, 100) == (5, 15)

    def test_clamps_at_right_edge_without_changing_width(self):
        start, end = pan_range(90, 100, 20, 0, 100)
        assert (start, end) == (90, 100)

    def test_clamps_at_left_edge_without_changing_width(self):
        start, end = pan_range(0, 10, -20, 0, 100)
        assert (start, end) == (0, 10)
