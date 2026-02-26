"""
Tests for bead model quality improvements:
- Connected component cleanup (spatial coherence)
- CIELAB perceptual color distance
"""

import numpy as np
import pytest

from beadmodels.services.image_processing import (
    _lab_distance,
    _rgb_to_lab,
    cleanup_small_components,
    reduce_colors,
)

# ---------------------------------------------------------------------------
# CIELAB helpers
# ---------------------------------------------------------------------------


class TestRgbToLab:
    def test_pure_black(self):
        rgb = np.array([[0, 0, 0]], dtype=np.uint8)
        lab = _rgb_to_lab(rgb)
        assert lab.shape == (1, 3)
        # L* for black is 0
        assert lab[0, 0] == pytest.approx(0.0, abs=1.0)

    def test_pure_white(self):
        rgb = np.array([[255, 255, 255]], dtype=np.uint8)
        lab = _rgb_to_lab(rgb)
        # OpenCV uint8 Lab: L* range is [0, 255] (not [0, 100])
        assert lab[0, 0] == pytest.approx(255.0, abs=2.0)

    def test_batch_shape(self):
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        lab = _rgb_to_lab(rgb)
        assert lab.shape == (3, 3)

    def test_3d_input(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        lab = _rgb_to_lab(rgb)
        assert lab.shape == (4, 5, 3)


class TestLabDistance:
    def test_same_color_zero_distance(self):
        lab = np.array([[50.0, 0.0, 0.0]])
        d = _lab_distance(lab, lab)
        assert d.shape == (1, 1)
        assert d[0, 0] == pytest.approx(0.0)

    def test_distance_matrix_shape(self):
        a = np.array([[50.0, 10.0, 20.0], [30.0, -5.0, 10.0]])
        b = np.array([[50.0, 10.0, 20.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        d = _lab_distance(a, b)
        assert d.shape == (2, 3)
        # First row, first col: identical → 0
        assert d[0, 0] == pytest.approx(0.0)

    def test_perceptual_ordering(self):
        """Red and orange should be closer than red and blue in Lab space."""
        colors = np.array([[255, 0, 0], [255, 128, 0], [0, 0, 255]], dtype=np.uint8)
        lab = _rgb_to_lab(colors)
        red_lab = lab[0:1]
        orange_lab = lab[1:2]
        blue_lab = lab[2:3]
        d_red_orange = _lab_distance(red_lab, orange_lab)[0, 0]
        d_red_blue = _lab_distance(red_lab, blue_lab)[0, 0]
        assert d_red_orange < d_red_blue


# ---------------------------------------------------------------------------
# Connected component cleanup
# ---------------------------------------------------------------------------


class TestCleanupSmallComponents:
    def test_single_isolated_pixel_removed(self):
        """A single red pixel surrounded by blue should be absorbed."""
        blue = [0, 0, 255]
        red = [255, 0, 0]
        grid = np.full((5, 5, 3), blue, dtype=np.uint8)
        grid[2, 2] = red  # single isolated red pixel

        result = cleanup_small_components(grid, min_component_size=3)

        # The red pixel should have been replaced by blue
        assert list(result[2, 2]) == blue

    def test_two_pixel_cluster_removed(self):
        """A 2-pixel cluster (< min_size=3) should be absorbed."""
        blue = [0, 0, 255]
        red = [255, 0, 0]
        grid = np.full((5, 5, 3), blue, dtype=np.uint8)
        grid[2, 2] = red
        grid[2, 3] = red  # 2-pixel component

        result = cleanup_small_components(grid, min_component_size=3)

        assert list(result[2, 2]) == blue
        assert list(result[2, 3]) == blue

    def test_large_cluster_preserved(self):
        """A component >= min_size should NOT be removed."""
        blue = [0, 0, 255]
        red = [255, 0, 0]
        grid = np.full((5, 5, 3), blue, dtype=np.uint8)
        # 3-pixel L-shaped red component: (1,1), (2,1), (2,2)
        grid[1, 1] = red
        grid[2, 1] = red
        grid[2, 2] = red

        result = cleanup_small_components(grid, min_component_size=3)

        # All still red
        assert list(result[1, 1]) == red
        assert list(result[2, 1]) == red
        assert list(result[2, 2]) == red

    def test_diagonal_not_connected(self):
        """Diagonally adjacent pixels are NOT 4-connected → separate components."""
        blue = [0, 0, 255]
        red = [255, 0, 0]
        grid = np.full((5, 5, 3), blue, dtype=np.uint8)
        # Diagonal: (1,1) and (2,2) — not 4-connected, each is size 1
        grid[1, 1] = red
        grid[2, 2] = red

        result = cleanup_small_components(grid, min_component_size=3)

        assert list(result[1, 1]) == blue
        assert list(result[2, 2]) == blue

    def test_content_mask_respected(self):
        """Pixels outside content_mask should not be modified or considered."""
        blue = [0, 0, 255]
        red = [255, 0, 0]
        white = [255, 255, 255]
        grid = np.full((5, 5, 3), white, dtype=np.uint8)
        # Content is only the central 3x3
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        # Blue content area with isolated red in center
        grid[1:4, 1:4] = blue
        grid[2, 2] = red

        result = cleanup_small_components(grid, min_component_size=3, content_mask=mask)

        # Red should be replaced by blue (its neighbour in content)
        assert list(result[2, 2]) == blue
        # White border untouched
        assert list(result[0, 0]) == white

    def test_no_change_when_already_clean(self):
        """A uniform grid should not be modified."""
        blue = [0, 0, 255]
        grid = np.full((5, 5, 3), blue, dtype=np.uint8)

        result = cleanup_small_components(grid, min_component_size=3)

        np.testing.assert_array_equal(result, grid)

    def test_multiple_small_components_cleaned(self):
        """Multiple isolated pixels of different colors are all cleaned."""
        blue = [0, 0, 255]
        red = [255, 0, 0]
        green = [0, 255, 0]
        grid = np.full((7, 7, 3), blue, dtype=np.uint8)
        grid[1, 1] = red
        grid[5, 5] = green

        result = cleanup_small_components(grid, min_component_size=3)

        assert list(result[1, 1]) == blue
        assert list(result[5, 5]) == blue

    def test_replaces_with_most_common_neighbour(self):
        """When surrounded by two colors, picks the most frequent neighbour."""
        red = [255, 0, 0]
        green = [0, 255, 0]
        blue = [0, 0, 255]
        # Layout: top half green, bottom half blue, isolated red in blue zone
        grid = np.full((6, 6, 3), blue, dtype=np.uint8)
        grid[0:3, :] = green
        grid[4, 3] = red  # isolated red, neighbours: blue (3 sides) + blue above-left

        result = cleanup_small_components(grid, min_component_size=3)

        # Should become blue (majority of orthogonal neighbours)
        assert list(result[4, 3]) == blue


# ---------------------------------------------------------------------------
# CIELAB integration in reduce_colors
# ---------------------------------------------------------------------------


class TestReduceColorsWithCielab:
    def test_snaps_to_closest_perceptual_color(self):
        """When user_colors are provided, centroids snap using Lab distance."""
        # Create a simple 5x5 red image
        red_img = np.full((5, 5, 3), [200, 30, 30], dtype=np.uint8)
        user_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        result = reduce_colors(red_img, n_colors=1, user_colors=user_colors)

        # Should snap to red (255,0,0), the perceptually closest user color
        unique = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique) == 1
        assert list(unique[0]) == [255, 0, 0]
