"""
Tests for concentric circle layout, image analysis suggestions,
and bead economy features.
"""

import math

import numpy as np
import pytest
from PIL import Image

from beadmodels.services.image_processing import (
    ConcentricLayout,
    ImageSuggestion,
    _build_concentric_layout,
    _cleanup_concentric_components,
    _detect_concentric_background,
    _draw_concentric_grid,
    _sample_concentric_colors,
    analyze_image_suggestions,
    generate_model,
    generate_preview,
    resolve_shape_spec,
)


# ---------------------------------------------------------------------------
# Concentric layout generation
# ---------------------------------------------------------------------------


class TestBuildConcentricLayout:
    def test_basic_structure(self):
        layout = _build_concentric_layout(29)
        assert isinstance(layout, ConcentricLayout)
        assert layout.num_rings == 14
        assert layout.total_pegs > 0
        assert layout.radius == 14.5

    def test_center_peg(self):
        layout = _build_concentric_layout(10)
        # First position is always the center
        assert layout.positions[0] == (0.0, 0.0)
        assert layout.ring_indices[0] == 0

    def test_ring_count_matches(self):
        layout = _build_concentric_layout(20)
        # Each ring k should have at least 6 pegs
        for ring in range(1, layout.num_rings + 1):
            count = sum(1 for r in layout.ring_indices if r == ring)
            assert count >= 6, f"Ring {ring} has only {count} pegs"

    def test_neighbor_graph_connectivity(self):
        layout = _build_concentric_layout(15)
        # Every peg except possibly some edge ones should have neighbors
        isolated = sum(1 for nb in layout.neighbors if len(nb) == 0)
        # At most the center could be isolated in a very small board (which
        # shouldn't happen with d=15)
        assert isolated == 0, f"{isolated} isolated pegs found"

    def test_center_has_neighbors(self):
        layout = _build_concentric_layout(10)
        # Center peg should connect to ring 1 pegs
        assert len(layout.neighbors[0]) >= 6

    def test_total_pegs_approximately_pi_r_squared(self):
        diameter = 29
        layout = _build_concentric_layout(diameter)
        expected = math.pi * (diameter / 2) ** 2
        # Should be within 5% of the theoretical area
        assert abs(layout.total_pegs - expected) / expected < 0.05

    def test_small_diameter(self):
        layout = _build_concentric_layout(3)
        assert layout.num_rings == 1
        assert layout.total_pegs >= 7  # center + ring 1 (6 pegs min)

    def test_positions_within_radius(self):
        layout = _build_concentric_layout(20)
        for x, y in layout.positions:
            dist = math.sqrt(x**2 + y**2)
            assert dist <= layout.radius + 0.01


# ---------------------------------------------------------------------------
# Color sampling at concentric positions
# ---------------------------------------------------------------------------


class TestSampleConcentricColors:
    def test_uniform_image(self):
        layout = _build_concentric_layout(10)
        # Solid red image
        img = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)
        colors = _sample_concentric_colors(img, layout)
        assert colors.shape == (layout.total_pegs, 3)
        assert np.all(colors[:, 0] == 255)
        assert np.all(colors[:, 1] == 0)

    def test_output_shape(self):
        layout = _build_concentric_layout(15)
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        colors = _sample_concentric_colors(img, layout)
        assert colors.shape == (layout.total_pegs, 3)
        assert colors.dtype == np.uint8


# ---------------------------------------------------------------------------
# Concentric grid drawing
# ---------------------------------------------------------------------------


class TestDrawConcentricGrid:
    def test_output_is_pil_image(self):
        layout = _build_concentric_layout(10)
        colors = np.random.randint(0, 255, (layout.total_pegs, 3), dtype=np.uint8)
        img = _draw_concentric_grid(colors, layout)
        assert isinstance(img, Image.Image)

    def test_canvas_size(self):
        layout = _build_concentric_layout(10)
        colors = np.random.randint(0, 255, (layout.total_pegs, 3), dtype=np.uint8)
        cell_size = 10
        img = _draw_concentric_grid(colors, layout, cell_size=cell_size)
        expected_size = int(2 * layout.radius * cell_size) + cell_size
        assert img.size[0] == expected_size
        assert img.size[1] == expected_size


# ---------------------------------------------------------------------------
# Concentric spatial coherence cleanup
# ---------------------------------------------------------------------------


class TestCleanupConcentricComponents:
    def test_isolated_peg_replaced(self):
        layout = _build_concentric_layout(10)
        # All blue, except one isolated peg is red
        colors = np.full((layout.total_pegs, 3), [0, 0, 255], dtype=np.uint8)
        colors[0] = [255, 0, 0]  # center is red, surrounded by blue
        result = _cleanup_concentric_components(colors, layout, min_component_size=3)
        # Center should be absorbed by blue
        assert tuple(result[0]) == (0, 0, 255)

    def test_large_cluster_preserved(self):
        layout = _build_concentric_layout(10)
        # Half red, half blue along some division
        colors = np.full((layout.total_pegs, 3), [0, 0, 255], dtype=np.uint8)
        # Make a large cluster of red (first 20 pegs)
        for i in range(min(20, layout.total_pegs)):
            colors[i] = [255, 0, 0]
        result = _cleanup_concentric_components(colors, layout, min_component_size=3)
        # Large clusters should mostly survive
        red_count = sum(1 for c in result if tuple(c) == (255, 0, 0))
        assert red_count > 0


# ---------------------------------------------------------------------------
# Concentric background detection
# ---------------------------------------------------------------------------


class TestDetectConcentricBackground:
    def test_uniform_background(self):
        layout = _build_concentric_layout(10)
        colors = np.full((layout.total_pegs, 3), [255, 255, 255], dtype=np.uint8)
        bg = _detect_concentric_background(colors, layout)
        assert bg == (255, 255, 255)

    def test_no_background_when_varied(self):
        layout = _build_concentric_layout(10)
        # Random colors, unlikely to pass the 15% threshold consistently
        np.random.seed(42)
        colors = np.random.randint(0, 255, (layout.total_pegs, 3), dtype=np.uint8)
        bg = _detect_concentric_background(colors, layout)
        # Could be None or a color; we just verify it doesn't crash
        assert bg is None or len(bg) == 3


# ---------------------------------------------------------------------------
# Image analysis suggestions
# ---------------------------------------------------------------------------


class TestAnalyzeImageSuggestions:
    @pytest.fixture
    def simple_image_b64(self):
        """Create a simple solid-color image encoded as base64."""
        import base64
        import io

        img = Image.new("RGB", (100, 100), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @pytest.fixture
    def complex_image_b64(self):
        """Create a noisy multi-color image encoded as base64."""
        import base64
        import io

        np.random.seed(42)
        arr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_returns_suggestion_object(self, simple_image_b64):
        result = analyze_image_suggestions(image_base64=simple_image_b64)
        assert isinstance(result, ImageSuggestion)

    def test_simple_image_low_complexity(self, simple_image_b64):
        result = analyze_image_suggestions(image_base64=simple_image_b64)
        assert result.complexity_score < 0.3

    def test_complex_image_higher_complexity(self, complex_image_b64):
        result = analyze_image_suggestions(image_base64=complex_image_b64)
        assert result.complexity_score > 0.1

    def test_square_image_suggests_circle(self, simple_image_b64):
        result = analyze_image_suggestions(image_base64=simple_image_b64)
        assert result.suggested_shape in ("circle", "square")

    def test_wide_image_suggests_rectangle(self):
        import base64
        import io

        img = Image.new("RGB", (300, 100), (0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        result = analyze_image_suggestions(image_base64=b64)
        assert result.suggested_shape == "rectangle"

    def test_dominant_colors_returned(self, simple_image_b64):
        result = analyze_image_suggestions(image_base64=simple_image_b64)
        assert isinstance(result.dominant_colors, list)
        assert len(result.dominant_colors) > 0
        assert all(c.startswith("#") for c in result.dominant_colors)

    def test_fill_ratio_range(self, simple_image_b64):
        result = analyze_image_suggestions(image_base64=simple_image_b64)
        assert 0 <= result.fill_ratio <= 1

    def test_none_input_returns_default(self):
        result = analyze_image_suggestions()
        assert isinstance(result, ImageSuggestion)
        assert result.suggested_colors == 16  # default fallback


# ---------------------------------------------------------------------------
# Integration: generate_preview with concentric
# ---------------------------------------------------------------------------


class TestConcentricPreviewIntegration:
    @pytest.fixture
    def test_image_b64(self):
        import base64
        import io

        np.random.seed(42)
        arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_preview_returns_base64(self, test_image_b64):
        """Preview for a circle shape should return a valid base64 image."""
        # We can't easily create a DB shape in a unit test, so we test
        # the concentric path directly
        from beadmodels.services.image_processing import (
            _build_concentric_layout,
            _draw_concentric_grid,
            _sample_concentric_colors,
            reduce_colors,
        )

        layout = _build_concentric_layout(15)
        img = Image.open(
            __import__("io").BytesIO(
                __import__("base64").b64decode(test_image_b64)
            )
        )
        img_resized = img.resize((200, 200), Image.Resampling.LANCZOS)
        colors = _sample_concentric_colors(np.array(img_resized), layout)

        colors_2d = colors.reshape(1, -1, 3)
        reduced = reduce_colors(colors_2d, 8)
        reduced_colors = reduced.reshape(-1, 3)

        bead_img = _draw_concentric_grid(reduced_colors, layout)
        assert isinstance(bead_img, Image.Image)
        assert bead_img.size[0] > 0


# ---------------------------------------------------------------------------
# Model result includes fill_ratio
# ---------------------------------------------------------------------------


class TestModelResultFillRatio:
    def test_fill_ratio_in_result(self):
        """generate_model should include fill_ratio in the result."""
        import base64
        import io

        np.random.seed(42)
        arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = generate_model(image_base64=b64, color_reduction=4)
        assert hasattr(result, "fill_ratio")
        assert hasattr(result, "useful_beads")
        assert 0 <= result.fill_ratio <= 1
        assert result.useful_beads <= result.total_beads
