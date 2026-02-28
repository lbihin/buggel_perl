"""
Service de traitement d'image pour le wizard.

Fonctions pures : pas de dependance a request/messages.
"""

import base64
import io
import logging
import math
import os
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np
from django.core.files.storage import default_storage
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreviewResult:
    image_base64: str
    reduced_pixels: np.ndarray | None = None
    content_mask: np.ndarray | None = None
    concentric_colors: np.ndarray | None = None  # (N, 3) for concentric mode


@dataclass
class ShapeSpec:
    grid_width: int = 29
    grid_height: int = 29
    use_circle_mask: bool = False
    circle_diameter: int = 0
    shape_found: bool = False


@dataclass
class ConcentricLayout:
    """Peg positions for a concentric circular board."""

    positions: list[tuple[float, float]]  # (x, y) center-based coords
    ring_indices: list[int]  # ring index for each peg
    num_rings: int
    total_pegs: int
    radius: float  # in ring-spacing units
    neighbors: list[list[int]]  # adjacency list for spatial coherence


@dataclass
class ModelResult:
    image_base64: str = ""
    grid_width: int = 29
    grid_height: int = 29
    shape_id: str | None = None
    color_reduction: int = 16
    total_beads: int = 0
    useful_beads: int = 0  # beads excluding background
    fill_ratio: float = 1.0  # useful / total
    palette: list[dict] = field(default_factory=list)


@dataclass
class ImageSuggestion:
    """Results from automated image analysis for parameter suggestion."""

    suggested_shape: str = "square"  # "circle", "square", "rectangle"
    suggested_colors: int = 16
    suggested_size: int = 29  # pegs on smallest dimension
    complexity_score: float = 0.5  # 0=simple, 1=complex
    content_ratio: float = 1.0  # how much of the image is content vs bg
    aspect_ratio: float = 1.0
    dominant_colors: list[str] = field(default_factory=list)  # hex colors
    fill_ratio: float = 1.0  # estimated board usage efficiency


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def save_temp_image(uploaded_file) -> str:
    ext = (uploaded_file.name.split(".")[-1] or "png").lower()
    filename = f"temp_wizard/{uuid.uuid4()}.{ext}"
    return default_storage.save(filename, uploaded_file)


def cleanup_temp_images(max_age_seconds: int = 3600) -> Iterable[str]:
    base_dir = "temp_wizard"
    deleted: list[str] = []
    try:
        if not default_storage.exists(base_dir):
            return deleted
        _, files = default_storage.listdir(base_dir)
        for entry in files:
            rel_path = f"{base_dir}/{entry}"
            try:
                absolute_path = default_storage.path(rel_path)
                mtime = os.path.getmtime(absolute_path)
                if (time.time() - mtime) > max_age_seconds:
                    default_storage.delete(rel_path)
                    deleted.append(rel_path)
            except Exception:
                continue
    except Exception:
        pass
    return deleted


def file_to_base64(path: str) -> str:
    if not path:
        return ""
    with default_storage.open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Color helpers  (CIELAB perceptual distance)
# ---------------------------------------------------------------------------


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an array of RGB uint8 colors to CIELAB.

    *rgb* can be (N, 3) or (H, W, 3).  Returns same shape with float64 Lab
    values.
    """
    # cv2 expects uint8 BGR
    if rgb.ndim == 2:
        bgr = rgb[:, ::-1].reshape(1, -1, 3).astype(np.uint8)
    else:
        bgr = rgb[:, :, ::-1].astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    if rgb.ndim == 2:
        return lab.reshape(-1, 3)
    return lab


def _lab_distance(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Euclidean distance in CIELAB space (Delta-E 76).

    *lab_a*: (N, 3)  *lab_b*: (M, 3) → returns (N, M) distance matrix.
    """
    return np.sqrt(np.sum((lab_a[:, None, :] - lab_b[None, :, :]) ** 2, axis=2))


# ---------------------------------------------------------------------------
# Color reduction
# ---------------------------------------------------------------------------


def estimate_optimal_colors(
    image_array: np.ndarray,
    max_k: int = 24,
    min_k: int = 2,
) -> int:
    """Estimate the optimal number of distinct colours in *image_array*.

    Strategy
    --------
    1. Sub-sample pixels (≤ 2 000) for speed.
    2. Run KMeans for k = *min_k* … *max_k* and record inertia (sum of
       squared distances to centroid).
    3. Detect the "elbow" (knee) in the inertia curve — the k after which
       adding more clusters no longer yields a significant drop.

    The elbow is found by computing the second-order finite differences of
    inertia and picking the k where curvature is maximal (Kneedle-style
    heuristic, without an extra dependency).

    Returns
    -------
    int — suggested number of colours.
    """
    pixels = image_array.reshape(-1, 3).astype(np.float64)

    # Quick pre-check: count truly unique colours (up to max_k + 1)
    unique = np.unique(pixels, axis=0)
    if len(unique) <= max_k:
        # Image has fewer distinct colours than max_k → just use that count
        return max(min_k, len(unique))

    # Sub-sample for speed
    rng = np.random.RandomState(42)
    sample_size = min(2000, len(pixels))
    sample = pixels[rng.choice(len(pixels), sample_size, replace=False)]

    # Compute inertia for each k
    ks = list(range(min_k, max_k + 1))
    inertias: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init="auto", max_iter=100)
        km.fit(sample)
        inertias.append(km.inertia_)

    inertias_arr = np.array(inertias)

    # Normalise k and inertia to [0, 1] for curvature detection
    k_norm = (np.array(ks, dtype=float) - ks[0]) / (ks[-1] - ks[0])
    i_norm = (inertias_arr - inertias_arr[-1]) / (
        inertias_arr[0] - inertias_arr[-1] + 1e-9
    )

    # Distance of each point to the line from first to last point
    # (geometric elbow detection)
    p0 = np.array([k_norm[0], i_norm[0]])
    p1 = np.array([k_norm[-1], i_norm[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return min_k

    distances = []
    for j in range(len(ks)):
        pj = np.array([k_norm[j], i_norm[j]])
        # perpendicular distance to line p0→p1
        d = abs(np.cross(line_vec, p0 - pj)) / line_len
        distances.append(d)

    best_idx = int(np.argmax(distances))
    return ks[best_idx]


def reduce_colors(
    image_array: np.ndarray,
    n_colors: int,
    user_colors: np.ndarray | None = None,
    content_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = image_array.shape[:2]

    # When a content mask is provided, only cluster content pixels so that
    # background/padding pixels don't consume a user colour slot.
    if content_mask is not None:
        flat_mask = content_mask.reshape(-1)
        all_pixels = image_array.reshape(-1, 3)
        content_pixels = all_pixels[flat_mask]
        if content_pixels.shape[0] == 0:
            return image_array.copy()
        n_colors = min(n_colors, content_pixels.shape[0])
        kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
        kmeans.fit(content_pixels)
        centroids = kmeans.cluster_centers_.astype(int)

        # Snap near-black centroids to pure black so thin details
        # (eyes, outlines) render crisply instead of muddy grey.
        for i in range(len(centroids)):
            if centroids[i].mean() < 50:
                centroids[i] = [0, 0, 0]

        if user_colors is not None and len(user_colors):
            centroids_lab = _rgb_to_lab(centroids.astype(np.uint8))
            user_lab = _rgb_to_lab(user_colors.astype(np.uint8))
            dists = _lab_distance(centroids_lab, user_lab)
            for i in range(len(centroids)):
                centroids[i] = user_colors[int(np.argmin(dists[i]))]

        # Assign content pixels to their nearest centroid
        labels = kmeans.predict(content_pixels)
        result = image_array.copy().reshape(-1, 3)
        result[flat_mask] = centroids[labels]
        # Background pixels stay as-is (white padding)
        return result.reshape(h, w, 3).astype("uint8")

    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_.astype(int)

    # Snap near-black centroids to pure black
    for i in range(len(centroids)):
        if centroids[i].mean() < 50:
            centroids[i] = [0, 0, 0]

    if user_colors is not None and len(user_colors):
        # Use CIELAB perceptual distance for color matching
        centroids_lab = _rgb_to_lab(centroids.astype(np.uint8))
        user_lab = _rgb_to_lab(user_colors.astype(np.uint8))
        dists = _lab_distance(centroids_lab, user_lab)  # (n_centroids, n_user)
        for i in range(len(centroids)):
            centroids[i] = user_colors[int(np.argmin(dists[i]))]

    labels = kmeans.labels_
    reduced_pixels = centroids[labels].reshape(image_array.shape)
    return reduced_pixels.astype("uint8")


# ---------------------------------------------------------------------------
# Spatial coherence  (connected component cleanup)
# ---------------------------------------------------------------------------


def cleanup_small_components(
    pixels: np.ndarray,
    min_component_size: int = 3,
    content_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove small isolated bead clusters to guarantee spatial coherence.

    For each unique color, find 4-connected components.  Components smaller
    than *min_component_size* are replaced by the most-frequent neighbouring
    colour (within a 3×3 window).  The process repeats until stable.

    Only pixels inside *content_mask* (if provided) are processed.

    Parameters
    ----------
    pixels : np.ndarray, shape (H, W, 3), uint8
        The colour-reduced pixel grid.
    min_component_size : int
        Components with fewer pixels than this are absorbed.
    content_mask : np.ndarray | None, shape (H, W), bool
        True for pixels that belong to actual content (not background/padding).

    Returns
    -------
    np.ndarray – cleaned pixel grid, same shape.
    """
    result = pixels.copy()
    h, w = result.shape[:2]

    # 4-connectivity structuring element (no diagonals — beads only touch
    # orthogonal neighbours when placed on a pegboard)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected

    MAX_ITERATIONS = 10
    for _iteration in range(MAX_ITERATIONS):
        changed = False
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)

        for color in unique_colors:
            # Binary mask for this colour
            color_mask = np.all(result == color, axis=2)
            if content_mask is not None:
                color_mask &= content_mask

            if not color_mask.any():
                continue

            # Preserve dark/black components regardless of size — they
            # represent important fine details (eyes, outlines, nostrils).
            if float(color.mean()) < 50:
                continue

            labeled, n_components = ndimage.label(color_mask, structure=struct)
            if n_components == 0:
                continue

            sizes = ndimage.sum(color_mask, labeled, range(1, n_components + 1))

            for comp_id, size in enumerate(sizes, start=1):
                if size >= min_component_size:
                    continue

                # Find pixels of this small component
                comp_mask = labeled == comp_id
                ys, xs = np.where(comp_mask)

                # Collect neighbouring colours (3×3 window, exclude self)
                neighbour_colors = []
                for py, px in zip(ys, xs):
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if content_mask is not None and not content_mask[ny, nx]:
                                continue
                            nc = tuple(result[ny, nx])
                            if nc != tuple(color):
                                neighbour_colors.append(nc)

                if not neighbour_colors:
                    continue  # surrounded by same colour or edge — skip

                # Pick the most frequent neighbour colour
                from collections import Counter

                best_color = Counter(neighbour_colors).most_common(1)[0][0]
                result[comp_mask] = best_color
                changed = True

        if not changed:
            break

    return result


# ---------------------------------------------------------------------------
# Concentric circle layout
# ---------------------------------------------------------------------------


def _build_concentric_layout(diameter: int) -> ConcentricLayout:
    """Generate concentric peg positions for a circular board.

    Pegs are arranged in concentric rings radiating from the center.
    Each ring at distance *k* from center has approximately ``2*pi*k``
    pegs evenly distributed, matching the physical peg spacing of real
    circular bead boards.

    Parameters
    ----------
    diameter : int
        Board diameter in peg-spacing units.

    Returns
    -------
    ConcentricLayout with positions, neighbor graph, and metadata.
    """
    radius = diameter / 2.0
    num_rings = int(radius)

    positions: list[tuple[float, float]] = [(0.0, 0.0)]  # center peg
    ring_indices: list[int] = [0]

    for ring in range(1, num_rings + 1):
        n_pegs = max(6, round(2 * math.pi * ring))
        for i in range(n_pegs):
            angle = 2 * math.pi * i / n_pegs
            x = ring * math.cos(angle)
            y = ring * math.sin(angle)
            positions.append((x, y))
            ring_indices.append(ring)

    total_pegs = len(positions)

    # Build neighbor graph using spatial proximity
    pos_arr = np.array(positions)
    neighbors: list[list[int]] = [[] for _ in range(total_pegs)]

    if total_pegs > 0:
        tree = cKDTree(pos_arr)
        # Neighbor distance threshold: ~1.5× ring spacing
        max_dist = 1.6
        for i in range(total_pegs):
            nearby = tree.query_ball_point(pos_arr[i], max_dist)
            neighbors[i] = [j for j in nearby if j != i]

    return ConcentricLayout(
        positions=positions,
        ring_indices=ring_indices,
        num_rings=num_rings,
        total_pegs=total_pegs,
        radius=radius,
        neighbors=neighbors,
    )


def _sample_concentric_colors(
    img_array: np.ndarray, layout: ConcentricLayout
) -> np.ndarray:
    """Sample pixel colors from an image at concentric peg positions.

    Maps each peg position from board coordinates ``[-radius, radius]``
    to pixel coordinates and performs bilinear interpolation.
    """
    h, w = img_array.shape[:2]
    colors = np.zeros((layout.total_pegs, 3), dtype=np.uint8)
    diameter = 2 * layout.radius

    for i, (x, y) in enumerate(layout.positions):
        # Map from center-based coords to [0, 1]
        nx = (x + layout.radius) / diameter
        ny = (y + layout.radius) / diameter
        # Map to pixel coords
        px = min(w - 1, max(0, int(nx * (w - 1))))
        py = min(h - 1, max(0, int(ny * (h - 1))))
        colors[i] = img_array[py, px]

    return colors


def _draw_concentric_grid(
    colors: np.ndarray,
    layout: ConcentricLayout,
    cell_size: int = 10,
) -> Image.Image:
    """Draw a concentric bead visualization.

    Each bead is rendered as a circle at its peg position with a subtle
    border, producing the characteristic concentric-ring visual of
    circular bead boards.
    """
    canvas_px = int(2 * layout.radius * cell_size) + cell_size
    canvas = Image.new("RGB", (canvas_px, canvas_px), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    center = canvas_px / 2.0
    bead_r = cell_size * 0.42  # visual radius of each bead dot

    for i, (x, y) in enumerate(layout.positions):
        cx = center + x * cell_size
        cy = center + y * cell_size
        r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])

        # Border circle
        draw.ellipse(
            [cx - bead_r, cy - bead_r, cx + bead_r, cy + bead_r],
            fill=(r, g, b),
            outline=(200, 200, 200),
            width=1,
        )

    return canvas


def _cleanup_concentric_components(
    colors: np.ndarray,
    layout: ConcentricLayout,
    min_component_size: int = 3,
) -> np.ndarray:
    """Remove small isolated bead clusters in a concentric layout.

    Works like ``cleanup_small_components`` but operates on a graph
    structure (neighbor list) instead of a 2D pixel grid.
    """
    result = colors.copy()
    n = layout.total_pegs

    MAX_ITERATIONS = 10
    for _iteration in range(MAX_ITERATIONS):
        changed = False
        unique_colors = np.unique(result, axis=0)

        for color in unique_colors:
            # Find all pegs with this color
            mask = np.all(result == color, axis=1)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            # BFS to find connected components using neighbor graph
            visited = set()
            components: list[list[int]] = []

            for idx in indices:
                if idx in visited:
                    continue
                # BFS from this peg
                component = []
                queue = [idx]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    if not mask[node]:
                        continue
                    visited.add(node)
                    component.append(node)
                    for nb in layout.neighbors[node]:
                        if nb not in visited and mask[nb]:
                            queue.append(nb)
                components.append(component)

            # Check each component
            for comp in components:
                if len(comp) >= min_component_size:
                    continue

                # Find most common neighbor color
                neighbor_colors: list[tuple] = []
                for peg_idx in comp:
                    for nb in layout.neighbors[peg_idx]:
                        nc = tuple(result[nb])
                        if nc != tuple(color):
                            neighbor_colors.append(nc)

                if not neighbor_colors:
                    continue

                best = Counter(neighbor_colors).most_common(1)[0][0]
                for peg_idx in comp:
                    result[peg_idx] = best
                changed = True

        if not changed:
            break

    return result


def _detect_concentric_background(
    colors: np.ndarray, layout: ConcentricLayout
) -> tuple | None:
    """Detect background color in concentric layout.

    Uses the outermost ring (border) as the sampling area.
    """
    if layout.total_pegs < 3:
        return None

    # Collect colors from outermost ring
    max_ring = max(layout.ring_indices)
    border_indices = [i for i, r in enumerate(layout.ring_indices) if r == max_ring]
    if not border_indices:
        return None

    border_colors = colors[border_indices]
    unique, counts = np.unique(border_colors, axis=0, return_counts=True)
    bg_color = tuple(unique[np.argmax(counts)])

    # Verify it's significant (> 15% of all pegs)
    matches = np.all(colors == bg_color, axis=1)
    ratio = matches.sum() / len(colors)
    if ratio > 0.15:
        return bg_color
    return None


def _detect_background_color(
    reduced_pixels: np.ndarray,
    content_mask: np.ndarray | None = None,
) -> tuple | None:
    """Detect the background color by sampling edge pixels.

    Heuristic: the most frequent color among the 1-pixel border of the grid
    is likely the background.  If that color represents > 15% of all content
    pixels it is flagged as background.
    """
    h, w = reduced_pixels.shape[:2]
    if h < 3 or w < 3:
        return None

    # Collect border pixels (top, bottom, left, right rows/cols)
    border = np.concatenate(
        [
            reduced_pixels[0, :],  # top row
            reduced_pixels[-1, :],  # bottom row
            reduced_pixels[1:-1, 0],  # left col (excl. corners)
            reduced_pixels[1:-1, -1],  # right col (excl. corners)
        ]
    )

    # Most frequent border color
    unique, counts = np.unique(border, axis=0, return_counts=True)
    bg_color = tuple(unique[np.argmax(counts)])

    # Verify it is significant inside the content area
    if content_mask is not None:
        content_pixels = reduced_pixels[content_mask]
    else:
        content_pixels = reduced_pixels.reshape(-1, 3)

    matches = np.all(content_pixels == bg_color, axis=1)
    ratio = matches.sum() / content_pixels.shape[0]
    if ratio > 0.15:
        return bg_color
    return None


def compute_palette(
    total_beads: int = 0,
    reduced_pixels: np.ndarray | None = None,
    content_mask: np.ndarray | None = None,
    image_base64: str | None = None,
    concentric_colors: np.ndarray | None = None,
    concentric_layout: ConcentricLayout | None = None,
) -> list[dict]:
    # Determine pixel source and background
    bg_color = None
    if concentric_colors is not None:
        pixels = concentric_colors  # already (N, 3)
        if concentric_layout is not None:
            bg_color = _detect_concentric_background(
                concentric_colors, concentric_layout
            )
    elif reduced_pixels is not None:
        if content_mask is not None:
            pixels = reduced_pixels[content_mask]
        else:
            pixels = reduced_pixels.reshape(-1, 3)
        bg_color = _detect_background_color(reduced_pixels, content_mask)
    elif image_base64:
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pixels = np.array(img).reshape(-1, 3)
    else:
        return []

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    safe_total = total_beads if total_beads > 0 else pixels.shape[0]
    palette = []
    for color, count in zip(unique_colors, counts):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        percentage = (int(count) / safe_total) * 100
        is_bg = bg_color is not None and (r, g, b) == bg_color
        palette.append(
            {
                "color": f"rgb({r}, {g}, {b})",
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "count": int(count),
                "percentage": round(percentage, 1),
                "is_background": is_bg,
            }
        )
    palette.sort(key=lambda x: x["count"], reverse=True)
    return palette


# ---------------------------------------------------------------------------
# Shape resolution
# ---------------------------------------------------------------------------


def resolve_shape_spec(shape_id) -> ShapeSpec:
    if not shape_id:
        return ShapeSpec()
    from shapes.models import BeadShape

    try:
        shape = BeadShape.objects.get(pk=shape_id)
    except BeadShape.DoesNotExist:
        logger.warning("Shape %s introuvable, dimensions par defaut", shape_id)
        return ShapeSpec()

    spec = ShapeSpec(shape_found=True)
    if shape.shape_type == "rectangle" and shape.width and shape.height:
        spec.grid_width = shape.width
        spec.grid_height = shape.height
    elif shape.shape_type == "square" and shape.size:
        spec.grid_width = shape.size
        spec.grid_height = shape.size
    elif shape.shape_type == "circle" and shape.diameter:
        spec.grid_width = shape.diameter
        spec.grid_height = shape.diameter
        spec.use_circle_mask = True
        spec.circle_diameter = shape.diameter
    return spec


# ---------------------------------------------------------------------------
# Internal image helpers
# ---------------------------------------------------------------------------


def _load_image(image_path: str | None, image_base64: str | None) -> Image.Image | None:
    if image_path:
        with default_storage.open(image_path, "rb") as f:
            raw_bytes = f.read()
        img = Image.open(io.BytesIO(raw_bytes))
    elif image_base64:
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    else:
        return None
    return img.convert("RGB") if img.mode != "RGB" else img


# ---------------------------------------------------------------------------
# Saturation boost (keeps colors vivid after aggressive downsampling)
# ---------------------------------------------------------------------------


def _boost_saturation(img: Image.Image, factor: float = 1.2) -> Image.Image:
    """Boost colour saturation by *factor* to counteract muting during resize."""
    from PIL import ImageEnhance

    return ImageEnhance.Color(img).enhance(factor)


# ---------------------------------------------------------------------------
# Dark-pixel consolidation
# ---------------------------------------------------------------------------


def _consolidate_blacks(img_array: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Snap very dark pixels to pure black before colour quantisation.

    Anti-aliased outlines produce dark-grey pixels that KMeans may merge
    with neighbouring coloured clusters (e.g. dark-pink).  By collapsing
    everything below *threshold* brightness to ``[0, 0, 0]`` we give
    KMeans a clear, consolidated dark cluster.  This preserves thin
    structural details such as eyes, nostrils and outlines.
    """
    result = img_array.copy()
    brightness = np.mean(result.astype(np.float64), axis=2)
    result[brightness < threshold] = [0, 0, 0]
    return result


# ---------------------------------------------------------------------------
# Subject / background detection
# ---------------------------------------------------------------------------


def _detect_subject_mask(
    img_array: np.ndarray, padding_mask: np.ndarray | None = None
) -> np.ndarray:
    """Detect subject vs white background using connected-component analysis.

    Strategy:
    - Any near-white pixel (brightness > threshold) that is connected to the
      image border is considered *background*.
    - Interior white regions (eyes, teeth, etc.) that are NOT connected to the
      border remain as *subject*, preserving internal detail.

    Returns a boolean mask: True = subject pixel, False = background.
    When the image has no white border (e.g. a photo), the mask is all-True.
    """
    h, w = img_array.shape[:2]
    gray = np.mean(img_array.astype(np.float64), axis=2)

    WHITE_THRESHOLD = 235
    is_white = gray >= WHITE_THRESHOLD

    # Label connected white regions
    labeled, n_components = ndimage.label(is_white)

    # Find which components touch the image border
    border_labels: set[int] = set()
    border_labels.update(labeled[0, :].tolist())  # top row
    border_labels.update(labeled[-1, :].tolist())  # bottom row
    border_labels.update(labeled[:, 0].tolist())  # left col
    border_labels.update(labeled[:, -1].tolist())  # right col
    border_labels.discard(0)  # 0 = not a component

    # Background = any white region touching the border
    background = np.zeros((h, w), dtype=bool)
    for lbl in border_labels:
        background[labeled == lbl] = True

    subject = ~background

    # Intersect with padding mask (if image was placed inside a larger grid)
    if padding_mask is not None:
        subject &= padding_mask

    return subject


def _add_boundary_contour(
    reduced_pixels: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    """Add a thin black contour at the subject-background boundary.

    Only replaces *light* boundary pixels (brightness > threshold) with
    black.  Dark or coloured boundary pixels already provide a natural
    visual separation and are left untouched.

    This prevents white subject areas from visually merging with the white
    background while keeping the overall look clean (no oppressive black).
    """
    result = reduced_pixels.copy()
    h, w = result.shape[:2]

    # Dilate the background into the subject by 1 pixel to find border
    bg_mask = ~subject_mask
    dilated_bg = ndimage.binary_dilation(bg_mask)
    boundary = dilated_bg & subject_mask  # subject pixels adjacent to bg

    LIGHT_THRESHOLD = 180  # per-channel average
    for y in range(h):
        for x in range(w):
            if not boundary[y, x]:
                continue
            pixel = result[y, x]
            avg_brightness = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3.0
            if avg_brightness > LIGHT_THRESHOLD:
                result[y, x] = [0, 0, 0]

    return result


def _resize_for_circle(img: Image.Image, spec: ShapeSpec):
    ow, oh = img.size
    scale = spec.circle_diameter / min(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    grid_img = Image.new("RGB", (spec.grid_width, spec.grid_height), (255, 255, 255))
    img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    ox, oy = (spec.grid_width - nw) // 2, (spec.grid_height - nh) // 2
    grid_img.paste(img_resized, (ox, oy))

    cx, cy = spec.grid_width // 2, spec.grid_height // 2
    r = spec.circle_diameter // 2
    yy, xx = np.ogrid[: spec.grid_height, : spec.grid_width]
    content_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r**2
    return grid_img, content_mask


def _resize_for_rect(img: Image.Image, spec: ShapeSpec):
    ow, oh = img.size
    orig_ratio = ow / oh
    grid_ratio = spec.grid_width / spec.grid_height
    if orig_ratio > grid_ratio:
        tw, th = spec.grid_width, max(1, int(spec.grid_width / orig_ratio))
    else:
        th, tw = spec.grid_height, max(1, int(spec.grid_height * orig_ratio))
    tw = min(tw, spec.grid_width)
    th = min(th, spec.grid_height)
    ox, oy = (spec.grid_width - tw) // 2, (spec.grid_height - th) // 2

    grid_img = Image.new("RGB", (spec.grid_width, spec.grid_height), (255, 255, 255))
    grid_img.paste(img.resize((tw, th), Image.Resampling.LANCZOS), (ox, oy))

    mask = np.zeros((spec.grid_height, spec.grid_width), dtype=bool)
    mask[oy : oy + th, ox : ox + tw] = True
    return grid_img, mask


def _apply_circle_mask(grid_img: Image.Image, spec: ShapeSpec) -> Image.Image:
    arr = np.array(grid_img)
    h, w = arr.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = spec.circle_diameter // 2
    outside = ((xx - cx) ** 2 + (yy - cy) ** 2) > r**2
    arr[outside] = 255
    return Image.fromarray(arr)


def _draw_bead_grid(
    reduced_pixels: np.ndarray,
    spec: ShapeSpec,
    cell_size: int = 10,
    content_mask: np.ndarray | None = None,
) -> Image.Image:
    gh, gw = spec.grid_height, spec.grid_width
    canvas = np.full((gh * cell_size, gw * cell_size, 3), 255, dtype=np.uint8)

    if reduced_pixels.shape[:2] != (gh, gw):
        tmp = Image.fromarray(reduced_pixels.astype("uint8")).resize(
            (gw, gh), Image.Resampling.LANCZOS
        )
        reduced_pixels = np.array(tmp)

    cx, cy, r = 0, 0, 0
    if spec.use_circle_mask:
        cx, cy = gw // 2, gh // 2
        r = spec.circle_diameter // 2

    for y in range(gh):
        for x in range(gw):
            if spec.use_circle_mask:
                if (x - cx) ** 2 + (y - cy) ** 2 > r**2:
                    continue
            # Skip background pixels — leave as empty peg space
            if content_mask is not None and not content_mask[y, x]:
                continue
            color = reduced_pixels[y, x]
            y0, x0 = y * cell_size, x * cell_size
            # border
            canvas[y0, x0 : x0 + cell_size] = 200
            canvas[y0 + cell_size - 1, x0 : x0 + cell_size] = 200
            canvas[y0 : y0 + cell_size, x0] = 200
            canvas[y0 : y0 + cell_size, x0 + cell_size - 1] = 200
            # fill
            canvas[y0 + 1 : y0 + cell_size - 1, x0 + 1 : x0 + cell_size - 1] = color

    return Image.fromarray(canvas)


def _count_circle_beads(diameter: int) -> int:
    r = diameter / 2
    cx, cy = diameter // 2, diameter // 2
    count = 0
    for y in range(diameter):
        for x in range(diameter):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_preview(
    image_path: str | None = None,
    image_base64: str | None = None,
    shape_id=None,
    color_reduction: int = 16,
    use_available_colors: bool = False,
    user_bead_colors: np.ndarray | None = None,
) -> PreviewResult:
    """Generate a pixelised bead-grid preview of the uploaded image."""
    img = _load_image(image_path, image_base64)
    if img is None:
        return PreviewResult(image_base64="")

    spec = resolve_shape_spec(shape_id)
    user_colors = user_bead_colors if use_available_colors else None

    # Mild saturation boost to keep colours vivid after downsampling
    img = _boost_saturation(img, factor=1.2)

    # ── Grid path (all board types including circles) ───────────
    if spec.use_circle_mask:
        grid_img, padding_mask = _resize_for_circle(img, spec)
    else:
        grid_img, padding_mask = _resize_for_rect(img, spec)

    if spec.use_circle_mask:
        grid_img = _apply_circle_mask(grid_img, spec)

    grid_img = grid_img.resize(
        (spec.grid_width, spec.grid_height), Image.Resampling.LANCZOS
    )
    img_array = np.array(grid_img)

    # Detect subject vs white background (flood-fill from edges)
    subject_mask = _detect_subject_mask(img_array, padding_mask=padding_mask)

    # Use subject mask when meaningful background was found (> 5 % of area);
    # otherwise fall back to the plain padding mask.
    if subject_mask.sum() > 0 and subject_mask.sum() < padding_mask.sum() * 0.95:
        content_mask = subject_mask
    else:
        content_mask = padding_mask

    # Consolidate dark pixels to pure black before clustering so that
    # anti-aliased outlines form a single clear KMeans cluster.
    img_array = _consolidate_blacks(img_array)

    reduced_pixels = reduce_colors(
        img_array, color_reduction, user_colors, content_mask=content_mask
    )

    # Add black contour at subject–background transitions where the pixel
    # would otherwise be white/light (prevents shape from merging with bg).
    reduced_pixels = _add_boundary_contour(reduced_pixels, content_mask)

    reduced_pixels = cleanup_small_components(
        reduced_pixels, min_component_size=3, content_mask=content_mask
    )

    bead_img = _draw_bead_grid(reduced_pixels, spec, content_mask=content_mask)

    buf = io.BytesIO()
    bead_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return PreviewResult(
        image_base64=b64, reduced_pixels=reduced_pixels, content_mask=content_mask
    )


def generate_model(
    image_path: str | None = None,
    image_base64: str | None = None,
    shape_id=None,
    color_reduction: int = 16,
    use_available_colors: bool = False,
    user_bead_colors: np.ndarray | None = None,
) -> ModelResult:
    """Generate the final model: preview image + palette + bead count."""
    preview = generate_preview(
        image_path=image_path,
        image_base64=image_base64,
        shape_id=shape_id,
        color_reduction=color_reduction,
        use_available_colors=use_available_colors,
        user_bead_colors=user_bead_colors,
    )
    spec = resolve_shape_spec(shape_id)

    # Compute total beads from content mask (correct for circles)
    if preview.content_mask is not None:
        total_beads = int(preview.content_mask.sum())
    else:
        total_beads = spec.grid_width * spec.grid_height

    palette = compute_palette(
        reduced_pixels=preview.reduced_pixels,
        total_beads=total_beads,
        content_mask=preview.content_mask,
    )

    # Compute useful beads (total minus background)
    bg_beads = sum(c["count"] for c in palette if c.get("is_background"))
    useful_beads = total_beads - bg_beads
    fill_ratio = useful_beads / total_beads if total_beads > 0 else 1.0

    return ModelResult(
        image_base64=preview.image_base64,
        grid_width=spec.grid_width,
        grid_height=spec.grid_height,
        shape_id=shape_id,
        color_reduction=color_reduction,
        total_beads=total_beads,
        useful_beads=useful_beads,
        fill_ratio=round(fill_ratio, 3),
        palette=palette,
    )


def suggest_color_count(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> int:
    """Analyse an image and return the suggested number of colours.

    This is a lightweight call (no grid generation, no clustering of the final
    image) meant to be invoked once after upload so the wizard can pre-select
    the best radio-button value.
    """
    img = _load_image(image_path, image_base64)
    if img is None:
        return 16  # safe fallback

    # Down-sample to ~100px on longest side for speed
    max_dim = 100
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    arr = np.array(img)
    return estimate_optimal_colors(arr)


# ---------------------------------------------------------------------------
# Smart image analysis & suggestions
# ---------------------------------------------------------------------------


def _compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are edges (Canny), 0–1."""
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.astype(bool).sum()) / edges.size


def _compute_color_variance(img_array: np.ndarray) -> float:
    """Normalised colour variance (0–1 scale)."""
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab).astype(np.float64)
    var = np.mean(np.var(lab.reshape(-1, 3), axis=0))
    # Typical range 0–2000; normalise
    return min(1.0, var / 2000.0)


def _extract_dominant_colors(
    img_array: np.ndarray, k: int = 5, mask: np.ndarray | None = None
) -> list[str]:
    """Return hex strings of the *k* dominant colours via KMeans.

    When *mask* is provided, only pixels where ``mask`` is True are
    considered (typically the subject area, excluding the white
    background).  This avoids wasting a colour slot on background white.
    """
    if mask is not None:
        pixels = img_array[mask].astype(np.float64)
    else:
        pixels = img_array.reshape(-1, 3).astype(np.float64)
    sample_size = min(1000, len(pixels))
    rng = np.random.RandomState(42)
    sample = pixels[rng.choice(len(pixels), sample_size, replace=False)]
    km = KMeans(
        n_clusters=min(k, len(np.unique(sample, axis=0))),
        random_state=0,
        n_init="auto",
        max_iter=100,
    )
    km.fit(sample)
    centers = km.cluster_centers_.astype(int)
    # Sort by cluster size (largest first)
    labels, counts = np.unique(km.labels_, return_counts=True)
    order = np.argsort(-counts)
    return [
        f"#{centers[labels[i]][0]:02x}{centers[labels[i]][1]:02x}{centers[labels[i]][2]:02x}"
        for i in order
    ]


def _detect_subject_bounds(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Detect main subject bounding box using edge-based saliency.

    Returns (x, y, w, h) in pixel coordinates.
    """
    h, w = gray.shape

    # Edge detection + dilation to form connected subject regions
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Find contours and get bounding box of all significant contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h)

    # Filter out tiny contours (< 1% of image area)
    min_area = 0.01 * h * w
    significant = [c for c in contours if cv2.contourArea(c) > min_area]
    if not significant:
        significant = contours

    # Union of all bounding boxes
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for c in significant:
        bx, by, bw, bh = cv2.boundingRect(c)
        x_min = min(x_min, bx)
        y_min = min(y_min, by)
        x_max = max(x_max, bx + bw)
        y_max = max(y_max, by + bh)

    # Add small padding (5%)
    pad_x = int(0.05 * w)
    pad_y = int(0.05 * h)
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _compute_subject_circularity(gray: np.ndarray) -> float:
    """Compute how circular the main subject outline is.

    Uses the isoperimetric quotient (4π·area / perimeter²) on the
    largest detected contour.  Returns 0.0 (not circular at all)
    to 1.0 (perfect circle).
    """
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    h, w = gray.shape
    if perimeter == 0 or area < 0.01 * h * w:
        return 0.0

    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    return min(1.0, max(0.0, circularity))


def _suggest_bead_colors(img_array: np.ndarray, mask: np.ndarray | None = None) -> int:
    """Suggest optimal colour count for bead art.

    Bead art looks best with fewer, well-separated colours.  This
    function progressively increases *k* until adding another colour
    would either:
    - split two perceptually-close colours (ΔE < threshold), or
    - create a cluster too small to matter (< 3 % of pixels).

    When *mask* is provided, only subject pixels are analysed
    (background white is excluded).

    Returns an int in the range [2, 12].
    """
    if mask is not None:
        rgb = img_array[mask].astype(np.uint8)  # (N, 3)
        bgr = rgb[:, ::-1].reshape(1, -1, 3)
    else:
        bgr = img_array[:, :, ::-1].astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    pixels_lab = lab.reshape(-1, 3)

    rng = np.random.RandomState(42)
    sample_size = min(3000, len(pixels_lab))
    sample = pixels_lab[rng.choice(len(pixels_lab), sample_size, replace=False)]

    n_unique = len(np.unique(sample.astype(np.int32).reshape(-1, 3), axis=0))

    DE_THRESHOLD = 18.0  # min ΔE between cluster centres
    MIN_CLUSTER_RATIO = 0.03  # each colour must cover ≥ 3 %

    best_k = 2
    for k in range(2, 13):
        actual_k = min(k, n_unique)
        if actual_k < k:
            best_k = actual_k
            break

        km = KMeans(n_clusters=actual_k, random_state=0, n_init="auto", max_iter=100)
        km.fit(sample)
        centers = km.cluster_centers_

        # Check: are all clusters perceptually distinct?
        too_close = False
        for i in range(actual_k):
            for j in range(i + 1, actual_k):
                d = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
                if d < DE_THRESHOLD:
                    too_close = True
                    break
            if too_close:
                break

        if too_close:
            break  # k-1 was the last valid split

        # Check: are all clusters large enough?
        _, counts = np.unique(km.labels_, return_counts=True)
        if float(counts.min()) / counts.sum() < MIN_CLUSTER_RATIO:
            break

        best_k = k

    return max(2, best_k)


def _suggest_optimal_size(
    img_array: np.ndarray,
    subject_bounds: tuple[int, int, int, int],
) -> int:
    """Suggest the minimum board size that keeps the image recognisable.

    Strategy: simulate bead pixelisation at multiple scales by
    down-sampling and up-sampling the **subject region**, then measure
    perceptual quality loss (mean ΔE in CIELAB).  The "elbow" in the
    quality-vs-size curve is the optimal tradeoff.
    """
    h, w = img_array.shape[:2]
    bx, by, bw, bh = subject_bounds

    # Crop to subject
    subject = img_array[by : by + bh, bx : bx + bw]
    sh, sw = subject.shape[:2]
    if sh < 2 or sw < 2:
        return 15

    subj_bgr = subject[:, :, ::-1].astype(np.uint8)
    ref_lab = cv2.cvtColor(subj_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)

    test_sizes = [10, 15, 20, 25, 29, 35, 40, 50, 57]
    mae_scores: list[float] = []

    for size in test_sizes:
        if sw >= sh:
            nw = size
            nh = max(1, round(size * sh / sw))
        else:
            nh = size
            nw = max(1, round(size * sw / sh))

        small = cv2.resize(subject, (nw, nh), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)

        big_bgr = big[:, :, ::-1].astype(np.uint8)
        big_lab = cv2.cvtColor(big_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
        delta_e = float(np.mean(np.sqrt(np.sum((ref_lab - big_lab) ** 2, axis=2))))
        mae_scores.append(delta_e)

    scores = np.array(mae_scores)

    # If quality barely changes across sizes, image is very simple
    if scores[0] - scores[-1] < 2.0:
        return test_sizes[0]

    # Geometric elbow detection (same approach as estimate_optimal_colors)
    s_norm = np.arange(len(test_sizes), dtype=float)
    s_norm /= s_norm[-1]
    q_norm = (scores - scores[-1]) / (scores[0] - scores[-1] + 1e-9)

    p0 = np.array([s_norm[0], q_norm[0]])
    p1 = np.array([s_norm[-1], q_norm[-1]])
    line_vec = p1 - p0
    line_len = float(np.linalg.norm(line_vec))

    if line_len < 1e-9:
        return test_sizes[0]

    distances = [
        abs(float(np.cross(line_vec, p0 - np.array([s_norm[j], q_norm[j]])))) / line_len
        for j in range(len(test_sizes))
    ]

    best_idx = int(np.argmax(distances))
    return test_sizes[best_idx]


def analyze_image_suggestions(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> ImageSuggestion:
    """Analyse an uploaded image and suggest optimal bead-model parameters.

    The algorithm optimises for the best tradeoff between:

    * **Visual recognisability** — the bead model should look like the
      original when placed next to it.
    * **Bead economy** — fewer beads = simpler, cheaper build.
    * **Colour simplicity** — fewer colours = cleaner bead art.

    Shape is determined from the **subject's** actual outline (not the
    canvas).  Circle is only suggested when the subject has a genuinely
    circular silhouette (e.g. a ball, mandala, face).
    """
    img = _load_image(image_path, image_base64)
    if img is None:
        return ImageSuggestion()

    # Work on a manageable size
    max_dim = 300
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    w, h = img.size  # after possible resize

    # ---- 1) Subject detection ----
    bounds = _detect_subject_bounds(gray)
    bx, by, bw, bh = bounds
    content_area = bw * bh
    total_area = w * h
    content_ratio = content_area / total_area if total_area > 0 else 1.0

    # ---- 2) Shape suggestion based on SUBJECT outline ----
    subject_ar = bw / bh if bh > 0 else 1.0
    circularity = _compute_subject_circularity(gray)

    # Circle only for genuinely circular subjects
    if circularity > 0.70 and 0.80 <= subject_ar <= 1.25:
        suggested_shape = "circle"
    elif 0.75 <= subject_ar <= 1.33:
        suggested_shape = "square"
    else:
        suggested_shape = "rectangle"

    # ---- 2b) Subject mask for colour analysis ----
    # Exclude white background so it doesn't consume a colour slot.
    subject_mask_analysis = _detect_subject_mask(arr)
    total_px = arr.shape[0] * arr.shape[1]
    color_mask = (
        subject_mask_analysis if subject_mask_analysis.sum() < total_px * 0.95 else None
    )

    # ---- 3) Colour suggestion (biased toward simplicity) ----
    suggested_colors = _suggest_bead_colors(arr, mask=color_mask)

    # ---- 4) Size suggestion (multi-scale recognisability) ----
    suggested_size = _suggest_optimal_size(arr, bounds)

    # ---- 5) Complexity score (for UI display) ----
    edge_density = _compute_edge_density(gray)
    color_variance = _compute_color_variance(arr)
    complexity = 0.6 * edge_density + 0.4 * color_variance
    complexity = min(1.0, max(0.0, complexity))

    # ---- 6) Fill ratio ----
    if suggested_shape == "circle":
        fill_ratio = content_ratio * (math.pi / 4)
    else:
        fill_ratio = content_ratio

    # ---- 7) Dominant colours for visualisation ----
    dominant = _extract_dominant_colors(arr, k=5, mask=color_mask)

    return ImageSuggestion(
        suggested_shape=suggested_shape,
        suggested_colors=suggested_colors,
        suggested_size=suggested_size,
        complexity_score=round(complexity, 3),
        content_ratio=round(content_ratio, 3),
        aspect_ratio=round(subject_ar, 3),
        dominant_colors=dominant,
        fill_ratio=round(fill_ratio, 3),
    )
