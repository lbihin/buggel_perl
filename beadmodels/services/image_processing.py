"""
Service de traitement d'image pour le wizard.

Fonctions pures : pas de dependance a request/messages.
"""

import base64
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import cv2
import numpy as np
from django.core.files.storage import default_storage
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreviewResult:
    image_base64: str
    reduced_pixels: Optional[np.ndarray] = None
    content_mask: Optional[np.ndarray] = None


@dataclass
class ShapeSpec:
    grid_width: int = 29
    grid_height: int = 29
    use_circle_mask: bool = False
    circle_diameter: int = 0
    shape_found: bool = False


@dataclass
class ModelResult:
    image_base64: str = ""
    grid_width: int = 29
    grid_height: int = 29
    shape_id: Optional[str] = None
    color_reduction: int = 16
    total_beads: int = 0
    palette: List[dict] = field(default_factory=list)


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


def reduce_colors(
    image_array: np.ndarray,
    n_colors: int,
    user_colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_.astype(int)

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
    content_mask: Optional[np.ndarray] = None,
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


def _detect_background_color(
    reduced_pixels: np.ndarray,
    content_mask: Optional[np.ndarray] = None,
) -> Optional[tuple]:
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
    reduced_pixels: Optional[np.ndarray] = None,
    content_mask: Optional[np.ndarray] = None,
    image_base64: Optional[str] = None,
) -> List[dict]:
    if reduced_pixels is not None:
        if content_mask is not None:
            pixels = reduced_pixels[content_mask]
        else:
            pixels = reduced_pixels.reshape(-1, 3)
    elif image_base64:
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pixels = np.array(img).reshape(-1, 3)
    else:
        return []

    # Detect background
    bg_color = None
    if reduced_pixels is not None:
        bg_color = _detect_background_color(reduced_pixels, content_mask)

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


def _load_image(
    image_path: Optional[str], image_base64: Optional[str]
) -> Optional[Image.Image]:
    if image_path:
        with default_storage.open(image_path, "rb") as f:
            raw_bytes = f.read()
        img = Image.open(io.BytesIO(raw_bytes))
    elif image_base64:
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    else:
        return None
    return img.convert("RGB") if img.mode != "RGB" else img


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
    reduced_pixels: np.ndarray, spec: ShapeSpec, cell_size: int = 10
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
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    shape_id=None,
    color_reduction: int = 16,
    use_available_colors: bool = False,
    user_bead_colors: Optional[np.ndarray] = None,
) -> PreviewResult:
    """Generate a pixelised bead-grid preview of the uploaded image."""
    img = _load_image(image_path, image_base64)
    if img is None:
        return PreviewResult(image_base64="")

    spec = resolve_shape_spec(shape_id)

    # Resize
    if spec.use_circle_mask:
        grid_img, content_mask = _resize_for_circle(img, spec)
    else:
        grid_img, content_mask = _resize_for_rect(img, spec)

    # Circle mask
    if spec.use_circle_mask:
        grid_img = _apply_circle_mask(grid_img, spec)

    # Ensure exact grid size for clustering
    grid_img = grid_img.resize(
        (spec.grid_width, spec.grid_height), Image.Resampling.LANCZOS
    )
    img_array = np.array(grid_img)

    # Color reduction
    user_colors = user_bead_colors if use_available_colors else None
    reduced_pixels = reduce_colors(img_array, color_reduction, user_colors)

    # Spatial coherence: remove isolated beads / tiny fragments
    reduced_pixels = cleanup_small_components(
        reduced_pixels, min_component_size=3, content_mask=content_mask
    )

    # Draw bead grid
    bead_img = _draw_bead_grid(reduced_pixels, spec)

    # Encode
    buf = io.BytesIO()
    bead_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return PreviewResult(
        image_base64=b64, reduced_pixels=reduced_pixels, content_mask=content_mask
    )


def generate_model(
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    shape_id=None,
    color_reduction: int = 16,
    use_available_colors: bool = False,
    user_bead_colors: Optional[np.ndarray] = None,
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

    if spec.use_circle_mask and spec.circle_diameter:
        total_beads = _count_circle_beads(spec.circle_diameter)
    else:
        total_beads = spec.grid_width * spec.grid_height

    palette = compute_palette(
        reduced_pixels=preview.reduced_pixels,
        total_beads=total_beads,
        content_mask=preview.content_mask,
    )

    return ModelResult(
        image_base64=preview.image_base64,
        grid_width=spec.grid_width,
        grid_height=spec.grid_height,
        shape_id=shape_id,
        color_reduction=color_reduction,
        total_beads=total_beads,
        palette=palette,
    )
