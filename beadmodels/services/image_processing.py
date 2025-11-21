import os
import time
import uuid
from typing import Iterable

from django.core.files.storage import default_storage


def save_temp_image(uploaded_file) -> str:
    """Persist uploaded image to a temporary media location and return its storage path.
    Uses uuid for uniqueness. Returns relative storage path usable with default_storage.open().
    """
    ext = (uploaded_file.name.split(".")[-1] or "png").lower()
    filename = f"temp_wizard/{uuid.uuid4()}.{ext}"
    return default_storage.save(filename, uploaded_file)


def cleanup_temp_images(max_age_seconds: int = 3600) -> Iterable[str]:
    """Delete temp wizard images older than max_age_seconds.

    Returns an iterable of deleted paths for optional logging.
    Safe: ignores errors if file was already removed.
    """
    base_dir = "temp_wizard"
    deleted = []
    try:
        if not default_storage.exists(base_dir):
            return deleted
        # List files in temp_wizard
        _, files = default_storage.listdir(base_dir)
        for entry in files:  # files only
            rel_path = f"{base_dir}/{entry}"
            try:
                # Get modification time via storage path
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
    """Load a stored file path via default_storage and return base64 PNG string."""
    if not path:
        return ""
    with default_storage.open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


import base64
import io
from typing import List, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def reduce_colors(
    image_array: np.ndarray, n_colors: int, user_colors: Optional[np.ndarray] = None
) -> np.ndarray:
    """Reduce colors of an RGB image array using KMeans and optionally map
    cluster centroids to nearest user-provided colors.

    Args:
        image_array: H×W×3 uint8 array.
        n_colors: Number of color clusters.
        user_colors: Optional N×3 array of RGB bead colors.
    Returns:
        Reduced H×W×3 uint8 array.
    """
    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_.astype(int)

    if user_colors is not None and len(user_colors):
        # Map each centroid to closest user color
        for i, c in enumerate(centroids):
            distances = np.sqrt(np.sum((user_colors - c) ** 2, axis=1))
            centroids[i] = user_colors[int(np.argmin(distances))]

    labels = kmeans.labels_
    reduced_pixels = centroids[labels].reshape(image_array.shape)
    return reduced_pixels.astype("uint8")


def compute_palette(
    image_base64: Optional[str] = None,
    total_beads: int = 0,
    reduced_pixels: Optional[np.ndarray] = None,
    content_mask: Optional[np.ndarray] = None,
) -> List[dict]:
    """Compute palette (unique colors with counts and percentages) from reduced pixel array or base64 image.

    Args:
        image_base64: Base64-encoded PNG image string (optional if reduced_pixels provided).
        total_beads: Total bead count for percentage calculation.
        reduced_pixels: H×W×3 numpy array of reduced colors (preferred source).
        content_mask: H×W boolean array indicating which pixels are actual content (True) vs padding (False).
    Returns:
        Sorted list of palette entries dict(color, hex, count, percentage).
    """
    if reduced_pixels is not None:
        # Use reduced pixels directly (excludes grid borders)
        if content_mask is not None:
            # Filter pixels to only count content areas (exclude white padding)
            pixels = reduced_pixels[content_mask]
        else:
            pixels = reduced_pixels.reshape(-1, 3)
    elif image_base64:
        # Fallback to base64 image
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        pixels = arr.reshape(-1, 3)
    else:
        return []

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    palette = []
    safe_total = total_beads if total_beads > 0 else pixels.shape[0]
    for color, count in zip(unique_colors, counts):
        r, g, b = color
        percentage = (count / safe_total) * 100
        palette.append(
            {
                "color": f"rgb({r}, {g}, {b})",
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "count": int(count),
                "percentage": round(percentage, 1),
            }
        )
    palette.sort(key=lambda x: x["count"], reverse=True)
    return palette
