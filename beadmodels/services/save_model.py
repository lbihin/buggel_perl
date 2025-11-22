import base64
import uuid

from django.core.files.base import ContentFile


def save_bead_model_from_wizard(user, wizard_data, form):
    """Create and persist a BeadModel from wizard data and a validated form.

    Args:
        user: Django user who owns the model
        wizard_data: dict containing `image_data` and `final_model` keys
        form: a validated BeadModelFinalizeForm (not yet saved)

    Returns:
        The saved BeadModel instance.
    """
    new_model = form.save(commit=False)
    new_model.creator = user

    image_data = wizard_data.get("image_data", {})
    final_model = wizard_data.get("final_model", {})

    # Ensure original image base64 exists when only a path is stored
    if not image_data.get("image_base64") and image_data.get("image_path"):
        try:
            from .image_processing import file_to_base64

            image_data["image_base64"] = file_to_base64(image_data.get("image_path"))
        except Exception:
            image_data["image_base64"] = None

    # Save original image if provided
    if image_data.get("image_base64"):
        image_bytes = base64.b64decode(image_data.get("image_base64"))
        image_format = image_data.get("image_format", "PNG")
        image_name = f"original_{uuid.uuid4()}.{image_format.lower()}"
        new_model.original_image.save(image_name, ContentFile(image_bytes), save=False)

    # Ensure final image base64 exists when only a path is stored
    if not final_model.get("image_base64") and final_model.get("image_path"):
        try:
            from .image_processing import file_to_base64

            final_model["image_base64"] = file_to_base64(final_model.get("image_path"))
        except Exception:
            final_model["image_base64"] = None

    # Save bead pattern image if provided
    if final_model.get("image_base64"):
        pattern_bytes = base64.b64decode(final_model.get("image_base64"))
        pattern_name = f"pattern_{uuid.uuid4()}.png"
        new_model.bead_pattern.save(
            pattern_name, ContentFile(pattern_bytes), save=False
        )

    # Persist metadata if available
    metadata = {
        "grid_width": final_model.get("grid_width", 29),
        "grid_height": final_model.get("grid_height", 29),
        "shape_id": final_model.get("shape_id"),
        "initial_color_reduction": wizard_data.get("color_reduction", 16),
        "final_color_reduction": final_model.get("color_reduction", 16),
        "total_beads": final_model.get("total_beads", 0),
        "palette": final_model.get("palette", []),
        "excluded_colors": final_model.get("excluded_colors", []),
    }

    if hasattr(new_model, "metadata"):
        new_model.metadata = metadata

    new_model.save()

    return new_model
