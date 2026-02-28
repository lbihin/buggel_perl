"""
Vues HTMX pour la mise à jour live de la prévisualisation (Step 2 du wizard).

Ces vues sont appelées par les boutons radio forme/couleur dans configure_model.html.
Elles mettent à jour la session, régénèrent la preview via le service
et renvoient le fragment HTML du preview-container.
"""

from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse
from django.template.loader import render_to_string
from django.utils.translation import gettext as _

from beadmodels.services.image_processing import generate_preview
from beadmodels.views.wizard_views import (_build_preview_kwargs,
                                           _fibonacci_color_values, _safe_int)
from beads.models import Bead

SESSION_KEY = "model_creation_wizard"


def _get_wizard_data(request) -> dict:
    return request.session.get(SESSION_KEY, {})


def _update_wizard_data(request, updates: dict):
    data = _get_wizard_data(request)
    data.update(updates)
    request.session[SESSION_KEY] = data


def _render_preview(request) -> HttpResponse:
    """Regenerate preview from current session state and return HTML fragment."""
    wizard_data = _get_wizard_data(request)
    kwargs = _build_preview_kwargs(wizard_data, request.user)
    result = generate_preview(**kwargs)
    html = render_to_string(
        "beadmodels/wizard/partials/preview.html",
        {"preview_image_base64": result.image_base64},
    )
    return HttpResponse(html)


@login_required
def change_shape_hx_view(request, pk: int):
    """Handle HTMX request to change shape."""
    if not getattr(request, "htmx", False):
        raise Http404(_("Cette vue est uniquement accessible via HTMX."))

    _update_wizard_data(request, {"shape_id": pk})
    return _render_preview(request)


@login_required
def change_max_colors_hx_view(request, color_reduction: int):
    """Handle HTMX request to change max colors."""
    if not getattr(request, "htmx", False):
        raise Http404("Cette vue est uniquement accessible via HTMX.")

    _update_wizard_data(request, {"color_reduction": color_reduction})
    return _render_preview(request)


@login_required
def toggle_use_colors_hx_view(request):
    """Handle HTMX request to toggle 'use available colors'.

    Returns the updated preview as main response, plus an OOB swap for the
    colour-chip radio buttons (since the available values change when the
    toggle is flipped).
    """
    if not getattr(request, "htmx", False):
        raise Http404(_("Cette vue est uniquement accessible via HTMX."))

    use_available = request.POST.get("use_available_colors") == "on"

    # Keep current selections from the included form fields
    wizard_data = _get_wizard_data(request)
    shape_id = request.POST.get("shape_id") or wizard_data.get("shape_id", "")
    color_reduction = _safe_int(
        request.POST.get("color_reduction"),
        _safe_int(wizard_data.get("color_reduction"), 16),
    )

    _update_wizard_data(
        request,
        {
            "shape_id": shape_id,
            "color_reduction": color_reduction,
            "use_available_colors": use_available,
        },
    )

    # Regenerate preview
    wizard_data = _get_wizard_data(request)
    kwargs = _build_preview_kwargs(wizard_data, request.user)
    result = generate_preview(**kwargs)

    preview_html = render_to_string(
        "beadmodels/wizard/partials/preview.html",
        {"preview_image_base64": result.image_base64},
    )

    # Recalculate colour chips
    suggestions = wizard_data.get("suggestions", {})
    suggested_colors = suggestions.get("suggested_colors") or wizard_data.get(
        "suggested_colors", 16
    )
    color_values = _fibonacci_color_values(suggested_colors)
    user_bead_count = Bead.objects.filter(creator=request.user).count()

    if use_available and user_bead_count:
        color_values = [v for v in color_values if v <= user_bead_count]
        if not color_values:
            color_values = [user_bead_count]

    suggested_snapped = min(color_values, key=lambda v: abs(v - suggested_colors))

    # If current selection is outside the new range, snap to closest
    if color_reduction not in color_values:
        color_reduction = min(color_values, key=lambda v: abs(v - color_reduction))
        _update_wizard_data(request, {"color_reduction": color_reduction})

    chips_html = render_to_string(
        "beadmodels/wizard/partials/color_chips.html",
        {
            "color_values": color_values,
            "suggested_colors": suggested_snapped,
            "selected_color": color_reduction,
        },
        request=request,
    )

    # Return preview as main + colour chips as OOB swap
    oob_chips = chips_html.replace(
        'id="color-chips-container"',
        'id="color-chips-container" hx-swap-oob="true"',
        1,
    )
    return HttpResponse(preview_html + oob_chips)
