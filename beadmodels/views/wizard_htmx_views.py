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
from beadmodels.views.wizard_views import _build_preview_kwargs

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
