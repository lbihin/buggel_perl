from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beadmodels.views.wizard_views import ConfigureModel

from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import render

from beadmodels.views.wizard_views import ModelCreatorWizard
from shapes.models import BeadShape


@login_required
def change_shape_hx_view(request, pk: int):
    """Handle HTMX request to change shape."""

    if not request.htmx:
        raise Http404("Cette vue est uniquement accessible via HTMX.")

    shape_obj = BeadShape.objects.filter(id=pk, creator=request.user).first()

    # Create a wizard instance properly and set request
    wz = ModelCreatorWizard()
    wz.request = request
    wizard_data = wz.get_session_data()
    wz.update_session_data({"shape_id": pk})
    configuration_step: ConfigureModel = wz.get_current_step()

    preview_image_base64 = configuration_step.generate_preview(wizard_data)

    return render(
        request,
        "beadmodels/wizard/partials/preview.html",
        {"preview_image_base64": preview_image_base64, "shape": shape_obj},
    )


@login_required
def change_max_colors_hx_view(request, color_reduction: int):
    """Handle HTMX request to change max colors."""

    if not request.htmx:
        raise Http404("Cette vue est uniquement accessible via HTMX.")

    # Create a wizard instance properly and set request
    wz = ModelCreatorWizard()
    wz.request = request
    wizard_data = wz.get_session_data()
    wz.update_session_data({"color_reduction": color_reduction})
    configuration_step: ConfigureModel = wz.get_current_step()

    preview_image_base64 = configuration_step.generate_preview(wizard_data)

    return render(
        request,
        "beadmodels/wizard/partials/preview.html",
        {"preview_image_base64": preview_image_base64},
    )
