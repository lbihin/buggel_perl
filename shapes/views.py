"""
Views pour l'application shapes.

Ce module contient les vues pour gérer les formes de perles (BeadShape).
Les vues principales utilisent les CBVs Django, tandis que les petites
vues HTMX d'édition inline restent des FBVs.
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.generic import ListView

from .forms import BeadShapeForm
from .models import BeadShape

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_shape_permission(request, shape, action="modifier"):
    """Vérifie que l'utilisateur a le droit de modifier/supprimer la forme.

    Retourne une HttpResponse si non autorisé, None sinon.
    """
    if shape.creator != request.user and not shape.is_default:
        msg = _("Vous n'avez pas l'autorisation de %(action)s cette forme") % {
            "action": action
        }
        if getattr(request, "htmx", False):
            return HttpResponse(msg, status=403)
        request.session["error_message"] = msg
        return redirect(reverse("shapes:shape_list_columns"))
    return None


def _require_htmx(request):
    """Lève Http404 si la requête n'est pas HTMX."""
    if not getattr(request, "htmx", False):
        raise Http404(_("Cette vue est uniquement accessible via HTMX."))


def _get_shapes_context(user):
    """Retourne le contexte standard pour la liste des formes."""
    custom_shapes = BeadShape.objects.filter(creator=user, is_default=False)
    default_shapes = BeadShape.objects.filter(is_default=True)
    shared_shapes = (
        BeadShape.objects.filter(is_shared=True)
        .exclude(is_default=True)
        .exclude(creator=user)
    )
    return {
        "default_shapes": default_shapes,
        "custom_shapes": custom_shapes,
        "shared_shapes": shared_shapes,
        "shapes": custom_shapes,
    }


# ---------------------------------------------------------------------------
# Vues principales (CBVs)
# ---------------------------------------------------------------------------


class ShapeListView(LoginRequiredMixin, ListView):
    """Redirige vers la vue en colonnes."""

    def get(self, request, *args, **kwargs):
        return redirect("shapes:shape_list_columns")


class ShapeListColumnsView(LoginRequiredMixin, ListView):
    """Affiche la liste des plaques à picots en tableau simple."""

    template_name = "shapes/shape_list_columns.html"
    context_object_name = "user_shapes"

    def get_queryset(self):
        return BeadShape.objects.filter(
            Q(creator=self.request.user) | Q(is_default=True) | Q(is_shared=True)
        ).select_related("creator")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["all_shapes"] = context["user_shapes"]

        # Messages flash stockés en session (pour compatibilité HTMX)
        context["success_message"] = self.request.session.pop("success_message", None)
        context["error_message"] = self.request.session.pop("error_message", None)

        return context


# ---------------------------------------------------------------------------
# Vues CRUD (form processing)
# ---------------------------------------------------------------------------


@login_required
def create_shape(request):
    """Crée une nouvelle forme via le formulaire de la vue colonnes."""
    if request.method != "POST":
        return redirect("shapes:shape_list_columns")

    form = BeadShapeForm(request.POST)
    if form.is_valid():
        shape = form.save(commit=False)
        shape.creator = request.user
        shape.update_from_dimensions()
        shape.save()

        msg = _("Forme créée avec succès!")
        if getattr(request, "htmx", False):
            request.session["success_message"] = msg
        else:
            messages.success(request, msg)
    else:
        msg = _("Erreur lors de la création de la forme.")
        if getattr(request, "htmx", False):
            request.session["error_message"] = msg
        else:
            messages.error(request, msg)

    return redirect("shapes:shape_list_columns")


@login_required
def update_shape(request, shape_id):
    """Met à jour une forme existante via le formulaire de la vue colonnes."""
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response

    if request.method != "POST":
        return redirect("shapes:shape_list_columns")

    form = BeadShapeForm(request.POST, instance=shape)
    if form.is_valid():
        shape = form.save(commit=False)
        shape.update_from_dimensions()
        shape.save()

        msg = _("Forme mise à jour avec succès!")
        request.session["success_message"] = msg
    else:
        msg = _("Erreur lors de la mise à jour de la forme.")
        request.session["error_message"] = msg

    return redirect("shapes:shape_list_columns")


@login_required
def delete_shape(request, shape_id):
    """Supprime une forme."""
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape, action="supprimer")
    if error_response:
        return error_response

    if shape.is_default:
        msg = _("Les formes par défaut ne peuvent pas être supprimées")
        request.session["error_message"] = msg
        return redirect("shapes:shape_list_columns")

    shape_name = shape.name
    shape.delete()

    msg = _("Forme %(name)s supprimée avec succès!") % {"name": shape_name}
    request.session["success_message"] = msg

    return redirect("shapes:shape_list_columns")


# ---------------------------------------------------------------------------
# Vues HTMX — Formulaires inline
# ---------------------------------------------------------------------------


@login_required
def shape_form_hx_view(request, shape_id=None):
    """Vue HTMX pour afficher le formulaire de création/édition."""
    _require_htmx(request)

    instance = None
    if shape_id:
        instance = get_object_or_404(BeadShape, id=shape_id)
        error_response = _check_shape_permission(request, instance)
        if error_response:
            return error_response

    url = reverse("shapes:shape_create_hx")
    if instance:
        url = reverse("shapes:shape_save_edit", kwargs={"shape_id": instance.id})

    return render(
        request,
        "shapes/partials/create_edit_shape.html",
        {"url": url, "shape": instance},
    )


@login_required
def shape_save_or_update_hx_view(request, shape_id=None):
    """Vue HTMX unifiée pour la création ou mise à jour d'une forme."""
    _require_htmx(request)

    if request.method != "POST":
        return HttpResponse(_("Méthode non autorisée"), status=405)

    is_new = shape_id is None
    instance = None

    if not is_new:
        instance = get_object_or_404(BeadShape, id=shape_id)
        error_response = _check_shape_permission(request, instance)
        if error_response:
            return error_response

    form = BeadShapeForm(request.POST, instance=instance)
    if not form.is_valid():
        errors = "; ".join(
            e for field_errors in form.errors.values() for e in field_errors
        )
        return HttpResponse(_("Erreur : %(errors)s") % {"errors": errors}, status=400)

    shape = form.save(commit=False)
    if is_new:
        shape.creator = request.user
    shape.update_from_dimensions()
    shape.save()

    if is_new:
        context = _get_shapes_context(request.user)
        context["success_message"] = _("Forme %(name)s créée avec succès!") % {
            "name": shape.name
        }
        return render(request, "shapes/shape_row_list.html", context)
    else:
        return render(
            request,
            "shapes/partials/shape_row.html",
            {
                "shape": shape,
                "success": True,
                "message": _("Forme %(name)s mise à jour avec succès!")
                % {"name": shape.name},
            },
        )


@login_required
def shape_delete_hx_view(request, shape_id):
    """Vue HTMX pour supprimer une forme."""
    _require_htmx(request)

    shape = get_object_or_404(BeadShape, id=shape_id)

    if shape.creator != request.user:
        return HttpResponse(
            _("Vous n'avez pas l'autorisation de supprimer cette forme"), status=403
        )

    if shape.is_default:
        return HttpResponse(
            _("Les formes par défaut ne peuvent pas être supprimées"), status=403
        )

    shape_name = shape.name
    shape.delete()

    context = _get_shapes_context(request.user)
    context["success_message"] = _("Forme %(name)s supprimée avec succès!") % {
        "name": shape_name
    }

    return render(request, "shapes/partials/shape_list_content.html", context)


@login_required
def get_inline_add_form(request):
    """Vue HTMX pour obtenir le formulaire d'ajout en ligne."""
    _require_htmx(request)
    return render(request, "shapes/partials/shape_inline_add_form.html")


# ---------------------------------------------------------------------------
# Vues HTMX — Édition inline des dimensions
# ---------------------------------------------------------------------------


@login_required
def shape_dimensions(request, shape_id):
    """Affiche les dimensions d'une forme."""
    _require_htmx(request)
    shape = get_object_or_404(BeadShape, id=shape_id)
    return render(request, "shapes/partials/shape_dimensions.html", {"shape": shape})


@login_required
def shape_inline_edit(request, shape_id):
    """Affiche l'éditeur inline des dimensions."""
    _require_htmx(request)
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response

    type_changed = request.GET.get("type_changed", False)
    return render(
        request,
        "shapes/partials/shape_inline_edit.html",
        {"shape": shape, "type_changed": type_changed},
    )


@login_required
def shape_inline_update(request, shape_id):
    """Met à jour les dimensions via l'éditeur inline."""
    _require_htmx(request)

    if request.method != "POST":
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response

    try:
        shape_type = shape.shape_type

        if shape_type == "rectangle":
            shape.width = int(request.POST.get("width", 0))
            shape.height = int(request.POST.get("height", 0))
        elif shape_type == "square":
            shape.size = int(request.POST.get("size", 0))
        elif shape_type == "circle":
            shape.diameter = int(request.POST.get("diameter", 0))

        shape.update_from_dimensions()
        shape.save()

        response = render(
            request, "shapes/partials/shape_dimensions.html", {"shape": shape}
        )
        response["HX-Trigger"] = (
            f'{{"shapeDimensionsUpdated": {{"shapeId": "{shape.id}"}}}}'
        )
        return response

    except (ValueError, TypeError) as e:
        return HttpResponse(
            _("Erreur lors de la mise à jour des dimensions : %(error)s")
            % {"error": e},
            status=400,
        )


# ---------------------------------------------------------------------------
# Vues HTMX — Édition inline du nom
# ---------------------------------------------------------------------------


@login_required
def shape_name(request, shape_id):
    """Affiche le nom d'une forme."""
    _require_htmx(request)
    shape = get_object_or_404(BeadShape, id=shape_id)
    return render(request, "shapes/partials/shape_name.html", {"shape": shape})


@login_required
def shape_name_edit(request, shape_id):
    """Affiche l'éditeur inline du nom."""
    _require_htmx(request)
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response
    return render(request, "shapes/partials/shape_name_edit.html", {"shape": shape})


@login_required
def shape_name_update(request, shape_id):
    """Met à jour le nom via l'éditeur inline."""
    _require_htmx(request)

    if request.method != "POST":
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response

    try:
        shape.name = request.POST.get("name", "")
        shape.save()
        return render(request, "shapes/partials/shape_name.html", {"shape": shape})
    except Exception as e:
        return HttpResponse(
            _("Erreur lors de la mise à jour du nom : %(error)s") % {"error": e},
            status=400,
        )


# ---------------------------------------------------------------------------
# Vues HTMX — Édition inline du type
# ---------------------------------------------------------------------------


@login_required
def shape_type(request, shape_id):
    """Affiche le type d'une forme."""
    _require_htmx(request)
    shape = get_object_or_404(BeadShape, id=shape_id)
    return render(request, "shapes/partials/shape_type.html", {"shape": shape})


@login_required
def shape_type_edit(request, shape_id):
    """Affiche l'éditeur inline du type."""
    _require_htmx(request)
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response
    return render(request, "shapes/partials/shape_type_edit.html", {"shape": shape})


@login_required
def shape_type_update(request, shape_id):
    """Met à jour le type via l'éditeur inline."""
    _require_htmx(request)

    if request.method != "POST":
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response

    try:
        old_type = shape.shape_type
        new_type = request.POST.get("shape_type", "")
        shape.shape_type = new_type

        # Réinitialiser les dimensions non pertinentes
        if new_type == "rectangle":
            if old_type != "rectangle":
                shape.width = 10
                shape.height = 10
            shape.size = None
            shape.diameter = None
        elif new_type == "square":
            if old_type != "square":
                shape.size = 10
            shape.width = None
            shape.height = None
            shape.diameter = None
        elif new_type == "circle":
            if old_type != "circle":
                shape.diameter = 10
            shape.width = None
            shape.height = None
            shape.size = None

        shape.save()

        response = render(request, "shapes/partials/shape_type.html", {"shape": shape})
        if old_type != new_type:
            response["HX-Trigger"] = (
                f'{{"shapeTypeChanged": {{"shapeId": "{shape.id}"}}}}'
            )
        return response

    except Exception as e:
        return HttpResponse(
            _("Erreur lors de la mise à jour du type : %(error)s") % {"error": e},
            status=400,
        )


# ---------------------------------------------------------------------------
# Vues HTMX — Row-level inline editing (new table approach)
# ---------------------------------------------------------------------------


@login_required
def shape_edit_row_htmx(request, shape_id):
    """GET: returns edit row; GET?cancel=true: returns display row."""
    shape = get_object_or_404(BeadShape, id=shape_id)
    if request.GET.get("cancel"):
        return render(
            request, "shapes/partials/shape_row_display.html", {"shape": shape}
        )
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response
    return render(request, "shapes/partials/shape_row_edit.html", {"shape": shape})


@login_required
def shape_save_row_htmx(request, shape_id):
    """POST: saves all fields, returns display row."""
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape)
    if error_response:
        return error_response

    if request.method != "POST":
        return shape_edit_row_htmx(request, shape_id)

    name = request.POST.get("name", "").strip()
    if name:
        shape.name = name

    shape_type = request.POST.get("shape_type", shape.shape_type)
    shape.shape_type = shape_type

    try:
        if shape_type == "rectangle":
            shape.width = int(request.POST.get("width") or 10)
            shape.height = int(request.POST.get("height") or 10)
            shape.size = None
            shape.diameter = None
        elif shape_type == "square":
            shape.size = int(request.POST.get("size") or 10)
            shape.width = None
            shape.height = None
            shape.diameter = None
        elif shape_type == "circle":
            shape.diameter = int(request.POST.get("diameter") or 10)
            shape.width = None
            shape.height = None
            shape.size = None

        shape.save()
    except (ValueError, TypeError):
        return render(
            request,
            "shapes/partials/shape_row_edit.html",
            {"shape": shape, "error": _("Dimensions invalides.")},
        )

    return render(request, "shapes/partials/shape_row_display.html", {"shape": shape})


@login_required
def shape_new_row_htmx(request):
    """GET: returns an empty new row for inline creation."""
    return render(request, "shapes/partials/shape_row_new.html")


@login_required
def shape_create_inline_htmx(request):
    """POST: creates a new shape inline, returns display row."""
    if request.method != "POST":
        return shape_new_row_htmx(request)

    name = request.POST.get("name", "").strip()
    shape_type = request.POST.get("shape_type", "rectangle")

    try:
        width = height = size = diameter = None
        if shape_type == "rectangle":
            width = int(request.POST.get("width") or 10)
            height = int(request.POST.get("height") or 10)
        elif shape_type == "square":
            size = int(request.POST.get("size") or 10)
        elif shape_type == "circle":
            diameter = int(request.POST.get("diameter") or 10)

        if not name:
            dims = ""
            if shape_type == "rectangle":
                dims = f"{width}×{height}"
            elif shape_type == "square":
                dims = f"{size}×{size}"
            elif shape_type == "circle":
                dims = f"∅{diameter}"
            name = _("Plaque %(type)s %(dims)s") % {
                "type": dict(BeadShape.SHAPE_TYPES).get(shape_type, ""),
                "dims": dims,
            }

        shape = BeadShape.objects.create(
            creator=request.user,
            name=name,
            shape_type=shape_type,
            width=width,
            height=height,
            size=size,
            diameter=diameter,
        )
    except (ValueError, TypeError):
        return render(
            request,
            "shapes/partials/shape_row_new.html",
            {
                "error": _("Dimensions invalides."),
                "name": name,
                "shape_type": shape_type,
            },
        )

    return render(request, "shapes/partials/shape_row_display.html", {"shape": shape})


@login_required
def shape_delete_row_htmx(request, shape_id):
    """DELETE: deletes shape, returns empty to remove the row."""
    shape = get_object_or_404(BeadShape, id=shape_id)
    error_response = _check_shape_permission(request, shape, action="supprimer")
    if error_response:
        return error_response

    if shape.is_default:
        return HttpResponse(
            _("Les plaques par défaut ne peuvent pas être supprimées"), status=403
        )

    shape.delete()
    return HttpResponse("")
