from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from .forms import BeadShapeForm
from .models import BeadShape


@login_required
def shape_list(request):
    """Vue pour afficher la liste des formes de l'utilisateur"""
    # Récupération des paramètres de filtrage et recherche
    search_term = request.GET.get("search", "")
    shape_type_filter = request.GET.get("shape_type_filter", "")

    # Filtrer les formes de l'utilisateur
    shapes = BeadShape.objects.filter(creator=request.user)

    # Appliquer le filtre de recherche si présent
    if search_term:
        shapes = shapes.filter(
            Q(name__icontains=search_term) | Q(id__icontains=search_term)
        )

    # Appliquer le filtre de type si présent
    if shape_type_filter:
        shapes = shapes.filter(shape_type=shape_type_filter)

    # Tri par défaut
    shapes = shapes.order_by("id")

    # Pagination
    page = request.GET.get("page", 1)
    paginator = Paginator(shapes, 10)  # 10 formes par page
    shapes = paginator.get_page(page)

    # Si c'est une requête HTMX, rendre seulement le fragment de la liste
    if request.headers.get("HX-Request"):
        return render(
            request,
            "beadmodels/shapes/shape_row_list.html",
            {
                "shapes": shapes,
                "search": search_term,
                "shape_type_filter": shape_type_filter,
            },
        )

    # Sinon, rendre la page complète
    return render(
        request,
        "beadmodels/shapes/shape_list.html",
        {
            "shapes": shapes,
            "search": search_term,
            "shape_type_filter": shape_type_filter,
        },
    )


@login_required
def shape_form(request, shape_id=None):
    """Vue pour afficher le formulaire d'ajout/modification d'une forme"""
    shape = None
    if shape_id:
        shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
        form = BeadShapeForm(instance=shape)
    else:
        form = BeadShapeForm()

    return render(
        request,
        "beadmodels/shapes/shape_form_htmx.html",
        {"form": form, "shape": shape},
    )


@login_required
@require_POST
def shape_save(request, shape_id=None):
    """Vue pour sauvegarder une forme (création ou mise à jour)"""
    if shape_id:
        shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
        form = BeadShapeForm(request.POST, instance=shape)
    else:
        form = BeadShapeForm(request.POST)

    if form.is_valid():
        # Vérifier si une forme avec le même nom existe déjà pour cet utilisateur
        shape_name = form.cleaned_data.get("name")
        shape_type = form.cleaned_data.get("shape_type")

        # Pour une nouvelle forme ou changement de nom
        if not shape_id or (shape_id and shape.name != shape_name):
            existing_shape = BeadShape.objects.filter(
                name=shape_name, creator=request.user
            ).first()

            if existing_shape:
                form.add_error(
                    "name",
                    f"Une forme avec le nom '{shape_name}' existe déjà. Veuillez choisir un autre nom.",
                )
                return render(
                    request,
                    "beadmodels/shapes/shape_form_htmx.html",
                    {"form": form, "shape": shape if shape_id else None},
                    status=422,
                )

        # Vérifier les dimensions pour les formes similaires
        duplicate_params = False
        message = ""
        existing_shapes = BeadShape.objects.filter(
            creator=request.user, shape_type=shape_type
        )
        if shape_id:
            existing_shapes = existing_shapes.exclude(id=shape_id)

        if shape_type == "rectangle":
            width = form.cleaned_data.get("width")
            height = form.cleaned_data.get("height")
            for existing_shape in existing_shapes:
                if existing_shape.width == width and existing_shape.height == height:
                    duplicate_params = True
                    message = "Une forme rectangulaire avec ces dimensions existe déjà"
                    break
        elif shape_type == "square":
            size = form.cleaned_data.get("size")
            for existing_shape in existing_shapes:
                if existing_shape.size == size:
                    duplicate_params = True
                    message = "Un carré avec cette taille existe déjà"
                    break
        elif shape_type == "circle":
            diameter = form.cleaned_data.get("diameter")
            for existing_shape in existing_shapes:
                if existing_shape.diameter == diameter:
                    duplicate_params = True
                    message = "Un cercle avec ce diamètre existe déjà"
                    break

        if duplicate_params:
            form.add_error("shape_type", message)
            return render(
                request,
                "beadmodels/shapes/shape_form_htmx.html",
                {"form": form, "shape": shape if shape_id else None},
                status=422,
            )

        # Tout est bon, on sauvegarde
        shape_instance = form.save(commit=False)
        if not shape_id:
            shape_instance.creator = request.user
            shape_instance.is_shared = True
        shape_instance.save()

        # Récupérer toutes les formes pour afficher la liste mise à jour (avec pagination)
        shapes = BeadShape.objects.filter(creator=request.user).order_by("id")
        paginator = Paginator(shapes, 10)  # 10 formes par page
        shapes = paginator.get_page(1)  # Première page après sauvegarde

        success_message = (
            "Forme modifiée avec succès" if shape_id else "Forme créée avec succès"
        )

        # Retourner le template partiel avec la liste mise à jour
        return render(
            request,
            "beadmodels/shapes/shape_row_list.html",
            {"shapes": shapes, "success_message": success_message},
        )

    # En cas d'erreur dans le formulaire
    return render(
        request,
        "beadmodels/shapes/shape_form_htmx.html",
        {"form": form, "shape": shape if shape_id else None},
        status=422,
    )


@login_required
@require_POST
def shape_delete(request, shape_id):
    """Vue pour supprimer une forme"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    shape_name = shape.name
    shape.delete()

    # Récupérer toutes les formes pour afficher la liste mise à jour (avec pagination)
    shapes = BeadShape.objects.filter(creator=request.user).order_by("id")
    paginator = Paginator(shapes, 10)  # 10 formes par page
    shapes = paginator.get_page(1)  # Première page après suppression

    # Retourner le template partiel avec la liste mise à jour
    return render(
        request,
        "beadmodels/shapes/shape_row_list.html",
        {
            "shapes": shapes,
            "success_message": f"Forme '{shape_name}' supprimée avec succès",
        },
    )


@login_required
def shape_detail(request, shape_id):
    """Vue pour afficher les détails d'une forme"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_detail.html", {"shape": shape})


# Vues pour l'édition en ligne des dimensions
@login_required
def shape_inline_edit(request, shape_id):
    """Vue pour afficher le formulaire d'édition en ligne des dimensions"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_inline_edit.html", {"shape": shape})


@login_required
def shape_dimensions(request, shape_id):
    """Vue pour afficher les dimensions d'une forme"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_dimensions.html", {"shape": shape})


@login_required
@require_POST
def shape_inline_update(request, shape_id):
    """Vue pour mettre à jour les dimensions d'une forme en ligne"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)

    # Récupérer les données selon le type de forme
    if shape.shape_type == "rectangle":
        try:
            width = int(request.POST.get("width", 0))
            height = int(request.POST.get("height", 0))

            if width < 1 or height < 1:
                return HttpResponse(
                    "Les dimensions doivent être supérieures à 0", status=400
                )

            # Vérifier s'il existe déjà une forme avec ces dimensions
            existing = (
                BeadShape.objects.filter(
                    creator=request.user,
                    shape_type="rectangle",
                    width=width,
                    height=height,
                )
                .exclude(id=shape_id)
                .first()
            )

            if existing:
                return HttpResponse(
                    "Une forme avec ces dimensions existe déjà", status=400
                )

            shape.width = width
            shape.height = height
            shape.save()

        except ValueError:
            return HttpResponse("Veuillez entrer des nombres valides", status=400)

    elif shape.shape_type == "square":
        try:
            size = int(request.POST.get("size", 0))

            if size < 1:
                return HttpResponse("La taille doit être supérieure à 0", status=400)

            # Vérifier s'il existe déjà une forme avec cette taille
            existing = (
                BeadShape.objects.filter(
                    creator=request.user, shape_type="square", size=size
                )
                .exclude(id=shape_id)
                .first()
            )

            if existing:
                return HttpResponse(
                    "Une forme avec cette taille existe déjà", status=400
                )

            shape.size = size
            shape.save()

        except ValueError:
            return HttpResponse("Veuillez entrer un nombre valide", status=400)

    elif shape.shape_type == "circle":
        try:
            diameter = int(request.POST.get("diameter", 0))

            if diameter < 1:
                return HttpResponse("Le diamètre doit être supérieur à 0", status=400)

            # Vérifier s'il existe déjà une forme avec ce diamètre
            existing = (
                BeadShape.objects.filter(
                    creator=request.user, shape_type="circle", diameter=diameter
                )
                .exclude(id=shape_id)
                .first()
            )

            if existing:
                return HttpResponse(
                    "Une forme avec ce diamètre existe déjà", status=400
                )

            shape.diameter = diameter
            shape.save()

        except ValueError:
            return HttpResponse("Veuillez entrer un nombre valide", status=400)

    # Retourner le template mis à jour - Correction ici pour s'assurer que les changements sont appliqués
    response = render(
        request, "beadmodels/shapes/shape_dimensions.html", {"shape": shape}
    )
    response["HX-Trigger"] = "shapeDimensionsUpdated"
    return response


# Nouvelles vues pour l'édition en ligne du nom et du type
@login_required
def shape_name(request, shape_id):
    """Vue pour afficher le nom d'une forme avec le bouton d'édition"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_name.html", {"shape": shape})


@login_required
def shape_name_edit(request, shape_id):
    """Vue pour afficher le formulaire d'édition en ligne du nom"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_name_edit.html", {"shape": shape})


@login_required
@require_POST
def shape_name_update(request, shape_id):
    """Vue pour mettre à jour le nom d'une forme en ligne"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    new_name = request.POST.get("name", "").strip()

    if not new_name:
        return HttpResponse("Le nom ne peut pas être vide", status=400)

    # Vérifier si le nom existe déjà
    existing = (
        BeadShape.objects.filter(creator=request.user, name=new_name)
        .exclude(id=shape_id)
        .first()
    )

    if existing:
        return HttpResponse("Une forme avec ce nom existe déjà", status=400)

    shape.name = new_name
    shape.save()

    return render(request, "beadmodels/shapes/shape_name.html", {"shape": shape})


@login_required
def shape_type(request, shape_id):
    """Vue pour afficher le type d'une forme avec le bouton d'édition"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_type.html", {"shape": shape})


@login_required
def shape_type_edit(request, shape_id):
    """Vue pour afficher le formulaire d'édition en ligne du type"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_type_edit.html", {"shape": shape})


@login_required
@require_POST
def shape_type_update(request, shape_id):
    """Vue pour mettre à jour le type d'une forme en ligne"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    new_type = request.POST.get("shape_type", "")

    if new_type not in ["rectangle", "square", "circle"]:
        return HttpResponse("Type de forme invalide", status=400)

    # Si le type ne change pas, on ne fait rien
    if shape.shape_type == new_type:
        return render(request, "beadmodels/shapes/shape_type.html", {"shape": shape})

    # Si le type change, on réinitialise les dimensions selon le nouveau type
    old_type = shape.shape_type
    shape.shape_type = new_type

    if new_type == "rectangle":
        shape.width = 10
        shape.height = 10
        shape.size = None
        shape.diameter = None
    elif new_type == "square":
        shape.width = None
        shape.height = None
        shape.size = 10
        shape.diameter = None
    elif new_type == "circle":
        shape.width = None
        shape.height = None
        shape.size = None
        shape.diameter = 10

    shape.save()

    # Après changement de type, on renvoie directement le formulaire d'édition des dimensions
    # pour obliger l'utilisateur à les mettre à jour
    response = render(
        request,
        "beadmodels/shapes/shape_inline_edit.html",
        {"shape": shape, "type_changed": True},
    )
    response["HX-Trigger"] = "shapeTypeChanged"
    return response


@login_required
@require_POST
def shape_dimensions_update(request, shape_id):
    """Vue pour mettre à jour les dimensions d'une forme en ligne"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)

    # Récupérer les dimensions du formulaire
    if shape.shape_type == "rectangle":
        width = request.POST.get("width")
        height = request.POST.get("height")
        if width and height:
            try:
                shape.width = int(width)
                shape.height = int(height)
                shape.size = None
                shape.diameter = None
                shape.save()
            except ValueError:
                return HttpResponse(
                    "Les dimensions doivent être des nombres entiers", status=400
                )

    elif shape.shape_type == "square":
        size = request.POST.get("size")
        if size:
            try:
                shape.size = int(size)
                shape.width = None
                shape.height = None
                shape.diameter = None
                shape.save()
            except ValueError:
                return HttpResponse("La taille doit être un nombre entier", status=400)

    elif shape.shape_type == "circle":
        diameter = request.POST.get("diameter")
        if diameter:
            try:
                shape.diameter = int(diameter)
                shape.width = None
                shape.height = None
                shape.size = None
                shape.save()
            except ValueError:
                return HttpResponse(
                    "Le diamètre doit être un nombre entier", status=400
                )

    # Retourner la vue mise à jour des dimensions
    return render(request, "beadmodels/shapes/shape_dimensions.html", {"shape": shape})


@login_required
def shape_preview(request, shape_id):
    """Vue pour renvoyer l'aperçu HTML d'une forme pour la visualisation en direct"""
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
    return render(request, "beadmodels/shapes/shape_preview.html", {"shape": shape})
