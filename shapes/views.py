from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .models import BeadShape


@login_required
def shape_list(request):
    """Affiche la liste des formes disponibles"""
    # Optimisation : récupérer toutes les données nécessaires en une seule requête
    shapes = BeadShape.objects.filter(creator=request.user).select_related("creator")

    # Traitement pour la recherche et le filtrage si nécessaire
    search = request.GET.get("search", "")
    shape_type_filter = request.GET.get("shape_type_filter", "")

    if search:
        shapes = shapes.filter(name__icontains=search)
    if shape_type_filter:
        shapes = shapes.filter(shape_type=shape_type_filter)

    # Déterminer si c'est une requête HTMX pour le rendu partiel
    if request.htmx:
        return render(request, "shapes/shape_row_list.html", {"shapes": shapes})

    return render(request, "shapes/shape_list.html", {"shapes": shapes})


@login_required
def create_shape(request):
    """Crée une nouvelle forme"""
    if request.method == "POST":
        shape_type = request.POST.get("shape_type")
        name = request.POST.get("name")

        if shape_type == "rectangle":
            width = int(request.POST.get("width"))
            height = int(request.POST.get("height"))
            new_shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                width=width,
                height=height,
                creator=request.user,
            )
        elif shape_type == "square":
            size = int(request.POST.get("size"))
            new_shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                size=size,
                creator=request.user,
            )
        elif shape_type == "circle":
            diameter = int(request.POST.get("diameter"))
            new_shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                diameter=diameter,
                creator=request.user,
            )
        else:
            messages.error(request, "Type de forme inconnu")
            return redirect(reverse("shapes:shape_list"))

        messages.success(request, f"Forme {name} créée avec succès!")
        return redirect(reverse("shapes:shape_list"))

    return render(request, "partials/create_edit_shape.html")


@login_required
def update_shape(request, shape_id):
    """Met à jour une forme existante"""
    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        messages.error(
            request, "Vous n'avez pas l'autorisation de modifier cette forme"
        )
        return redirect(reverse("shapes:shape_list"))

    if request.method == "POST":
        shape.name = request.POST.get("name")
        shape_type = request.POST.get("shape_type")
        shape.shape_type = shape_type

        if shape_type == "rectangle":
            shape.width = int(request.POST.get("width"))
            shape.height = int(request.POST.get("height"))
            shape.size = None
            shape.diameter = None
        elif shape_type == "square":
            shape.size = int(request.POST.get("size"))
            shape.width = None
            shape.height = None
            shape.diameter = None
        elif shape_type == "circle":
            shape.diameter = int(request.POST.get("diameter"))
            shape.width = None
            shape.height = None
            shape.size = None

        # Option pour partager la forme
        share_shape = request.POST.get("share_shape") == "on"
        shape.is_shared = share_shape

        shape.save()
        messages.success(request, f"Forme {shape.name} mise à jour avec succès!")
        return redirect(reverse("shapes:shape_list"))

    context = {"shape": shape}
    return render(request, "shapes/update_shape.html", context)


@login_required
def delete_shape(request, shape_id):
    """Supprime une forme"""
    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut supprimer cette forme
    if shape.creator != request.user:
        messages.error(
            request, "Vous n'avez pas l'autorisation de supprimer cette forme"
        )
        return redirect(reverse("shapes:shape_list"))

    # Empêcher la suppression des formes par défaut
    if shape.is_default:
        messages.error(request, "Les formes par défaut ne peuvent pas être supprimées")
        return redirect(reverse("shapes:shape_list"))

    shape_name = shape.name
    shape.delete()
    messages.success(request, f"Forme {shape_name} supprimée avec succès!")
    return redirect(reverse("shapes:shape_list"))


@login_required
def shape_form_hx_view(request, shape_id=None):
    """Vue HTMX pour créer ou mettre à jour une forme"""
    if not request.htmx:
        raise Http404

    instance = None
    if shape_id is not None:
        try:
            instance = BeadShape.objects.get(id=shape_id)
            # Vérifier que l'utilisateur peut modifier cette forme
            if instance.creator != request.user and not instance.is_default:
                return HttpResponse(
                    "Vous n'avez pas l'autorisation de modifier cette forme", status=403
                )
        except:
            instance = None

    url = reverse("shapes:shape_create_hx")
    if instance:
        url = reverse("shapes:shape_update_hx", kwargs={"shape_id": instance.id})

    context = {"url": url, "shape": instance}

    return render(request, "partials/create_edit_shape.html", context)


@login_required
def shape_create_hx_view(request):
    """Vue HTMX pour traiter la création d'une forme"""
    if not request.htmx:
        raise Http404

    if request.method != "POST":
        return HttpResponse("Méthode non autorisée", status=405)

    shape_type = request.POST.get("shape_type")
    name = request.POST.get("name")
    share_shape = request.POST.get("share_shape") == "on"

    new_shape = None
    try:
        if shape_type == "rectangle":
            width = int(request.POST.get("width"))
            height = int(request.POST.get("height"))
            new_shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                width=width,
                height=height,
                is_shared=share_shape,
                creator=request.user,
            )
        elif shape_type == "square":
            size = int(request.POST.get("size"))
            new_shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                size=size,
                is_shared=share_shape,
                creator=request.user,
            )
        elif shape_type == "circle":
            diameter = int(request.POST.get("diameter"))
            new_shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                diameter=diameter,
                is_shared=share_shape,
                creator=request.user,
            )
        else:
            return HttpResponse("Type de forme inconnu", status=400)
    except Exception as e:
        return HttpResponse(f"Erreur : {str(e)}", status=400)

    # Récupérer toutes les formes pour mettre à jour la liste complète
    if request.user.is_authenticated:
        custom_shapes = BeadShape.objects.filter(creator=request.user, is_default=False)
    else:
        custom_shapes = BeadShape.objects.none()

    default_shapes = BeadShape.objects.filter(is_default=True)
    shared_shapes = BeadShape.objects.filter(is_shared=True).exclude(is_default=True)

    if request.user.is_authenticated:
        shared_shapes = shared_shapes.exclude(creator=request.user)

    context = {
        "default_shapes": default_shapes,
        "custom_shapes": custom_shapes,
        "shared_shapes": shared_shapes,
        "shapes": custom_shapes,  # Pour la compatibilité avec le template existant
        "success_message": f"Forme {name} créée avec succès!",
    }

    # Retourner le contenu mis à jour de la liste
    return render(request, "shape_row_list.html", context)


@login_required
def shape_update_hx_view(request, shape_id):
    """Vue HTMX pour traiter la mise à jour d'une forme"""
    if not request.htmx:
        raise Http404

    try:
        shape = BeadShape.objects.get(id=shape_id)
    except:
        return HttpResponse("Forme non trouvée", status=404)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

    if request.method != "POST":
        return HttpResponse("Méthode non autorisée", status=405)

    try:
        shape.name = request.POST.get("name")
        shape_type = request.POST.get("shape_type")
        shape.shape_type = shape_type
        shape.is_shared = request.POST.get("share_shape") == "on"

        if shape_type == "rectangle":
            shape.width = int(request.POST.get("width"))
            shape.height = int(request.POST.get("height"))
            shape.size = None
            shape.diameter = None
        elif shape_type == "square":
            shape.size = int(request.POST.get("size"))
            shape.width = None
            shape.height = None
            shape.diameter = None
        elif shape_type == "circle":
            shape.diameter = int(request.POST.get("diameter"))
            shape.width = None
            shape.height = None
            shape.size = None

        shape.save()
    except Exception as e:
        return HttpResponse(f"Erreur : {str(e)}", status=400)

    context = {
        "shape": shape,
        "success": True,
        "message": f"Forme {shape.name} mise à jour avec succès!",
    }

    # Retourner un rendu de la ligne mise à jour dans la liste
    return render(request, "partials/shape_row.html", context)


@login_required
def shape_delete_hx_view(request, shape_id):
    """Vue HTMX pour supprimer une forme"""
    if not request.htmx:
        raise Http404

    try:
        shape = BeadShape.objects.get(id=shape_id)
    except:
        return HttpResponse("Forme non trouvée", status=404)

    # Vérifier que l'utilisateur peut supprimer cette forme
    if shape.creator != request.user:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de supprimer cette forme", status=403
        )

    # Empêcher la suppression des formes par défaut
    if shape.is_default:
        return HttpResponse(
            "Les formes par défaut ne peuvent pas être supprimées", status=403
        )

    shape_name = shape.name
    shape.delete()

    # Vous pourriez vouloir retourner la liste complète mise à jour
    default_shapes = BeadShape.objects.filter(is_default=True)
    custom_shapes = BeadShape.objects.filter(creator=request.user, is_default=False)
    shared_shapes = (
        BeadShape.objects.filter(is_shared=True)
        .exclude(is_default=True)
        .exclude(creator=request.user)
    )

    context = {
        "default_shapes": default_shapes,
        "custom_shapes": custom_shapes,
        "shared_shapes": shared_shapes,
        "success_message": f"Forme {shape_name} supprimée avec succès!",
    }

    return render(request, "partials/shape_list_content.html", context)


@login_required
def shape_dimensions(request, shape_id):
    """Vue pour afficher les dimensions d'une forme"""
    if not request.htmx:
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)
    return render(request, "partials/shape_dimensions.html", {"shape": shape})


@login_required
def shape_inline_edit(request, shape_id):
    """Vue pour afficher l'éditeur inline des dimensions"""
    if not request.htmx:
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

    # Vérifier s'il y a eu un changement de type pour ajuster l'interface
    type_changed = request.GET.get("type_changed", False)

    context = {"shape": shape, "type_changed": type_changed}

    return render(request, "partials/shape_inline_edit.html", context)


@login_required
def shape_inline_update(request, shape_id):
    """Vue pour mettre à jour les dimensions d'une forme via l'éditeur inline"""
    if not request.htmx or request.method != "POST":
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

    try:
        shape_type = shape.shape_type

        if shape_type == "rectangle":
            shape.width = int(request.POST.get("width", 0))
            shape.height = int(request.POST.get("height", 0))
        elif shape_type == "square":
            shape.size = int(request.POST.get("size", 0))
        elif shape_type == "circle":
            shape.diameter = int(request.POST.get("diameter", 0))

        shape.save()

        # Dispatcher un événement HTMX pour mettre à jour les prévisualisations
        response = render(request, "partials/shape_dimensions.html", {"shape": shape})
        response["HX-Trigger"] = (
            f'{{"shapeDimensionsUpdated": {{"shapeId": "{shape.id}"}}}}'
        )
        return response

    except Exception as e:
        return HttpResponse(
            f"Erreur lors de la mise à jour des dimensions: {str(e)}", status=400
        )


@login_required
def shape_name(request, shape_id):
    """Vue pour afficher le nom d'une forme"""
    if not request.htmx:
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)
    return render(request, "partials/shape_name.html", {"shape": shape})


@login_required
def shape_name_edit(request, shape_id):
    """Vue pour afficher l'éditeur inline du nom"""
    if not request.htmx:
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

    return render(request, "partials/shape_name_edit.html", {"shape": shape})


@login_required
def shape_name_update(request, shape_id):
    """Vue pour mettre à jour le nom d'une forme via l'éditeur inline"""
    if not request.htmx or request.method != "POST":
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

    try:
        shape.name = request.POST.get("name", "")
        shape.save()
        return render(request, "partials/shape_name.html", {"shape": shape})

    except Exception as e:
        return HttpResponse(
            f"Erreur lors de la mise à jour du nom: {str(e)}", status=400
        )


@login_required
def shape_type(request, shape_id):
    """Vue pour afficher le type d'une forme"""
    if not request.htmx:
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)
    return render(request, "partials/shape_type.html", {"shape": shape})


@login_required
def shape_type_edit(request, shape_id):
    """Vue pour afficher l'éditeur inline du type"""
    if not request.htmx:
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

    return render(request, "partials/shape_type_edit.html", {"shape": shape})


@login_required
def shape_type_update(request, shape_id):
    """Vue pour mettre à jour le type d'une forme via l'éditeur inline"""
    if not request.htmx or request.method != "POST":
        raise Http404

    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        return HttpResponse(
            "Vous n'avez pas l'autorisation de modifier cette forme", status=403
        )

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

        response = render(request, "partials/shape_type.html", {"shape": shape})

        # Si le type a changé, déclencher une mise à jour des dimensions
        if old_type != new_type:
            response["HX-Trigger"] = (
                f'{{"shapeTypeChanged": {{"shapeId": "{shape.id}"}}}}'
            )

        return response

    except Exception as e:
        return HttpResponse(
            f"Erreur lors de la mise à jour du type: {str(e)}", status=400
        )


@login_required
def get_inline_add_form(request):
    """Vue HTMX pour obtenir le formulaire d'ajout en ligne directement dans le tableau"""
    if not request.htmx:
        raise Http404

    return render(request, "partials/shape_inline_add_form.html")


@login_required
def shape_save_or_update_hx_view(request, shape_id=None):
    """Vue HTMX unifiée pour traiter la création ou la mise à jour d'une forme"""
    if not request.htmx:
        raise Http404

    if request.method != "POST":
        return HttpResponse("Méthode non autorisée", status=405)

    # Déterminer si création ou mise à jour
    is_new = shape_id is None
    shape = None

    if not is_new:
        try:
            shape = BeadShape.objects.get(id=shape_id)
            # Vérifier que l'utilisateur peut modifier cette forme
            if shape.creator != request.user and not shape.is_default:
                return HttpResponse(
                    "Vous n'avez pas l'autorisation de modifier cette forme", status=403
                )
        except BeadShape.DoesNotExist:
            return HttpResponse("Forme non trouvée", status=404)

    shape_type = request.POST.get("shape_type")
    name = request.POST.get("name")
    share_shape = request.POST.get("share_shape") == "on"

    try:
        # Création d'une nouvelle forme
        if is_new:
            if shape_type == "rectangle":
                width = int(request.POST.get("width"))
                height = int(request.POST.get("height"))
                shape = BeadShape.objects.create(
                    name=name,
                    shape_type=shape_type,
                    width=width,
                    height=height,
                    is_shared=share_shape,
                    creator=request.user,
                )
            elif shape_type == "square":
                size = int(request.POST.get("size"))
                shape = BeadShape.objects.create(
                    name=name,
                    shape_type=shape_type,
                    size=size,
                    is_shared=share_shape,
                    creator=request.user,
                )
            elif shape_type == "circle":
                diameter = int(request.POST.get("diameter"))
                shape = BeadShape.objects.create(
                    name=name,
                    shape_type=shape_type,
                    diameter=diameter,
                    is_shared=share_shape,
                    creator=request.user,
                )
            else:
                return HttpResponse("Type de forme inconnu", status=400)

        # Mise à jour d'une forme existante
        else:
            shape.name = name
            shape.shape_type = shape_type
            shape.is_shared = share_shape

            if shape_type == "rectangle":
                shape.width = int(request.POST.get("width"))
                shape.height = int(request.POST.get("height"))
                shape.size = None
                shape.diameter = None
            elif shape_type == "square":
                shape.size = int(request.POST.get("size"))
                shape.width = None
                shape.height = None
                shape.diameter = None
            elif shape_type == "circle":
                shape.diameter = int(request.POST.get("diameter"))
                shape.width = None
                shape.height = None
                shape.size = None

            shape.save()

    except Exception as e:
        return HttpResponse(f"Erreur : {str(e)}", status=400)

    # Pour une création, récupérer toutes les formes pour mettre à jour la liste complète
    if is_new:
        if request.user.is_authenticated:
            custom_shapes = BeadShape.objects.filter(
                creator=request.user, is_default=False
            )
        else:
            custom_shapes = BeadShape.objects.none()

        default_shapes = BeadShape.objects.filter(is_default=True)
        shared_shapes = BeadShape.objects.filter(is_shared=True).exclude(
            is_default=True
        )

        if request.user.is_authenticated:
            shared_shapes = shared_shapes.exclude(creator=request.user)

        context = {
            "default_shapes": default_shapes,
            "custom_shapes": custom_shapes,
            "shared_shapes": shared_shapes,
            "shapes": custom_shapes,  # Pour la compatibilité avec le template existant
            "success_message": f"Forme {name} créée avec succès!",
        }

        # Retourner le contenu mis à jour de la liste
        return render(request, "shape_row_list.html", context)

    # Pour une mise à jour, retourner juste la ligne mise à jour
    else:
        context = {
            "shape": shape,
            "success": True,
            "message": f"Forme {shape.name} mise à jour avec succès!",
        }

        # Retourner un rendu de la ligne mise à jour dans la liste
        return render(request, "partials/shape_row.html", context)
