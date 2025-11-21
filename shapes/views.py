from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .models import BeadShape


@login_required
def shape_list(request):
    """Redirige vers la vue en colonnes"""
    # Rediriger vers la vue en colonnes qui est maintenant l'interface principale
    return redirect("shapes:shape_list_columns")


@login_required
def shape_list_columns(request):
    """Affiche la liste des formes disponibles en colonnes par type"""
    # Récupérer toutes les formes de l'utilisateur
    user_shapes = BeadShape.objects.filter(
        Q(creator=request.user) | Q(is_shared=True)
    ).select_related("creator")

    # Vérifier si on est en mode d'édition
    edit_shape_id = request.GET.get("edit")
    edit_shape = None
    if edit_shape_id:
        try:
            edit_shape = BeadShape.objects.get(id=edit_shape_id)
            # Vérifier que l'utilisateur peut modifier cette forme
            if edit_shape.creator != request.user and not edit_shape.is_default:
                messages.error(
                    request, "Vous n'avez pas l'autorisation de modifier cette forme"
                )
                edit_shape = None
        except BeadShape.DoesNotExist:
            edit_shape = None

    # Séparer les formes par type
    rectangles = user_shapes.filter(shape_type="rectangle")
    squares = user_shapes.filter(shape_type="square")
    circles = user_shapes.filter(shape_type="circle")

    # Messages de succès ou d'erreur (venant de redirections)
    success_message = request.session.pop("success_message", None)
    error_message = request.session.pop("error_message", None)

    context = {
        "rectangles": rectangles,
        "squares": squares,
        "circles": circles,
        "edit_shape": edit_shape,
        "success_message": success_message,
        "error_message": error_message,
    }

    return render(request, "shapes/shape_list_columns.html", context)


@login_required
def create_shape(request):
    """Crée une nouvelle forme"""
    if request.method == "POST":
        shape_option = request.POST.get("shape_option", "rectangle")
        share_shape = request.POST.get("share_shape") == "on"

        # Générer un nom automatiquement basé sur les dimensions
        name = "Forme"

        # Définir les dimensions en fonction du type sélectionné par l'utilisateur
        if (
            shape_option == "rectangle"
            and request.POST.get("width")
            and request.POST.get("height")
        ):
            width = int(request.POST.get("width"))
            height = int(request.POST.get("height"))
            name = f"Rectangle {width}×{height}"

            new_shape = BeadShape(
                name=name,
                creator=request.user,
                is_shared=share_shape,
                width=width,
                height=height,
            )

        elif shape_option == "square" and request.POST.get("size"):
            size = int(request.POST.get("size"))
            name = f"Carré {size}×{size}"

            new_shape = BeadShape(
                name=name, creator=request.user, is_shared=share_shape, size=size
            )

        elif shape_option == "circle" and request.POST.get("diameter"):
            diameter = int(request.POST.get("diameter"))
            name = f"Cercle Ø{diameter}"

            new_shape = BeadShape(
                name=name,
                creator=request.user,
                is_shared=share_shape,
                diameter=diameter,
            )

        else:
            if request.htmx:
                request.session["error_message"] = "Aucune dimension valide fournie"
                return redirect(reverse("shapes:shape_list_columns"))
            else:
                messages.error(request, "Aucune dimension valide fournie")
                return redirect(reverse("shapes:shape_list_columns"))

        # Déterminer le type automatiquement à partir des dimensions
        new_shape.update_from_dimensions()
        new_shape.save()

        if request.htmx:
            request.session["success_message"] = "Forme créée avec succès!"
            return redirect(reverse("shapes:shape_list_columns"))
        else:
            messages.success(request, "Forme créée avec succès!")
            return redirect(reverse("shapes:shape_list_columns"))

    return redirect(reverse("shapes:shape_list_columns"))


@login_required
def update_shape(request, shape_id):
    """Met à jour une forme existante"""
    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut modifier cette forme
    if shape.creator != request.user and not shape.is_default:
        if request.htmx:
            request.session["error_message"] = (
                "Vous n'avez pas l'autorisation de modifier cette forme"
            )
            return redirect(reverse("shapes:shape_list_columns"))
        else:
            messages.error(
                request, "Vous n'avez pas l'autorisation de modifier cette forme"
            )
            return redirect(reverse("shapes:shape_list_columns"))

    if request.method == "POST":
        shape_option = request.POST.get("shape_option", "rectangle")
        share_shape = request.POST.get("share_shape") == "on"
        shape.is_shared = share_shape

        # Mettre à jour les dimensions et générer un nouveau nom en fonction du type
        if (
            shape_option == "rectangle"
            and request.POST.get("width")
            and request.POST.get("height")
        ):
            width = int(request.POST.get("width"))
            height = int(request.POST.get("height"))
            shape.name = f"Rectangle {width}×{height}"
            shape.width = width
            shape.height = height
            shape.size = None
            shape.diameter = None
        elif shape_option == "square" and request.POST.get("size"):
            size = int(request.POST.get("size"))
            shape.name = f"Carré {size}×{size}"
            shape.size = size
            shape.width = None
            shape.height = None
            shape.diameter = None
        elif shape_option == "circle" and request.POST.get("diameter"):
            diameter = int(request.POST.get("diameter"))
            shape.name = f"Cercle Ø{diameter}"
            shape.diameter = diameter
            shape.width = None
            shape.height = None
            shape.size = None
        else:
            if request.htmx:
                request.session["error_message"] = "Aucune dimension valide fournie"
                return redirect(reverse("shapes:shape_list_columns"))
            else:
                messages.error(request, "Aucune dimension valide fournie")
                return redirect(reverse("shapes:shape_list_columns"))

        # Déterminer le type automatiquement à partir des dimensions
        shape.update_from_dimensions()
        shape.save()

        if request.htmx:
            request.session["success_message"] = "Forme mise à jour avec succès!"
            return redirect(reverse("shapes:shape_list_columns"))
        else:
            messages.success(request, "Forme mise à jour avec succès!")
            return redirect(reverse("shapes:shape_list_columns"))

    # En théorie, on ne devrait jamais arriver ici car l'édition se fait via la vue shape_list_columns
    return redirect(reverse("shapes:shape_list_columns"))


@login_required
def delete_shape(request, shape_id):
    """Supprime une forme"""
    shape = get_object_or_404(BeadShape, id=shape_id)

    # Vérifier que l'utilisateur peut supprimer cette forme
    if shape.creator != request.user:
        if request.htmx:
            request.session["error_message"] = (
                "Vous n'avez pas l'autorisation de supprimer cette forme"
            )
            return redirect(reverse("shapes:shape_list_columns"))
        else:
            messages.error(
                request, "Vous n'avez pas l'autorisation de supprimer cette forme"
            )
            return redirect(reverse("shapes:shape_list_columns"))

    # Empêcher la suppression des formes par défaut
    if shape.is_default:
        if request.htmx:
            request.session["error_message"] = (
                "Les formes par défaut ne peuvent pas être supprimées"
            )
            return redirect(reverse("shapes:shape_list_columns"))
        else:
            messages.error(
                request, "Les formes par défaut ne peuvent pas être supprimées"
            )
            return redirect(reverse("shapes:shape_list_columns"))

    shape_name = shape.name
    shape.delete()

    if request.htmx:
        request.session["success_message"] = (
            f"Forme {shape_name} supprimée avec succès!"
        )
        return redirect(reverse("shapes:shape_list_columns"))
    else:
        messages.success(request, f"Forme {shape_name} supprimée avec succès!")
        return redirect(reverse("shapes:shape_list_columns"))


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

    name = request.POST.get("name")
    share_shape = request.POST.get("share_shape") == "on"
    dimension_type = request.POST.get(
        "dimension_type", request.POST.get("shape_type", "")
    )

    new_shape = None
    try:
        # Créer une nouvelle forme
        new_shape = BeadShape(name=name, is_shared=share_shape, creator=request.user)

        # Définir les dimensions en fonction des champs fournis
        if (
            "width" in request.POST
            and "height" in request.POST
            and request.POST.get("width")
            and request.POST.get("height")
        ):
            new_shape.width = int(request.POST.get("width"))
            new_shape.height = int(request.POST.get("height"))
        elif "size" in request.POST and request.POST.get("size"):
            new_shape.size = int(request.POST.get("size"))
        elif "diameter" in request.POST and request.POST.get("diameter"):
            new_shape.diameter = int(request.POST.get("diameter"))
        else:
            return HttpResponse("Aucune dimension valide fournie", status=400)

        # Déterminer le type automatiquement à partir des dimensions
        new_shape.update_from_dimensions()
        new_shape.save()
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

        # Enregistrer les dimensions selon le type actuel
        if shape_type == "rectangle":
            shape.width = int(request.POST.get("width", 0))
            shape.height = int(request.POST.get("height", 0))
        elif shape_type == "square":
            shape.size = int(request.POST.get("size", 0))
        elif shape_type == "circle":
            shape.diameter = int(request.POST.get("diameter", 0))

        # Déterminer si le type doit changer en fonction des dimensions
        shape.update_from_dimensions()

        # Sauvegarder les changements
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

        # Marquer le changement comme explicite
        request.POST._mutable = True
        request.POST["explicit_type_change"] = "true"
        request.POST._mutable = False

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
            # Créer une nouvelle forme avec les valeurs de base
            shape = BeadShape(name=name, is_shared=share_shape, creator=request.user)

            # Définir les dimensions en fonction des champs fournis
            if (
                "width" in request.POST
                and "height" in request.POST
                and request.POST.get("width")
                and request.POST.get("height")
            ):
                shape.width = int(request.POST.get("width"))
                shape.height = int(request.POST.get("height"))
            elif "size" in request.POST and request.POST.get("size"):
                shape.size = int(request.POST.get("size"))
            elif "diameter" in request.POST and request.POST.get("diameter"):
                shape.diameter = int(request.POST.get("diameter"))
            else:
                return HttpResponse("Aucune dimension valide fournie", status=400)

            # Déterminer le type automatiquement
            shape.update_from_dimensions()
            shape.save()

        # Mise à jour d'une forme existante
        else:
            shape.name = name
            shape.is_shared = share_shape

            # Mettre à jour les dimensions selon les champs fournis
            if (
                "width" in request.POST
                and "height" in request.POST
                and request.POST.get("width")
                and request.POST.get("height")
            ):
                shape.width = int(request.POST.get("width"))
                shape.height = int(request.POST.get("height"))
                shape.size = None
                shape.diameter = None
            elif "size" in request.POST and request.POST.get("size"):
                shape.size = int(request.POST.get("size"))
                shape.width = None
                shape.height = None
                shape.diameter = None
            elif "diameter" in request.POST and request.POST.get("diameter"):
                shape.diameter = int(request.POST.get("diameter"))
                shape.width = None
                shape.height = None
                shape.size = None

            # Déterminer automatiquement le type en fonction des dimensions
            shape.update_from_dimensions()
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
