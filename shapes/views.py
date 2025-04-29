from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .models import BeadShape


def shape_list(request):
    """Affiche la liste des formes disponibles"""
    # Récupération des formes par défaut (partagées) et des formes personnalisées de l'utilisateur
    default_shapes = BeadShape.objects.filter(is_default=True)

    if request.user.is_authenticated:
        custom_shapes = BeadShape.objects.filter(creator=request.user, is_default=False)
    else:
        custom_shapes = BeadShape.objects.none()

    shared_shapes = BeadShape.objects.filter(is_shared=True).exclude(is_default=True)

    if request.user.is_authenticated:
        shared_shapes = shared_shapes.exclude(creator=request.user)

    context = {
        "default_shapes": default_shapes,
        "custom_shapes": custom_shapes,
        "shared_shapes": shared_shapes,
    }

    return render(request, "shape_list.html", context)


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

    return render(request, "shapes/create_shape.html")


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


# def get_shape_details(request, shape_id):
#     """Récupère les détails d'une forme au format JSON pour utilisation via AJAX"""
#     shape = get_object_or_404(BeadShape, id=shape_id)

#     # Vérifier si l'utilisateur peut accéder à cette forme
#     if (
#         not shape.is_default
#         and not shape.is_shared
#         and (not request.user.is_authenticated or shape.creator != request.user)
#     ):
#         return JsonResponse({"error": "Accès refusé"}, status=403)

#     shape_data = {
#         "id": shape.id,
#         "name": shape.name,
#         "shape_type": shape.shape_type,
#         "parameters": shape.get_parameters(),
#     }

#     return JsonResponse(shape_data)
