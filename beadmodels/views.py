import json

import numpy as np
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods
from PIL import Image

from .forms import BeadModelForm, UserProfileForm, UserRegistrationForm
from .models import BeadModel, BeadShape, CustomShape


def home(request):
    public_models = BeadModel.objects.filter(is_public=True).order_by("-created_at")[
        :12
    ]
    return render(request, "beadmodels/home.html", {"models": public_models})


def register(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(
                request,
                "Votre compte a été créé avec succès ! Vous pouvez maintenant vous connecter.",
            )
            return redirect("login")
    else:
        form = UserRegistrationForm()
    return render(request, "registration/register.html", {"form": form})


@login_required
def user_settings(request):
    active_tab = request.GET.get("tab", "profile")
    context = {
        "active_tab": active_tab,
    }

    if request.method == "POST":
        if active_tab == "profile":
            form = UserProfileForm(request.POST, instance=request.user)
            if form.is_valid():
                form.save()
                messages.success(request, "Votre profil a été mis à jour avec succès!")
                return redirect("beadmodels:user_settings")
            context["form"] = form
        elif active_tab == "password":
            # TODO: Implémenter le changement de mot de passe
            pass
        elif active_tab == "preferences":
            # Gérer les préférences utilisateur
            default_grid_size = request.POST.get("default_grid_size")
            public_by_default = request.POST.get("public_by_default") == "on"
            # TODO: Sauvegarder les préférences
            messages.success(
                request, "Vos préférences ont été mises à jour avec succès!"
            )
            return redirect("beadmodels:user_settings")
    else:
        if active_tab == "profile":
            context["form"] = UserProfileForm(instance=request.user)
        elif active_tab == "shapes":
            # Charger les formes de l'utilisateur
            context["saved_shapes"] = BeadShape.objects.filter(
                creator=request.user
            ).order_by("-created_at")

    return render(request, "beadmodels/user_settings.html", context)


@login_required
def create_model(request):
    if request.method == "POST":
        form = BeadModelForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=False)
            model.creator = request.user
            model.save()
            messages.success(request, "Votre modèle a été créé avec succès!")
            return redirect("beadmodels:model_detail", pk=model.pk)
    else:
        form = BeadModelForm()
    return render(request, "beadmodels/create_model.html", {"form": form})


def model_detail(request, pk):
    model = get_object_or_404(BeadModel, pk=pk)
    if not model.is_public and model.creator != request.user:
        messages.error(request, "Vous n'avez pas accès à ce modèle.")
        return redirect("beadmodels:home")
    return render(request, "beadmodels/model_detail.html", {"model": model})


@login_required
def edit_model(request, pk):
    model = get_object_or_404(BeadModel, pk=pk)
    if model.creator != request.user:
        messages.error(request, "Vous n'avez pas le droit de modifier ce modèle.")
        return redirect("beadmodels:model_detail", pk=pk)

    if request.method == "POST":
        form = BeadModelForm(request.POST, request.FILES, instance=model)
        if form.is_valid():
            form.save()
            messages.success(request, "Votre modèle a été modifié avec succès!")
            return redirect("beadmodels:model_detail", pk=model.pk)
    else:
        form = BeadModelForm(instance=model)
    return render(request, "beadmodels/edit_model.html", {"form": form, "model": model})


@login_required
def delete_model(request, pk):
    model = get_object_or_404(BeadModel, pk=pk)
    if model.creator != request.user:
        messages.error(request, "Vous n'avez pas le droit de supprimer ce modèle.")
        return redirect("beadmodels:model_detail", pk=pk)

    if request.method == "POST":
        model.delete()
        messages.success(request, "Votre modèle a été supprimé avec succès!")
        return redirect("beadmodels:my_models")

    return render(request, "beadmodels/delete_model.html", {"model": model})


@login_required
def my_models(request):
    user_models = BeadModel.objects.filter(creator=request.user).order_by("-created_at")
    return render(request, "beadmodels/my_models.html", {"models": user_models})


@login_required
def save_shape(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print("Données reçues:", data)  # Log des données reçues
            name = data.get("name")
            shape_type = data.get("type")
            parameters = data.get("parameters", {})
            original_shape_id = data.get(
                "original_shape_id"
            )  # ID de la forme originale à supprimer

            # Si nous avons un ID de forme originale, la supprimer
            if original_shape_id:
                try:
                    original_shape = BeadShape.objects.get(
                        id=original_shape_id, creator=request.user
                    )
                    original_shape.delete()
                except BeadShape.DoesNotExist:
                    pass  # Ignorer si la forme n'existe pas

            # Créer une nouvelle forme
            shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                creator=request.user,
                is_shared=True,  # Par défaut, la forme est partagée
            )

            if shape_type == "rectangle":
                shape.width = parameters.get("width")
                shape.height = parameters.get("height")
            elif shape_type == "square":
                shape.size = parameters.get("size")
            elif shape_type == "circle":
                shape.diameter = parameters.get("diameter")
            shape.save()

            return JsonResponse(
                {
                    "success": True,
                    "message": "Forme sauvegardée avec succès",
                    "shape": {
                        "id": shape.id,
                        "name": shape.name,
                        "type": shape.shape_type,
                        "parameters": shape.get_parameters(),
                    },
                }
            )

        except json.JSONDecodeError as e:
            print("Erreur JSON:", str(e))  # Log de l'erreur JSON
            return JsonResponse(
                {"success": False, "message": "Données JSON invalides"}, status=400
            )
        except Exception as e:
            print("Erreur générale:", str(e))  # Log de l'erreur générale
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Méthode non autorisée"}, status=405
    )


@login_required
def delete_shape(request, shape_id):
    if request.method == "POST":
        try:
            shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)
            shape.delete()
            messages.success(request, "La forme a été supprimée avec succès.")
        except Exception as e:
            messages.error(
                request, f"Une erreur est survenue lors de la suppression : {str(e)}"
            )
    return redirect("beadmodels:user_settings", tab="shapes")


@login_required
def create_shape(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get("name")
            shape_type = data.get("type")
            parameters = data.get("parameters", {})

            shape = BeadShape.objects.create(
                name=name,
                shape_type=shape_type,
                creator=request.user,
                is_shared=True,
            )

            if shape_type == "rectangle":
                shape.width = parameters.get("width")
                shape.height = parameters.get("height")
            elif shape_type == "square":
                shape.size = parameters.get("size")
            elif shape_type == "circle":
                shape.diameter = parameters.get("diameter")
            shape.save()

            messages.success(request, "La forme a été créée avec succès.")
            return redirect("beadmodels:user_settings", tab="shapes")

        except json.JSONDecodeError:
            messages.error(request, "Données JSON invalides")
            return redirect("beadmodels:user_settings", tab="shapes")
        except Exception as e:
            messages.error(request, f"Une erreur est survenue : {str(e)}")
            return redirect("beadmodels:user_settings", tab="shapes")

    return render(request, "beadmodels/create_shape.html")


@login_required
def edit_shape(request, shape_id):
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get("name")
            shape_type = data.get("type")
            parameters = data.get("parameters", {})

            shape.name = name
            shape.shape_type = shape_type

            if shape_type == "rectangle":
                shape.width = parameters.get("width")
                shape.height = parameters.get("height")
            elif shape_type == "square":
                shape.size = parameters.get("size")
            elif shape_type == "circle":
                shape.diameter = parameters.get("diameter")
            shape.save()

            return JsonResponse(
                {"success": True, "message": "La forme a été modifiée avec succès"}
            )

        except json.JSONDecodeError:
            return JsonResponse(
                {"success": False, "message": "Données JSON invalides"}, status=400
            )
        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return render(request, "beadmodels/edit_shape.html", {"shape": shape})
