import json
import os
import time

import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods
from PIL import Image

from .forms import BeadModelForm, UserProfileForm, UserRegistrationForm
from .models import Bead, BeadModel, BeadShape, CustomShape


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
        elif active_tab == "beads":
            try:
                data = json.loads(request.body)
                action = data.get("action")
                bead_id = data.get("bead_id")
                name = data.get("name")
                red = data.get("red")
                green = data.get("green")
                blue = data.get("blue")
                quantity = data.get("quantity")
                notes = data.get("notes")

                if action == "add":
                    Bead.objects.create(
                        creator=request.user,
                        name=name,
                        red=red,
                        green=green,
                        blue=blue,
                        quantity=quantity,
                        notes=notes,
                    )
                    messages.success(request, "Perle ajoutée avec succès!")
                elif action == "update" and bead_id:
                    bead = get_object_or_404(Bead, id=bead_id, creator=request.user)
                    bead.name = name
                    bead.red = red
                    bead.green = green
                    bead.blue = blue
                    bead.quantity = quantity
                    bead.notes = notes
                    bead.save()
                    messages.success(request, "Perle mise à jour avec succès!")
                elif action == "delete" and bead_id:
                    bead = get_object_or_404(Bead, id=bead_id, creator=request.user)
                    bead.delete()
                    messages.success(request, "Perle supprimée avec succès!")

                return JsonResponse({"success": True})
            except Exception as e:
                return JsonResponse({"success": False, "message": str(e)})
    else:
        if active_tab == "profile":
            context["form"] = UserProfileForm(instance=request.user)
        elif active_tab == "shapes":
            context["saved_shapes"] = BeadShape.objects.filter(
                creator=request.user
            ).order_by("-created_at")
        elif active_tab == "beads":
            context["beads"] = Bead.objects.filter(creator=request.user)

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

            # Vérifier si une forme similaire existe déjà
            existing_shapes = BeadShape.objects.filter(creator=request.user)
            for existing_shape in existing_shapes:
                if existing_shape.shape_type == shape_type:
                    if shape_type == "rectangle":
                        if existing_shape.width == parameters.get(
                            "width"
                        ) and existing_shape.height == parameters.get("height"):
                            return JsonResponse(
                                {
                                    "success": False,
                                    "message": "Une forme rectangulaire avec ces dimensions existe déjà",
                                },
                                status=400,
                            )
                    elif shape_type == "square":
                        if existing_shape.size == parameters.get("size"):
                            return JsonResponse(
                                {
                                    "success": False,
                                    "message": "Un carré avec cette taille existe déjà",
                                },
                                status=400,
                            )
                    elif shape_type == "circle":
                        if existing_shape.diameter == parameters.get("diameter"):
                            return JsonResponse(
                                {
                                    "success": False,
                                    "message": "Un cercle avec ce diamètre existe déjà",
                                },
                                status=400,
                            )

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

            return JsonResponse(
                {"success": True, "message": "La forme a été créée avec succès"}
            )

        except json.JSONDecodeError:
            return JsonResponse(
                {"success": False, "message": "Données JSON invalides"}, status=400
            )
        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)}, status=500)

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

            # Vérifier si une autre forme similaire existe déjà (exclure la forme actuelle)
            existing_shapes = BeadShape.objects.filter(creator=request.user).exclude(
                id=shape.id
            )
            for existing_shape in existing_shapes:
                if existing_shape.shape_type == shape_type:
                    if shape_type == "rectangle":
                        if existing_shape.width == parameters.get(
                            "width"
                        ) and existing_shape.height == parameters.get("height"):
                            return JsonResponse(
                                {
                                    "success": False,
                                    "message": "Une forme rectangulaire avec ces dimensions existe déjà",
                                },
                                status=400,
                            )
                    elif shape_type == "square":
                        if existing_shape.size == parameters.get("size"):
                            return JsonResponse(
                                {
                                    "success": False,
                                    "message": "Un carré avec cette taille existe déjà",
                                },
                                status=400,
                            )
                    elif shape_type == "circle":
                        if existing_shape.diameter == parameters.get("diameter"):
                            return JsonResponse(
                                {
                                    "success": False,
                                    "message": "Un cercle avec ce diamètre existe déjà",
                                },
                                status=400,
                            )

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


@login_required
@require_http_methods(["POST"])
def transform_image(request, pk):
    try:
        model = get_object_or_404(BeadModel, pk=pk)
        grid_size = int(request.POST.get("grid_size", 32))
        color_reduction = int(request.POST.get("color_reduction", 32))
        dithering = request.POST.get("dithering", "none")

        # Vérifier les permissions
        if model.creator != request.user:
            return JsonResponse(
                {
                    "success": False,
                    "message": "Vous n'avez pas le droit de modifier ce modèle.",
                },
                status=403,
            )

        # Ouvrir l'image originale
        original_image = Image.open(model.original_image.path)

        # Convertir en RGB si nécessaire
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")

        # Redimensionner l'image selon la taille de la grille
        new_size = (grid_size, grid_size)
        resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)

        # Créer une nouvelle image avec espacement
        spacing = 2  # Espacement en pixels entre chaque pixel
        final_size = (grid_size * (1 + spacing), grid_size * (1 + spacing))
        final_image = Image.new("RGB", final_size, (255, 255, 255))  # Fond blanc

        # Copier les pixels avec espacement
        pixels = resized_image.load()
        final_pixels = final_image.load()
        for y in range(grid_size):
            for x in range(grid_size):
                final_pixels[x * (1 + spacing), y * (1 + spacing)] = pixels[x, y]

        # Réduire les couleurs
        if color_reduction < 256:
            # Convertir l'image en palette
            final_image = final_image.quantize(colors=color_reduction, method=2)

        # Appliquer le tramage si demandé
        if dithering == "floyd-steinberg":
            final_image = final_image.convert("RGB")
            # Implémenter l'algorithme Floyd-Steinberg
            pixels = final_image.load()
            width, height = final_image.size
            for y in range(height):
                for x in range(width):
                    if (
                        x % (1 + spacing) == 0 and y % (1 + spacing) == 0
                    ):  # Ne modifier que les pixels d'origine
                        old_pixel = pixels[x, y]
                        # Convertir en noir ou blanc
                        new_pixel = (
                            (0, 0, 0) if sum(old_pixel) < 384 else (255, 255, 255)
                        )
                        pixels[x, y] = new_pixel

                        # Calculer l'erreur
                        error = tuple(
                            old - new for old, new in zip(old_pixel, new_pixel)
                        )

                        # Distribuer l'erreur aux pixels voisins
                        if x + (1 + spacing) < width:
                            pixels[x + (1 + spacing), y] = tuple(
                                p + int(e * 7 / 16)
                                for p, e in zip(pixels[x + (1 + spacing), y], error)
                            )
                        if y + (1 + spacing) < height:
                            if x - (1 + spacing) >= 0:
                                pixels[x - (1 + spacing), y + (1 + spacing)] = tuple(
                                    p + int(e * 3 / 16)
                                    for p, e in zip(
                                        pixels[x - (1 + spacing), y + (1 + spacing)],
                                        error,
                                    )
                                )
                            pixels[x, y + (1 + spacing)] = tuple(
                                p + int(e * 5 / 16)
                                for p, e in zip(pixels[x, y + (1 + spacing)], error)
                            )
                            if x + (1 + spacing) < width:
                                pixels[x + (1 + spacing), y + (1 + spacing)] = tuple(
                                    p + int(e * 1 / 16)
                                    for p, e in zip(
                                        pixels[x + (1 + spacing), y + (1 + spacing)],
                                        error,
                                    )
                                )

        elif dithering == "ordered":
            # Implémenter le tramage ordonné
            dither_matrix = [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ]
            final_image = final_image.convert("RGB")
            pixels = final_image.load()
            width, height = final_image.size
            for y in range(height):
                for x in range(width):
                    if (
                        x % (1 + spacing) == 0 and y % (1 + spacing) == 0
                    ):  # Ne modifier que les pixels d'origine
                        old_pixel = pixels[x, y]
                        threshold = dither_matrix[y % 4][x % 4] * 16
                        new_pixel = (
                            (0, 0, 0) if sum(old_pixel) < threshold else (255, 255, 255)
                        )
                        pixels[x, y] = new_pixel

        # Redimensionner l'image finale à la taille de l'image originale
        original_width, original_height = original_image.size
        final_image = final_image.resize(
            (original_width, original_height), Image.Resampling.NEAREST
        )

        # Sauvegarder le résultat temporairement
        temp_filename = f"temp_pattern_{pk}_{int(time.time())}.png"
        temp_path = os.path.join(settings.MEDIA_ROOT, "patterns", temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        final_image.save(temp_path)

        # Retourner l'URL de l'image transformée
        return JsonResponse(
            {
                "success": True,
                "message": "Image transformée avec succès",
                "image_url": os.path.join(
                    settings.MEDIA_URL, "patterns", temp_filename
                ),
            }
        )

    except Exception as e:
        return JsonResponse(
            {
                "success": False,
                "message": f"Erreur lors de la transformation : {str(e)}",
            },
            status=500,
        )


@login_required
@require_http_methods(["POST"])
def save_transformation(request):
    try:
        data = json.loads(request.body)
        model_id = data.get("model_id")
        image_url = data.get("image_url")

        model = get_object_or_404(BeadModel, pk=model_id)

        # Vérifier les permissions
        if model.creator != request.user:
            return JsonResponse(
                {
                    "success": False,
                    "message": "Vous n'avez pas le droit de modifier ce modèle.",
                },
                status=403,
            )

        # Supprimer l'ancien motif s'il existe
        if model.bead_pattern:
            try:
                os.remove(model.bead_pattern.path)
            except:
                pass

        # Mettre à jour le modèle avec le nouveau motif
        model.bead_pattern = image_url
        model.save()

        return JsonResponse(
            {"success": True, "message": "Transformation sauvegardée avec succès"}
        )

    except Exception as e:
        return JsonResponse(
            {"success": False, "message": f"Erreur lors de la sauvegarde : {str(e)}"},
            status=500,
        )
