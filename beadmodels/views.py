import base64
import json
import os
import time
from io import BytesIO

import cv2
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.views.decorators.http import require_http_methods
from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
    UpdateView,
)
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .forms import BeadForm, BeadModelForm, TransformModelForm
from .models import Bead, BeadModel


# Vues basées sur des classes pour la gestion des modèles
class BeadModelListView(ListView):
    model = BeadModel
    template_name = "beadmodels/models/my_models.html"
    context_object_name = "models"

    def get_queryset(self):
        return BeadModel.objects.filter(creator=self.request.user).order_by(
            "-created_at"
        )


class BeadModelDetailView(DetailView):
    model = BeadModel
    template_name = "beadmodels/models/model_detail.html"
    context_object_name = "model"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["transform_form"] = TransformModelForm()
        return context

    def dispatch(self, request, *args, **kwargs):
        model = self.get_object()
        if not model.is_public and model.creator != request.user:
            messages.error(request, "Vous n'avez pas accès à ce modèle.")
            return redirect("beadmodels:home")
        return super().dispatch(request, *args, **kwargs)


class BeadModelCreateView(LoginRequiredMixin, CreateView):
    model = BeadModel
    form_class = BeadModelForm
    template_name = "beadmodels/models/create_model.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        form.instance.creator = self.request.user
        messages.success(self.request, "Votre modèle a été créé avec succès!")
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("beadmodels:model_detail", kwargs={"pk": self.object.pk})


class BeadModelUpdateView(LoginRequiredMixin, UpdateView):
    model = BeadModel
    form_class = BeadModelForm
    template_name = "beadmodels/models/edit_model.html"

    def dispatch(self, request, *args, **kwargs):
        model = self.get_object()
        if model.creator != request.user:
            messages.error(request, "Vous n'avez pas le droit de modifier ce modèle.")
            return redirect("beadmodels:model_detail", pk=model.pk)
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        # Vérifier si une nouvelle image est fournie
        if "original_image" in form.changed_data:
            # Récupérer l'instance de modèle avant modification
            old_instance = self.model.objects.get(pk=self.object.pk)

            # Stocker le chemin de l'ancienne image si elle existe
            if old_instance.original_image:
                old_image_path = old_instance.original_image.path

                # Supprimer également le motif en perles associé s'il existe
                if old_instance.bead_pattern:
                    try:
                        os.remove(old_instance.bead_pattern.path)
                        # Réinitialiser le champ bead_pattern car l'image originale a changé
                        self.object.bead_pattern = None
                    except (FileNotFoundError, ValueError):
                        # Ne rien faire si le fichier n'existe pas ou si le chemin est invalide
                        pass

                # Sauvegarder d'abord pour que la nouvelle image soit enregistrée
                response = super().form_valid(form)

                # Supprimer l'ancienne image après avoir sauvegardé la nouvelle
                try:
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                except (FileNotFoundError, ValueError):
                    # Ne rien faire si le fichier n'existe pas ou si le chemin est invalide
                    pass

                messages.success(
                    self.request,
                    "Votre modèle a été modifié avec succès! L'ancienne image a été supprimée.",
                )
                return response

        # Si aucune nouvelle image n'est fournie, procéder normalement
        messages.success(self.request, "Votre modèle a été modifié avec succès!")
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("beadmodels:model_detail", kwargs={"pk": self.object.pk})


class BeadModelDeleteView(LoginRequiredMixin, DeleteView):
    model = BeadModel
    template_name = "beadmodels/models/delete_model.html"
    success_url = reverse_lazy("beadmodels:my_models")

    def dispatch(self, request, *args, **kwargs):
        model = self.get_object()
        if model.creator != request.user:
            messages.error(request, "Vous n'avez pas le droit de supprimer ce modèle.")
            return redirect("beadmodels:model_detail", pk=model.pk)
        return super().dispatch(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        messages.success(request, "Votre modèle a été supprimé avec succès!")
        return super().delete(request, *args, **kwargs)


# Vues basées sur des classes pour la gestion des perles
class BeadListView(LoginRequiredMixin, ListView):
    model = Bead
    template_name = (
        "beadmodels/beads/bead_table.html"  # Utiliser notre nouveau template tabulaire
    )
    context_object_name = "beads"

    def get_queryset(self):
        return Bead.objects.filter(creator=self.request.user).order_by("name")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Ajouter le seuil d'alerte au contexte
        from django.conf import settings

        context["threshold"] = getattr(settings, "BEAD_LOW_QUANTITY_THRESHOLD", 20)

        # Identifier les perles sous le seuil
        threshold = context["threshold"]
        context["low_quantity_beads"] = [
            bead
            for bead in context["beads"]
            if bead.quantity is not None and bead.quantity <= threshold
        ]

        return context


class BeadCreateView(LoginRequiredMixin, CreateView):
    model = Bead
    form_class = BeadForm
    success_url = reverse_lazy("beadmodels:bead_list")

    # Supprimer la référence à template_name pour utiliser le formulaire intégré dans la page de tableau

    def form_valid(self, form):
        form.instance.creator = self.request.user
        messages.success(self.request, "Perle ajoutée avec succès!")
        return super().form_valid(form)


class BeadUpdateView(LoginRequiredMixin, UpdateView):
    model = Bead
    form_class = BeadForm
    template_name = "beadmodels/beads/bead_form.html"
    success_url = reverse_lazy("beadmodels:bead_list")

    def get_queryset(self):
        return Bead.objects.filter(creator=self.request.user)

    def form_valid(self, form):
        messages.success(self.request, "Perle mise à jour avec succès!")
        return super().form_valid(form)


class BeadDeleteView(LoginRequiredMixin, DeleteView):
    model = Bead
    success_url = reverse_lazy("beadmodels:bead_list")

    # Supprimer la référence à template_name car nous utilisons directement la confirmation JavaScript

    def delete(self, request, *args, **kwargs):
        messages.success(request, "Perle supprimée avec succès!")
        return super().delete(request, *args, **kwargs)

    def get_queryset(self):
        return Bead.objects.filter(creator=self.request.user)

    def delete(self, request, *args, **kwargs):
        messages.success(request, "Perle supprimée avec succès!")
        return super().delete(request, *args, **kwargs)


# Garder les vues fonctionnelles existantes temporairement
@login_required
def create_model(request):
    if request.method == "POST":
        form = BeadModelForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            model = form.save(commit=False)
            model.creator = request.user
            model.save()
            messages.success(request, "Votre modèle a été créé avec succès!")
            return redirect("beadmodels:model_detail", pk=model.pk)
    else:
        form = BeadModelForm(user=request.user)
    return render(request, "beadmodels/create_model.html", {"form": form})


def model_detail(request, pk):
    model = get_object_or_404(BeadModel, pk=pk)
    if not model.is_public and model.creator != request.user:
        messages.error(request, "Vous n'avez pas accès à ce modèle.")
        return redirect("beadmodels:home")

    transform_form = TransformModelForm()
    return render(
        request,
        "beadmodels/model_detail.html",
        {"model": model, "transform_form": transform_form},
    )


@login_required
def edit_model(request, pk):
    model = get_object_or_404(BeadModel, pk=pk)
    if model.creator != request.user:
        messages.error(request, "Vous n'avez pas le droit de modifier ce modèle.")
        return redirect("beadmodels:model_detail", pk=pk)

    if request.method == "POST":
        form = BeadModelForm(
            request.POST, request.FILES, instance=model, user=request.user
        )
        if form.is_valid():
            form.save()
            messages.success(request, "Votre modèle a été modifié avec succès!")
            return redirect("beadmodels:model_detail", pk=model.pk)
    else:
        form = BeadModelForm(instance=model, user=request.user)
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
    return redirect(f"{reverse('beadmodels:user_settings')}?tab=shapes")


@login_required
def edit_shape(request, shape_id):
    shape = get_object_or_404(BeadShape, id=shape_id, creator=request.user)

    if request.method == "POST":
        form = BeadShapeForm(request.POST, instance=shape)
        if form.is_valid():
            # Vérifier si une autre forme similaire existe déjà (exclure la forme actuelle)
            shape_name = form.cleaned_data.get("name")
            shape_type = form.cleaned_data.get("shape_type")

            # Vérifier si le nom a changé et s'il existe déjà
            if shape.name != shape_name:
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
                        "beadmodels/shapes/edit_shape.html",
                        {"form": form, "shape": shape},
                    )

            # Vérifier les dimensions pour les formes similaires
            duplicate_params = False
            existing_shapes = BeadShape.objects.filter(
                creator=request.user, shape_type=shape_type
            ).exclude(id=shape.id)

            if shape_type == "rectangle":
                width = form.cleaned_data.get("width")
                height = form.cleaned_data.get("height")
                for existing_shape in existing_shapes:
                    if (
                        existing_shape.width == width
                        and existing_shape.height == height
                    ):
                        duplicate_params = True
                        message = (
                            "Une forme rectangulaire avec ces dimensions existe déjà"
                        )
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
                messages.error(request, message)
                return render(
                    request,
                    "beadmodels/shapes/edit_shape.html",
                    {"form": form, "shape": shape},
                )

            # Tout est bon, enregistrer la forme
            form.save()
            messages.success(request, "La forme a été modifiée avec succès.")
            return redirect(f"{reverse('beadmodels:user_settings')}?tab=shapes")
    else:
        form = BeadShapeForm(instance=shape)

    return render(
        request, "beadmodels/shapes/edit_shape.html", {"form": form, "shape": shape}
    )


@login_required
@require_http_methods(["POST"])
def transform_image(request, pk):
    try:
        model = get_object_or_404(BeadModel, pk=pk)
        form = TransformModelForm(request.POST)

        if not form.is_valid():
            return JsonResponse(
                {
                    "success": False,
                    "message": "Formulaire invalide. Veuillez vérifier les champs.",
                },
                status=400,
            )

        if model.creator != request.user:
            return JsonResponse(
                {
                    "success": False,
                    "message": "Vous n'avez pas le droit de modifier ce modèle.",
                },
                status=403,
            )

        # Récupérer les paramètres du formulaire
        board = form.cleaned_data["board"]
        color_reduction = form.cleaned_data["color_reduction"]
        edge_detection = form.cleaned_data["edge_detection"]

        # Ouvrir l'image originale
        original_image = Image.open(model.original_image.path)
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")

        # Calculer les dimensions optimales
        target_width = board.width_pegs
        target_height = board.height_pegs

        # Ajouter une marge de 2 pixels
        margin = 2
        target_width_with_margin = target_width + 2 * margin
        target_height_with_margin = target_height + 2 * margin

        # Redimensionner l'image si nécessaire
        current_width, current_height = original_image.size
        scale = min(
            target_width_with_margin / current_width,
            target_height_with_margin / current_height,
        )

        if scale < 1:  # Image trop grande
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            resized_image = original_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
        elif scale > 1.5:  # Image trop petite
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            resized_image = original_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
        else:
            resized_image = original_image

        # Convertir en array numpy pour le traitement
        img_array = np.array(resized_image)

        # 1. Créer la grille
        grid_width = target_width
        grid_height = target_height
        cell_width = img_array.shape[1] // grid_width
        cell_height = img_array.shape[0] // grid_height

        # 2. Préparer les données pour le clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Créer un tableau pour stocker les couleurs moyennes de chaque cellule
        cell_colors = np.zeros((grid_height * grid_width, 3))

        # Calculer la couleur moyenne de chaque cellule
        for y in range(grid_height):
            for x in range(grid_width):
                y_start = y * cell_height
                y_end = (y + 1) * cell_height
                x_start = x * cell_width
                x_end = (x + 1) * cell_width

                # Extraire la cellule
                cell = img_array[y_start:y_end, x_start:x_end]

                # Calculer la couleur moyenne de la cellule
                if cell.size > 0:
                    cell_colors[y * grid_width + x] = np.mean(cell, axis=(0, 1))
                else:
                    cell_colors[y * grid_width + x] = [
                        255,
                        255,
                        255,
                    ]  # Blanc par défaut

        # Normaliser les couleurs
        scaler = StandardScaler()
        cell_colors_normalized = scaler.fit_transform(cell_colors)

        # 3. Appliquer K-means
        kmeans = KMeans(n_clusters=color_reduction, random_state=42)
        kmeans.fit(cell_colors_normalized)

        # 4. Créer l'image pixélisée
        clustered_colors = scaler.inverse_transform(kmeans.cluster_centers_)
        # S'assurer que les valeurs sont dans la plage [0, 255]
        clustered_colors = np.maximum(0, np.minimum(255, clustered_colors)).astype(
            np.uint8
        )

        # Créer l'image pixélisée
        pixelated_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for y in range(grid_height):
            for x in range(grid_width):
                cluster_idx = kmeans.labels_[y * grid_width + x]
                pixelated_image[y, x] = clustered_colors[cluster_idx]

        # Ajouter les contours si demandé
        if edge_detection:
            from scipy import ndimage

            # Convertir en niveaux de gris pour la détection des contours
            gray = np.dot(pixelated_image, [0.2989, 0.5870, 0.1140])

            # Appliquer les filtres Sobel
            sobel_h = ndimage.sobel(gray, axis=0)
            sobel_v = ndimage.sobel(gray, axis=1)

            # Combiner les gradients
            edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)

            # Normaliser et seuiller
            edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(
                np.uint8
            )
            edge_threshold = 50
            edges = edge_magnitude > edge_threshold

            # Superposer les contours
            pixelated_image[edges] = [0, 0, 0]

        # Créer l'image finale avec la grille
        final_image = Image.new(
            "RGB", (original_image.size[0], original_image.size[1]), (255, 255, 255)
        )
        final_pixels = final_image.load()

        # Calculer la taille de chaque cellule pour l'affichage
        display_cell_width = original_image.size[0] // grid_width
        display_cell_height = original_image.size[1] // grid_height

        # Dessiner les perles et la grille
        for y in range(grid_height):
            for x in range(grid_width):
                # Couleur de la perle
                bead_color = tuple(pixelated_image[y, x])

                # Dessiner la perle
                for py in range(display_cell_height):
                    for px in range(display_cell_width):
                        final_pixels[
                            x * display_cell_width + px, y * display_cell_height + py
                        ] = bead_color

                # Dessiner la grille horizontale
                if y < grid_height - 1:
                    for px in range(display_cell_width):
                        for g in range(2):  # Grille fine de 2 pixels
                            final_pixels[
                                x * display_cell_width + px,
                                (y + 1) * display_cell_height - 2 + g,
                            ] = (240, 240, 240)

                # Dessiner la grille verticale
                if x < grid_width - 1:
                    for py in range(display_cell_height):
                        for g in range(2):  # Grille fine de 2 pixels
                            final_pixels[
                                (x + 1) * display_cell_width - 2 + g,
                                y * display_cell_height + py,
                            ] = (240, 240, 240)

        # Sauvegarder le résultat temporairement
        temp_filename = f"temp_pattern_{pk}_{int(time.time())}.png"
        temp_path = os.path.join(settings.MEDIA_ROOT, "patterns", temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        final_image.save(temp_path)

        # Sauvegarder les informations du support utilisé
        model.board = board
        model.save()

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


def process_image_pixelization(
    image_array,
    grid_width,
    grid_height,
    color_reduction,
    use_available_colors=False,
    user=None,
):
    """
    Traite l'image en appliquant la pixelisation avec les paramètres spécifiés.
    """
    # Convertir la liste en tableau numpy s'il ne s'agit pas déjà d'un tableau
    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array)

    # Obtenir les dimensions de l'image
    height, width = image_array.shape[:2]

    # Calculer la taille de chaque cellule
    cell_height = height // grid_height
    cell_width = width // grid_width

    # Créer un tableau pour les couleurs moyennes
    cell_colors = np.zeros((grid_height * grid_width, 3))

    # Calculer la couleur moyenne de chaque cellule
    for y in range(grid_height):
        for x in range(grid_width):
            y_start = y * cell_height
            y_end = min((y + 1) * cell_height, height)
            x_start = x * cell_width
            x_end = min((x + 1) * cell_width, width)

            # Extraire la cellule
            cell = image_array[y_start:y_end, x_start:x_end]

            # Calculer la couleur moyenne
            if cell.size > 0:
                cell_colors[y * grid_width + x] = np.mean(cell, axis=(0, 1))
            else:
                cell_colors[y * grid_width + x] = [255, 255, 255]  # Blanc par défaut

    # Réduction des couleurs
    if use_available_colors and user:
        # Utiliser les couleurs des perles disponibles pour l'utilisateur
        available_beads = Bead.objects.filter(creator=user)
        if available_beads:
            bead_colors = np.array([[b.red, b.green, b.blue] for b in available_beads])
            color_centers = bead_colors
        else:
            color_centers = apply_kmeans(cell_colors, color_reduction)
    else:
        color_centers = apply_kmeans(cell_colors, color_reduction)

    # Créer l'image pixelisée
    pixelated_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Calculer les occurrences de chaque couleur
    color_counts = {}

    # Attribuer les couleurs des clusters à chaque cellule
    for i, color in enumerate(cell_colors):
        y = i // grid_width
        x = i % grid_width

        # Trouver la couleur la plus proche
        distances = np.sqrt(np.sum((color_centers - color) ** 2, axis=1))
        closest_color_idx = np.argmin(distances)
        pixelated_color = color_centers[closest_color_idx]
        pixelated_image[y, x] = pixelated_color

        # Incrémenter le compteur pour cette couleur
        color_tuple = tuple(map(int, pixelated_color))
        if color_tuple in color_counts:
            color_counts[color_tuple] += 1
        else:
            color_counts[color_tuple] = 1

    # Créer l'image finale avec grille
    cell_size = 20  # Taille fixe pour l'affichage
    final_width = grid_width * cell_size
    final_height = grid_height * cell_size

    final_image = Image.new("RGB", (final_width, final_height), (255, 255, 255))
    draw_pixels = final_image.load()

    # Dessiner les pixels et la grille
    for y in range(grid_height):
        for x in range(grid_width):
            color = tuple(map(int, pixelated_image[y, x]))

            # Dessiner le pixel
            for py in range(cell_size):
                for px in range(cell_size):
                    if px == 0 or px == cell_size - 1 or py == 0 or py == cell_size - 1:
                        # Bordure de la grille
                        draw_pixels[x * cell_size + px, y * cell_size + py] = (
                            200,
                            200,
                            200,
                        )
                    else:
                        # Pixel de couleur
                        draw_pixels[x * cell_size + px, y * cell_size + py] = color

    # Convertir en base64
    buffered = BytesIO()
    final_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Préparer la palette de couleurs
    palette = [
        {"color": f"rgb({r}, {g}, {b})", "count": count}
        for (r, g, b), count in sorted(
            color_counts.items(), key=lambda x: x[1], reverse=True
        )
    ]

    return {"image_base64": img_str, "palette": palette}


def process_image_for_wizard(image):
    """Traite l'image pour le wizard en utilisant les fonctionnalités du notebook."""
    # Ouvrir l'image
    if isinstance(image, str):
        pil_image = Image.open(image)
    else:
        pil_image = Image.open(image)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convertir en tableau numpy
    img_array = np.array(pil_image)

    # Etape 1: Centrer l'image (extraction de la zone d'intérêt)
    try:
        # Conversion de l'image en niveaux de gris
        if len(img_array.shape) == 2:
            gray_img = img_array.copy()
        else:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Appliquer le seuillage d'Otsu
        upper_threshold, thresh_img = cv2.threshold(
            gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        lower_threshold = 0.5 * upper_threshold

        # Détection des contours
        canny = cv2.Canny(img_array, lower_threshold, upper_threshold)
        pts = np.argwhere(canny > 0)

        # Trouver les points min et max
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)

        # Ajouter une marge
        BORDER = 10
        height, width = img_array.shape[:2]
        x1 = max(x1 - BORDER, 0)
        x2 = min(x2 + BORDER, width)
        y1 = max(y1 - BORDER, 0)
        y2 = min(y2 + BORDER, height)

        # Extraire la région
        cropped_region = img_array[y1:y2, x1:x2]
    except Exception as e:
        # En cas d'erreur, utiliser l'image entière
        cropped_region = img_array

    return {
        "image_array": cropped_region.tolist(),  # Convertir en liste pour la sérialisation JSON
        "image_base64": array_to_base64(cropped_region),
        "palette": [],  # Sera rempli plus tard dans le processus
    }


def array_to_base64(image_array):
    """Convertit un tableau numpy en chaîne base64 pour l'affichage dans le template."""
    image = Image.fromarray(np.uint8(image_array))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def apply_kmeans(colors, n_clusters):
    """Applique K-means pour réduire les couleurs"""
    # Normaliser les couleurs
    scaler = StandardScaler()
    normalized_colors = scaler.fit_transform(colors)

    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(normalized_colors)

    # Récupérer les centres des clusters
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers = np.clip(centers, 0, 255).astype(np.uint8)

    return centers


# HTMX views pour l'édition inline des perles


@login_required
def bead_edit_quantity_htmx(request, pk):
    """Vue HTMX pour afficher le formulaire d'édition de quantité de perle."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)

    context = {
        "bead": bead,
    }
    return render(request, "beadmodels/beads/htmx/bead_edit_quantity.html", context)


@login_required
def bead_update_quantity_htmx(request, pk):
    """Vue HTMX pour mettre à jour la quantité d'une perle."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)

    if request.method == "POST":
        try:
            quantity = int(request.POST.get("quantity", "0"))
            if quantity < 0:
                quantity = 0
            bead.quantity = quantity
            bead.save()

            # Récupérer le seuil d'alerte depuis les paramètres
            from django.conf import settings

            threshold = getattr(settings, "BEAD_LOW_QUANTITY_THRESHOLD", 20)

            context = {
                "bead": bead,
                "threshold": threshold,
            }
            return render(
                request, "beadmodels/beads/htmx/bead_quantity_display.html", context
            )
        except (ValueError, TypeError):
            # En cas d'erreur, retourner au formulaire d'édition
            context = {
                "bead": bead,
                "error": "La quantité doit être un nombre entier positif.",
            }
            return render(
                request, "beadmodels/beads/htmx/bead_edit_quantity.html", context
            )

    # Si la méthode n'est pas POST, retourner la vue de formulaire
    return bead_edit_quantity_htmx(request, pk)


@login_required
def bead_edit_color_htmx(request, pk):
    """Vue HTMX pour afficher le formulaire d'édition de couleur de perle."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)

    context = {
        "bead": bead,
    }
    return render(request, "beadmodels/beads/htmx/bead_edit_color.html", context)


@login_required
def bead_update_color_htmx(request, pk):
    """Vue HTMX pour mettre à jour la couleur d'une perle."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)

    if request.method == "POST":
        try:
            red = int(request.POST.get("red", "0"))
            green = int(request.POST.get("green", "0"))
            blue = int(request.POST.get("blue", "0"))

            # Valider les valeurs RGB (entre 0 et 255)
            red = max(0, min(255, red))
            green = max(0, min(255, green))
            blue = max(0, min(255, blue))

            # Mettre à jour les valeurs de couleur
            bead.red = red
            bead.green = green
            bead.blue = blue

            # Mettre à jour le nom de la perle (basé sur la couleur)
            bead.name = f"Perle #{red:02x}{green:02x}{blue:02x}"

            bead.save()

            # Retourner l'affichage de la couleur
            context = {
                "bead": bead,
            }
            return render(
                request, "beadmodels/beads/htmx/bead_color_display.html", context
            )
        except (ValueError, TypeError):
            # En cas d'erreur, retourner au formulaire d'édition
            context = {
                "bead": bead,
                "error": "Les valeurs de couleur doivent être des nombres entiers entre 0 et 255.",
            }
            return render(
                request, "beadmodels/beads/htmx/bead_edit_color.html", context
            )

    # Si la méthode n'est pas POST, retourner la vue de formulaire
    return bead_edit_color_htmx(request, pk)
