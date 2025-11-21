"""
Module pour le wizard de création de modèle.

Ce module contient l'implémentation du wizard à 3 étapes pour
la création de modèles de perles à repasser.
"""

import base64
import io
import logging
import uuid
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)
from django import forms
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.urls import reverse
from PIL import Image

from .forms import ImageUploadForm, ModelConfigurationForm
from .models import Bead, BeadBoard
from .wizards import LoginRequiredWizard, WizardStep


class ImageUploadStep(WizardStep):
    """Première étape: Chargement de l'image."""

    name = "Chargement de l'image"
    template = "beadmodels/model_creation/upload_image.html"
    form_class = ImageUploadForm
    position = 1

    def handle_get(self, **kwargs):
        """Gère l'affichage du formulaire de chargement d'image."""
        # Message de débogage pour vérifier si ce wizard est exécuté
        messages.info(
            self.wizard.request, "Démarrage du nouveau wizard de création de modèle"
        )

        # Vérifier si un model_id est fourni dans la requête
        model_id = self.wizard.request.GET.get("model_id")
        current_data = self.wizard.request.session.get(self.wizard.session_key, {})
        current_model_id = current_data.get("model_id")

        # On force le reset du wizard dans deux cas :
        # 1. Si explicitement demandé par ?reset=true
        # 2. Si un nouvel ID de modèle est fourni qui diffère de celui en session
        reset_wizard = self.wizard.request.GET.get("reset") == "true" or (
            model_id and str(current_model_id) != str(model_id)
        )

        if reset_wizard:
            self.wizard.reset_wizard()
            if model_id:
                messages.info(
                    self.wizard.request, f"Utilisation du modèle ID {model_id}"
                )
                # Réinitialiser mais conserver l'ID du modèle
                self.wizard.update_data({"model_id": model_id})

        wizard_data = self.wizard.get_data()

        # Si on est explicitement à l'étape 1, on réinitialise les données
        # sauf si on vient de faire un reset explicite
        current_step = self.wizard.get_current_step_number()
        if current_step == 1 and not reset_wizard:
            # Sauvegarder l'ID du modèle avant de réinitialiser
            model_id_to_keep = wizard_data.get("model_id")

            # Réinitialiser le wizard
            self.wizard.reset_wizard()

            # Restaurer l'ID du modèle si nécessaire
            if model_id_to_keep or model_id:
                self.wizard.update_data({"model_id": model_id or model_id_to_keep})

        # Initialiser le formulaire
        form = self.form_class()

        # Construire le contexte
        context = {"form": form, "wizard_step": self.position, "total_steps": 3}

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite le formulaire de chargement d'image."""
        form = self.form_class(self.wizard.request.POST, self.wizard.request.FILES)

        if form.is_valid():
            image = form.cleaned_data["image"]
            # Sauvegarde temporaire sur disque pour éviter surcharge session
            from .services.image_processing import save_temp_image

            stored_path = save_temp_image(image)
            self.wizard.update_data({"image_data": {"image_path": stored_path}})

            # Passer à l'étape suivante
            return self.wizard.go_to_next_step()
        else:
            # Afficher les erreurs du formulaire
            context = {"form": form, "wizard_step": self.position, "total_steps": 3}
            return self.render_template(context)


class ConfigurationStep(WizardStep):
    """Deuxième étape: Configuration et prévisualisation du modèle."""

    name = "Configuration du modèle"
    template = "beadmodels/model_creation/configure_model.html"
    form_class = ModelConfigurationForm
    position = 2

    def handle_get(self, **kwargs):
        """Gère l'affichage du formulaire de configuration."""
        wizard_data = self.wizard.get_data()
        image_data = wizard_data.get("image_data", {})

        if not image_data:
            messages.error(self.wizard.request, "Veuillez d'abord charger une image.")
            return redirect("beadmodels:model_creation_wizard")

        # Initialiser le formulaire avec les données existantes ou valeurs par défaut
        initial_data = {
            "color_reduction": wizard_data.get("color_reduction", 16),
            "use_available_colors": wizard_data.get("use_available_colors", False),
        }

        form = self.form_class(initial=initial_data)

        # Récupérer les boards disponibles
        boards = BeadBoard.objects.all()

        # Récupérer les formes de l'utilisateur
        from shapes.models import BeadShape

        user_shapes = BeadShape.objects.filter(
            creator=self.wizard.request.user
        ).order_by("name")

        # Définir les valeurs de couleurs disponibles
        color_values = [2, 4, 6, 8, 16, 24, 32]

        # Générer une prévisualisation avec les paramètres par défaut
        preview_image_base64 = self.generate_preview(wizard_data)

        # Construire le contexte
        context = {
            "form": form,
            "image_base64": image_data.get("image_base64", ""),
            "preview_image_base64": preview_image_base64,
            "boards": boards,
            "user_shapes": user_shapes,
            "has_grid_options": bool(user_shapes.exists()),
            "selected_shape_id": wizard_data.get("shape_id"),
            "color_values": color_values,
            "wizard_step": self.position,
            "total_steps": 3,
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite le formulaire de configuration."""
        wizard_data = self.wizard.get_data()

        # Vérifier si l'utilisateur a cliqué sur le bouton "Retour"
        if "previous_step" in self.wizard.request.POST:
            # Déléguer à la classe parente qui sait comment gérer la navigation
            return self.wizard.go_to_previous_step()

        # Prévisualisation déclenchée via HTMX
        if getattr(self.wizard.request, "htmx", False):
            # Mise à jour des paramètres
            shape_id = self.wizard.request.POST.get("shape_id", "")
            color_reduction = int(self.wizard.request.POST.get("color_reduction", 16))
            use_available_colors = (
                self.wizard.request.POST.get("use_available_colors") == "on"
            )

            # Mettre à jour les données du wizard
            self.wizard.update_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": color_reduction,
                    "use_available_colors": use_available_colors,
                }
            )

            # Générer la prévisualisation
            preview_image_base64 = self.generate_preview(self.wizard.get_data())

            # Renvoyer uniquement la partie prévisualisation
            context = {"preview_image_base64": preview_image_base64}

            html = render_to_string(
                "beadmodels/model_creation/preview_partial.html", context
            )
            return HttpResponse(html)

        # Si c'est une soumission normale du formulaire
        form = self.form_class(self.wizard.request.POST)

        if form.is_valid():
            # Récupérer l'ID de la forme sélectionnée
            shape_id = self.wizard.request.POST.get("shape_id", "")

            # Mettre à jour les données du wizard
            self.wizard.update_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": form.cleaned_data["color_reduction"],
                    "use_available_colors": form.cleaned_data["use_available_colors"],
                }
            )

            # Générer le modèle final
            final_model = self.generate_model(self.wizard.get_data())
            self.wizard.update_data({"final_model": final_model})

            # Passer à l'étape suivante
            return self.wizard.go_to_next_step()
        else:
            # Afficher les erreurs du formulaire
            preview_image_base64 = self.generate_preview(wizard_data)

            context = {
                "form": form,
                "image_base64": wizard_data.get("image_data", {}).get(
                    "image_base64", ""
                ),
                "preview_image_base64": preview_image_base64,
                "wizard_step": self.position,
                "total_steps": 3,
            }
            return self.render_template(context)

    def generate_preview(self, data):
        """Génère une prévisualisation pixelisée de l'image."""
        image_data = data.get("image_data", {})
        image_base64 = image_data.get("image_base64")
        image_path = image_data.get("image_path")
        if not image_base64 and not image_path:
            return ""

        # Paramètres de configuration
        shape_id = data.get("shape_id", "")
        color_reduction = data.get("color_reduction", 16)
        use_available_colors = data.get("use_available_colors", False)

        # Décoder l'image base64
        if image_path and not image_base64:
            # Charger depuis stockage
            from django.core.files.storage import default_storage

            with default_storage.open(image_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB") if img.mode != "RGB" else img
        else:
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes))

        # S'assurer que l'image est en mode RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Taille par défaut pour le modèle
        default_width = 29
        default_height = 29

        # Déterminer les dimensions de la grille en fonction de la forme choisie
        from shapes.models import BeadShape

        grid_width = 29  # Valeur par défaut
        grid_height = 29  # Valeur par défaut
        use_circle_mask = False
        circle_diameter = 0

        if shape_id:
            try:
                shape = BeadShape.objects.get(pk=shape_id)
                shape_type = shape.shape_type

                # Déterminer dimensions en fonction du type de forme
                if (
                    shape_type == "rectangle"
                    and shape.width is not None
                    and shape.height is not None
                ):
                    grid_width = shape.width
                    grid_height = shape.height
                    messages.info(
                        self.wizard.request,
                        f"Forme rectangle {grid_width}×{grid_height}",
                    )
                elif shape_type == "square" and shape.size is not None:
                    grid_width = shape.size
                    grid_height = shape.size
                    messages.info(
                        self.wizard.request, f"Forme carrée {grid_width}×{grid_height}"
                    )
                elif shape_type == "circle" and shape.diameter is not None:
                    grid_width = shape.diameter
                    grid_height = shape.diameter
                    use_circle_mask = True
                    circle_diameter = shape.diameter
                    messages.info(
                        self.wizard.request, f"Forme ronde diamètre {circle_diameter}"
                    )
            except BeadShape.DoesNotExist:
                messages.warning(
                    self.wizard.request,
                    "Forme non trouvée, utilisation des dimensions par défaut",
                )

        # Stratégie de redimensionnement en fonction de la forme
        if use_circle_mask:
            # Pour les cercles, on utilise une approche différente:
            # On redimensionne l'image pour remplir entièrement le cercle
            # puis on appliquera un masque après

            # Déterminer la dimension la plus petite de l'image source
            orig_width, orig_height = img.size
            orig_min_dim = min(orig_width, orig_height)

            # Calculer le facteur de redimensionnement pour que la plus petite dimension
            # corresponde au diamètre du cercle
            scale_factor = circle_diameter / orig_min_dim

            # Redimensionner l'image en préservant le ratio pour qu'elle couvre le cercle
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)

            # Créer une image carrée de la taille du cercle
            grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

            # Redimensionner l'image source
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Calculer les offsets pour centrer l'image redimensionnée
            offset_x = (grid_width - new_width) // 2
            offset_y = (grid_height - new_height) // 2

            # Coller l'image redimensionnée au centre
            grid_img.paste(img_resized, (offset_x, offset_y))

            # Message pour le débogage
            logger.debug(
                f"Redimensionnement circulaire: {new_width}x{new_height} offset=({offset_x},{offset_y})"
            )
        else:
            # Pour les formes rectangulaires, on préserve le ratio de l'image
            # mais on s'assure de ne pas dépasser les dimensions de la grille
            orig_width, orig_height = img.size
            orig_ratio = orig_width / orig_height

            # Approche intelligente pour le redimensionnement
            # On calcule les dimensions optimales pour remplir au maximum la forme
            # tout en préservant le ratio d'aspect

            # Calculer le ratio de la grille
            grid_ratio = grid_width / grid_height

            if orig_ratio > grid_ratio:
                # L'image est proportionnellement plus large que la grille
                # On ajuste sur la largeur
                target_width = grid_width
                target_height = int(grid_width / orig_ratio)
            else:
                # L'image est proportionnellement plus haute ou égale à la grille
                # On ajuste sur la hauteur
                target_height = grid_height
                target_width = int(grid_height * orig_ratio)

            # S'assurer que les dimensions sont au moins 1
            target_width = max(1, min(target_width, grid_width))
            target_height = max(1, min(target_height, grid_height))

            # Centrer l'image dans la grille
            offset_x = (grid_width - target_width) // 2
            offset_y = (grid_height - target_height) // 2

            # Redimensionner l'image
            img_resized = img.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

            # Créer une nouvelle image avec les dimensions de la grille
            grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

            # Coller l'image redimensionnée au centre
            grid_img.paste(img_resized, (offset_x, offset_y))

            # Message pour le débogage
            logger.debug(
                f"Redimensionnement standard: {target_width}x{target_height} offset=({offset_x},{offset_y})"
            )

        # Appliquer un masque circulaire si nécessaire
        if use_circle_mask and circle_diameter > 0:
            # Créer un masque circulaire plus efficace avec NumPy
            y, x = np.ogrid[:grid_height, :grid_width]
            center_x, center_y = grid_width // 2, grid_height // 2
            # Calculer la distance au carré par rapport au centre pour chaque pixel
            # Légèrement plus rapide que de calculer la racine carrée
            dist_squared = (x - center_x) ** 2 + (y - center_y) ** 2
            radius = circle_diameter // 2

            # Créer un masque binaire (True pour les pixels dans le cercle, False pour les autres)
            circle_mask = dist_squared <= radius**2

            # Créer une image blanche pour le fond
            white_bg = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            # Convertir en tableau 3D pour les opérations sur RGB
            mask_3d = np.stack([circle_mask] * 3, axis=2)

            # Créer le tableau final en utilisant le masque:
            # - Pour les pixels dans le cercle (True), prendre l'image originale
            # - Pour les pixels hors du cercle (False), utiliser le fond blanc
            masked_array = np.where(mask_3d, np.array(grid_img), white_bg)

            # Convertir en image PIL
            grid_img = Image.fromarray(masked_array.astype(np.uint8))

            # Message pour le débogage
            logger.debug(f"Masque circulaire appliqué rayon={radius}")

        # Assurons-nous que l'image est exactement de la taille de la grille pour le clustering
        img_resized = grid_img.resize(
            (grid_width, grid_height), Image.Resampling.LANCZOS
        )
        img_array = np.array(img_resized)
        pixels = img_array.reshape(-1, 3)

        # Réduction de couleurs via service partagé
        from .services.image_processing import reduce_colors

        user_colors = None
        if use_available_colors:
            user_beads = Bead.objects.filter(creator=self.wizard.request.user)
            if user_beads.exists():
                user_colors = np.array(
                    [[bead.red, bead.green, bead.blue] for bead in user_beads]
                )
        reduced_pixels = reduce_colors(img_array, color_reduction, user_colors)

        # Convertir en image PIL
        reduced_img = Image.fromarray(reduced_pixels.astype("uint8"))

        # Créer une image agrandie avec une grille
        cell_size = 10  # Taille de chaque cellule (perle)
        grid_img_width = grid_width * cell_size
        grid_img_height = grid_height * cell_size
        grid_img = Image.new("RGB", (grid_img_width, grid_img_height), (255, 255, 255))

        # Récupérer la forme personnalisée si spécifiée
        shape_mask = None
        if shape_id:
            try:
                from shapes.models import BeadShape

                shape = BeadShape.objects.get(pk=shape_id)
                # Ici, on pourrait utiliser la forme pour créer un masque
                # Pour l'instant, on se contente de noter la forme utilisée
                messages.info(self.wizard.request, f"Forme utilisée : {shape.name}")
            except Exception as e:
                messages.error(
                    self.wizard.request, f"Erreur lors du chargement de la forme: {e}"
                )

        # Dessiner chaque perle avec une bordure
        draw_pixels = np.array(grid_img)

        # Assurons-nous que reduced_pixels a bien les dimensions attendues
        if reduced_pixels.shape[:2] != (grid_height, grid_width):
            # Si les dimensions ne correspondent pas, redimensionner l'image réduite
            reduced_img = Image.fromarray(reduced_pixels.astype("uint8"))
            reduced_img_resized = reduced_img.resize(
                (grid_width, grid_height), Image.Resampling.LANCZOS
            )
            reduced_pixels = np.array(reduced_img_resized)

        # Maintenant dessiner les perles
        # Si c'est une forme circulaire, nous devons aussi masquer les perles hors du cercle
        if use_circle_mask:
            # Créer un masque pour déterminer quelles perles sont dans le cercle
            center_x, center_y = grid_width // 2, grid_height // 2
            radius = circle_diameter // 2

            # Pour chaque position de perle, vérifier si elle est dans le cercle
            for y in range(grid_height):
                for x in range(grid_width):
                    # Calculer la distance au centre
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                    # Si la perle est dans le cercle, la dessiner
                    if dist <= radius:
                        try:
                            color = reduced_pixels[y, x]

                            # Dessiner la perle avec une bordure
                            for py in range(cell_size):
                                for px in range(cell_size):
                                    cell_x = x * cell_size + px
                                    cell_y = y * cell_size + py

                                    if (
                                        px == 0
                                        or px == cell_size - 1
                                        or py == 0
                                        or py == cell_size - 1
                                    ):
                                        # Bordure de la grille (gris clair)
                                        draw_pixels[cell_y, cell_x] = (200, 200, 200)
                                    else:
                                        # Pixel de couleur
                                        draw_pixels[cell_y, cell_x] = color
                        except IndexError as e:
                            # En cas d'erreur d'index, afficher un message et continuer
                            messages.error(
                                self.wizard.request,
                                f"Erreur d'accès à l'index [{y},{x}] dans une image de dimensions {reduced_pixels.shape[:2]}",
                            )
        else:
            # Dessin standard pour les formes rectangulaires
            for y in range(grid_height):
                for x in range(grid_width):
                    try:
                        color = reduced_pixels[y, x]

                        # Dessin normal pour une grille carrée
                        for py in range(cell_size):
                            for px in range(cell_size):
                                if (
                                    px == 0
                                    or px == cell_size - 1
                                    or py == 0
                                    or py == cell_size - 1
                                ):
                                    # Bordure de la grille (gris clair)
                                    draw_pixels[
                                        y * cell_size + py, x * cell_size + px
                                    ] = (
                                        200,
                                        200,
                                        200,
                                    )
                                else:
                                    # Pixel de couleur
                                    draw_pixels[
                                        y * cell_size + py, x * cell_size + px
                                    ] = color
                    except IndexError as e:
                        # En cas d'erreur d'index, afficher un message et continuer
                        messages.error(
                            self.wizard.request,
                            f"Erreur d'accès à l'index [{y},{x}] dans une image de dimensions {reduced_pixels.shape[:2]}",
                        )

        # Convertir l'array en image PIL
        grid_img = Image.fromarray(draw_pixels)

        # Convertir en base64
        preview_bytes = io.BytesIO()
        grid_img.save(preview_bytes, format="PNG")
        preview_bytes.seek(0)
        preview_base64 = base64.b64encode(preview_bytes.read()).decode("utf-8")

        return preview_base64

    def generate_model(self, wizard_data):
        """Génère le modèle final."""
        # Cette méthode est similaire à generate_preview mais avec la taille finale
        preview_base64 = self.generate_preview(wizard_data)

        # Extraire les informations supplémentaires nécessaires
        color_reduction = wizard_data.get("color_reduction", 16)

        # Déterminer les dimensions de la forme
        shape_id = wizard_data.get("shape_id", "")
        grid_width = 29  # Valeur par défaut
        grid_height = 29  # Valeur par défaut

        # Récupérer les dimensions de la forme si une forme est sélectionnée
        if shape_id:
            from shapes.models import BeadShape

            try:
                shape = BeadShape.objects.get(pk=shape_id)
                if (
                    shape.shape_type == "rectangle"
                    and shape.width is not None
                    and shape.height is not None
                ):
                    grid_width = shape.width
                    grid_height = shape.height
                elif shape.shape_type == "square" and shape.size is not None:
                    grid_width = shape.size
                    grid_height = shape.size
                elif shape.shape_type == "circle" and shape.diameter is not None:
                    grid_width = shape.diameter
                    grid_height = shape.diameter

                # Logguer les dimensions utilisées
                messages.info(
                    self.wizard.request,
                    f"Modèle final créé avec dimensions: {grid_width}×{grid_height} (forme: {shape.name})",
                )
            except BeadShape.DoesNotExist:
                messages.warning(
                    self.wizard.request,
                    "Forme non trouvée, utilisation des dimensions par défaut",
                )

        # Calculer le nombre total de perles
        # Pour les cercles, ajuster pour ne compter que les perles dans le cercle
        if (
            shape_id
            and BeadShape.objects.filter(pk=shape_id, shape_type="circle").exists()
        ):
            # Pour un cercle, le nombre de perles est approximativement π × r²
            shape = BeadShape.objects.get(pk=shape_id)
            radius = shape.diameter / 2 if shape.diameter else 0

            # Méthode plus précise: compter le nombre de points dans le cercle
            # en utilisant l'équation du cercle x² + y² ≤ r²
            bead_count = 0
            center_x, center_y = grid_width // 2, grid_height // 2

            for y in range(grid_height):
                for x in range(grid_width):
                    # Vérifier si ce point est dans le cercle
                    if ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius**2):
                        bead_count += 1

            total_beads = bead_count
            messages.info(
                self.wizard.request,
                f"Nombre de perles dans le cercle: {total_beads} (rayon {radius})",
            )
        else:
            # Pour les rectangles et carrés, c'est simplement largeur × hauteur
            total_beads = grid_width * grid_height

        # Palette via service
        from .services.image_processing import compute_palette

        palette = compute_palette(preview_base64, total_beads)

        return {
            "image_base64": preview_base64,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "shape_id": wizard_data.get("shape_id"),
            "color_reduction": color_reduction,
            "total_beads": total_beads,
            "palette": palette,
        }


class SaveStep(WizardStep):
    """Troisième étape: Sauvegarde du modèle."""

    name = "Finalisation et Sauvegarde"
    template = "beadmodels/model_creation/save_step.html"
    position = 3

    def handle_get(self, **kwargs):
        """Affiche le résultat final et les options de sauvegarde."""
        wizard_data = self.wizard.get_data()
        final_model = wizard_data.get("final_model", {})

        if not final_model:
            messages.error(self.wizard.request, "Veuillez configurer votre modèle.")
            return redirect("beadmodels:model_creation_wizard")

        # Récupérer les informations de la forme si applicable
        shape_name = "Standard"
        shape_type = "rectangle"
        if final_model.get("shape_id"):
            from shapes.models import BeadShape

            try:
                shape = BeadShape.objects.get(pk=final_model.get("shape_id"))
                shape_name = shape.name
                shape_type = shape.shape_type
            except BeadShape.DoesNotExist:
                pass

        # Récupérer les perles disponibles pour l'utilisateur
        user_beads = Bead.objects.filter(creator=self.wizard.request.user)

        # Créer un formulaire pour enregistrer le modèle
        from .forms import BeadModelForm

        # Créer une classe de formulaire spécifique pour le wizard
        class WizardBeadModelForm(BeadModelForm):
            """Version améliorée du formulaire BeadModelForm pour le wizard."""

            tags = forms.CharField(
                required=False,
                widget=forms.TextInput(
                    attrs={
                        "class": "form-control",
                        "placeholder": "Séparez les tags par des virgules",
                        "data-role": "tagsinput",
                    }
                ),
                help_text="Ajoutez des tags pour retrouver facilement votre modèle",
            )

            favorite = forms.BooleanField(
                required=False,
                widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
                label="Ajouter aux favoris",
                help_text="Marquez ce modèle comme favori pour y accéder rapidement",
            )

            class Meta(BeadModelForm.Meta):
                fields = ["name", "description", "board", "is_public"]
                widgets = {
                    "name": forms.TextInput(
                        attrs={
                            "class": "form-control",
                            "placeholder": "Nom du modèle",
                            "id": "model-name-input",
                            "hx-trigger": "keyup changed delay:300ms",
                            "hx-target": "#model-preview-title",
                        }
                    ),
                    "description": forms.Textarea(
                        attrs={
                            "class": "form-control",
                            "rows": 3,
                            "placeholder": "Description (optionnelle)",
                        }
                    ),
                    "board": forms.Select(
                        attrs={
                            "class": "form-select",
                            "hx-trigger": "change",
                            "hx-target": "#board-preview",
                        }
                    ),
                    "is_public": forms.CheckboxInput(
                        attrs={"class": "form-check-input"}
                    ),
                }
                labels = {
                    "name": "Nom du modèle",
                    "description": "Description (optionnelle)",
                    "board": "Support de perles",
                    "is_public": "Rendre ce modèle public",
                }
                help_texts = {
                    "is_public": "Si coché, les autres utilisateurs pourront voir ce modèle",
                }

        # Obtenir la date actuelle formatée en français pour le nom par défaut
        import locale

        try:
            locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
        except locale.Error:
            pass  # Si la locale n'est pas disponible, on continue avec celle par défaut

        default_name = f"Modèle du {datetime.now().strftime('%d %B %Y')}"

        # Trouver le board par défaut ou celui précédemment utilisé
        default_board = None
        if final_model.get("board_id"):
            try:
                default_board = BeadBoard.objects.get(pk=final_model.get("board_id"))
            except BeadBoard.DoesNotExist:
                default_board = BeadBoard.objects.first()
        else:
            default_board = BeadBoard.objects.first()

        # Initialiser le formulaire avec des valeurs par défaut
        initial_data = {
            "name": default_name,
            "is_public": False,
            "board": default_board.pk if default_board else None,
        }

        form = WizardBeadModelForm(initial=initial_data)

        # Analyser la palette et trouver des correspondances avec les perles de l'utilisateur
        palette = final_model.get("palette", [])
        palette_with_matches = []

        for color_item in palette:
            # Extraire les valeurs RGB de la chaîne "rgb(r, g, b)"
            import re

            rgb_match = re.search(
                r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color_item["color"]
            )

            if rgb_match:
                r, g, b = map(int, rgb_match.groups())

                # Rechercher des perles similaires dans la collection de l'utilisateur
                color_matches = []
                if user_beads:
                    import numpy as np

                    for bead in user_beads:
                        # Calculer la distance euclidienne entre les couleurs
                        distance = np.sqrt(
                            (bead.red - r) ** 2
                            + (bead.green - g) ** 2
                            + (bead.blue - b) ** 2
                        )

                        # Si la distance est inférieure à un seuil (ex: 30), considérer comme similaire
                        if distance < 30:
                            color_matches.append(
                                {
                                    "name": bead.name,
                                    "color": f"rgb({bead.red}, {bead.green}, {bead.blue})",
                                    "hex": f"#{bead.red:02x}{bead.green:02x}{bead.blue:02x}",
                                    "distance": int(distance),
                                    "quantity": bead.quantity,
                                }
                            )

                # Trier les correspondances par distance (plus proche d'abord)
                color_matches.sort(key=lambda x: x["distance"])

                # Limiter à 3 correspondances maximum
                color_matches = color_matches[:3]

                # Ajouter les matches à l'élément de palette
                color_item["matches"] = color_matches

            palette_with_matches.append(color_item)

        # Construire le contexte
        context = {
            "image_base64": final_model.get("image_base64", ""),
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "shape_id": final_model.get("shape_id"),
            "shape_name": shape_name,
            "shape_type": shape_type,
            "total_beads": final_model.get("total_beads", 0),
            "palette": palette_with_matches,
            "beads_count": len(final_model.get("palette", [])),
            "wizard_step": self.position,
            "total_steps": 3,
            "form": form,
            "boards": BeadBoard.objects.all(),
            "user_has_beads": user_beads.exists() if user_beads else False,
            "default_board": default_board,
            "board_preview_url": (
                f"/media/board_previews/{default_board.pk}.png"
                if default_board
                else None
            ),
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Gère les actions finales avec une meilleure expérience utilisateur."""
        from django.contrib import messages

        from .forms import BeadModelForm

        # Récupérer les données du modèle final
        wizard_data = self.wizard.get_data()
        final_model = wizard_data.get("final_model", {})

        if not final_model:
            messages.error(self.wizard.request, "Erreur: données du modèle manquantes.")
            return redirect("beadmodels:model_creation_wizard")

        # Vérifier le type de soumission
        # Si l'utilisateur veut simplement télécharger l'image
        if "download" in self.wizard.request.POST:
            download_format = self.wizard.request.POST.get("download_format", "png")
            messages.info(
                self.wizard.request,
                f"Préparation du téléchargement au format {download_format}...",
            )
            return self.download_model(format=download_format)

        # Si l'utilisateur veut télécharger les instructions
        elif "download_instructions" in self.wizard.request.POST:
            messages.info(self.wizard.request, "Préparation des instructions...")
            return self.generate_instructions()

        # Si c'est une demande de retour à l'étape précédente
        elif "previous_step" in self.wizard.request.POST:
            return self.wizard.go_to_previous_step()

        # Si l'utilisateur souhaite sauvegarder le modèle en BDD (action par défaut)
        else:
            # Utiliser notre formulaire spécifique pour le wizard
            class WizardBeadModelForm(BeadModelForm):
                """Version améliorée du formulaire BeadModelForm pour le wizard."""

                tags = forms.CharField(required=False)
                favorite = forms.BooleanField(required=False)

                class Meta(BeadModelForm.Meta):
                    fields = ["name", "description", "board", "is_public"]

            form = WizardBeadModelForm(self.wizard.request.POST)

            if form.is_valid():
                # Créer une nouvelle instance de BeadModel sans la sauvegarder encore
                new_model = form.save(commit=False)
                new_model.creator = self.wizard.request.user

                # Sauvegarder l'image originale
                # L'image originale est celle que l'utilisateur a téléchargée à l'étape 1
                image_data = wizard_data.get("image_data", {})
                # Convertir image d'origine en base64 si seulement path présent
                if not image_data.get("image_base64") and image_data.get("image_path"):
                    from .services.image_processing import file_to_base64

                    image_data["image_base64"] = file_to_base64(
                        image_data.get("image_path")
                    )

                if image_data.get("image_base64"):
                    # Convertir la base64 en fichier
                    image_bytes = base64.b64decode(image_data.get("image_base64"))
                    image_format = image_data.get("image_format", "PNG")

                    from django.core.files.base import ContentFile

                    image_name = f"original_{uuid.uuid4()}.{image_format.lower()}"
                    new_model.original_image.save(
                        image_name, ContentFile(image_bytes), save=False
                    )

                # Sauvegarder le motif pixelisé
                if not final_model.get("image_base64") and final_model.get(
                    "image_path"
                ):
                    from .services.image_processing import file_to_base64

                    final_model["image_base64"] = file_to_base64(
                        final_model["image_path"]
                    )

                if final_model.get("image_base64"):
                    # Convertir la base64 en fichier
                    pattern_bytes = base64.b64decode(final_model.get("image_base64"))

                    from django.core.files.base import ContentFile

                    pattern_name = f"pattern_{uuid.uuid4()}.png"
                    new_model.bead_pattern.save(
                        pattern_name, ContentFile(pattern_bytes), save=False
                    )

                # Conserver les métadonnées du modèle
                metadata = {
                    "grid_width": final_model.get("grid_width", 29),
                    "grid_height": final_model.get("grid_height", 29),
                    "shape_id": final_model.get("shape_id"),
                    "color_reduction": final_model.get("color_reduction", 16),
                    "total_beads": final_model.get("total_beads", 0),
                    "palette": final_model.get("palette", []),
                }

                # Stocker les métadonnées si le champ existe
                if hasattr(new_model, "metadata"):
                    new_model.metadata = metadata

                # Traiter les tags (si le modèle BeadModel supporte cette fonctionnalité)
                # tags = form.cleaned_data.get("tags", "")
                # if tags and hasattr(new_model, "tags"):
                #     tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                #     new_model.tags = tag_list

                # Marquer comme favori (si le modèle BeadModel supporte cette fonctionnalité)
                # favorite = form.cleaned_data.get("favorite", False)
                # if favorite and hasattr(new_model, "is_favorite"):
                #     new_model.is_favorite = True

                # Sauvegarder le modèle
                new_model.save()

                # Ajouter un message de succès avec plus d'informations
                messages.success(
                    self.wizard.request,
                    f"Félicitations ! Votre modèle '{new_model.name}' a été créé avec succès ! "
                    f"Il contient {final_model.get('total_beads', 0)} perles et "
                    f"{len(final_model.get('palette', []))} couleurs différentes.",
                )

                # Réinitialiser le wizard et rediriger vers la page de détail du modèle
                self.wizard.reset_wizard()
                return redirect("beadmodels:model_detail", pk=new_model.pk)
            else:
                # En cas d'erreur dans le formulaire
                # Récupérer les informations de la forme si applicable
                shape_name = "Standard"
                shape_type = "rectangle"
                if final_model.get("shape_id"):
                    from shapes.models import BeadShape

                    try:
                        shape = BeadShape.objects.get(pk=final_model.get("shape_id"))
                        shape_name = shape.name
                        shape_type = shape.shape_type
                    except BeadShape.DoesNotExist:
                        pass

                # Construire le contexte avec le formulaire contenant des erreurs
                context = {
                    "image_base64": final_model.get("image_base64", ""),
                    "grid_width": final_model.get("grid_width", 29),
                    "grid_height": final_model.get("grid_height", 29),
                    "shape_id": final_model.get("shape_id"),
                    "shape_name": shape_name,
                    "shape_type": shape_type,
                    "total_beads": final_model.get("total_beads", 0),
                    "palette": final_model.get("palette", []),
                    "beads_count": len(final_model.get("palette", [])),
                    "wizard_step": self.position,
                    "total_steps": 3,
                    "form": form,
                    "boards": BeadBoard.objects.all(),
                }

                return self.render_template(context)

        # Par défaut, réafficher simplement le template
        return self.render_template()

    def download_model(self, format="png"):
        """Prépare le téléchargement du modèle dans différents formats."""
        wizard_data = self.wizard.get_data()
        final_model = wizard_data.get("final_model", {})

        if not final_model:
            messages.error(self.wizard.request, "Erreur lors du téléchargement.")
            return self.render_template()

        # Récupérer l'image en base64
        image_base64 = final_model.get("image_base64", "")

        if not image_base64:
            messages.error(self.wizard.request, "Image introuvable.")
            return self.render_template()

        # Décoder l'image
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes))

        # Préparer l'image dans le format demandé
        output = io.BytesIO()

        if format.lower() == "pdf":
            # Pour créer un PDF, on peut utiliser une bibliothèque comme reportlab
            # Cette fonctionnalité nécessiterait d'installer reportlab
            # Pour l'instant, on reste en PNG
            format = "png"
            messages.warning(
                self.wizard.request, "Export PDF non disponible. Téléchargement en PNG."
            )

        if format.lower() == "jpg" or format.lower() == "jpeg":
            # Convertir en JPEG si demandé
            if img.mode == "RGBA":
                # Le JPEG ne supporte pas la transparence, on convertit en RGB
                img = img.convert("RGB")
            img.save(output, format="JPEG", quality=90)
            content_type = "image/jpeg"
            extension = "jpg"
        else:
            # Par défaut, format PNG
            img.save(output, format="PNG")
            content_type = "image/png"
            extension = "png"

        output.seek(0)

        # Créer un nom de fichier basé sur la date
        from datetime import datetime

        now = datetime.now()
        filename = f"modele_perles_{now.strftime('%Y%m%d_%H%M%S')}.{extension}"

        # Préparer la réponse HTTP pour le téléchargement
        response = HttpResponse(output.getvalue(), content_type=content_type)
        response["Content-Disposition"] = f'attachment; filename="{filename}"'

        return response

    def generate_instructions(self):
        """Génère un PDF avec les instructions pour réaliser le modèle."""
        wizard_data = self.wizard.get_data()
        final_model = wizard_data.get("final_model", {})

        if not final_model:
            messages.error(
                self.wizard.request, "Erreur lors de la génération des instructions."
            )
            return self.render_template()

        # Récupérer l'image en base64
        image_base64 = final_model.get("image_base64", "")

        if not image_base64:
            messages.error(self.wizard.request, "Image introuvable.")
            return self.render_template()

        # Générer un HTML avec les instructions
        context = {
            "image_base64": image_base64,
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "total_beads": final_model.get("total_beads", 0),
            "palette": final_model.get("palette", []),
            "date": datetime.now().strftime("%d/%m/%Y"),
            "model_name": self.wizard.request.POST.get("name", "Nouveau modèle"),
        }

        # Générer le HTML pour les instructions
        instructions_html = render_to_string(
            "beadmodels/model_creation/instructions.html", context
        )

        # Pour l'instant, on retourne simplement le HTML car la génération de PDF
        # nécessiterait l'ajout d'une dépendance comme WeasyPrint ou xhtml2pdf
        response = HttpResponse(instructions_html, content_type="text/html")
        response["Content-Disposition"] = 'attachment; filename="instructions.html"'

        return response


class ModelCreationWizard(LoginRequiredWizard):
    """Wizard complet de création de modèle à 3 étapes."""

    name = "Création de modèle"
    # Assurez-vous que les étapes sont dans le bon ordre et que SaveStep est bien la dernière
    steps = [ImageUploadStep, ConfigurationStep, SaveStep]
    session_key = "model_creation_wizard"

    def get_url_name(self):
        """Renvoie le nom d'URL du wizard."""
        return "beadmodels:model_creation_wizard"

    def get_redirect_kwargs(self):
        """Récupère les paramètres à conserver lors des redirections."""
        # Conserver l'ID du modèle dans les redirections
        model_id = self.get_data().get("model_id")
        if model_id:
            return {"model_id": model_id}
        return {}

    def dispatch(self, request, *args, **kwargs):
        """Vérifie si une réinitialisation du wizard est demandée."""
        # Réinitialiser le wizard seulement si demandé explicitement
        if "reset" in request.GET and request.GET.get("reset") == "true":
            self.reset_wizard()
            # Forcer le retour à l'étape 1
            request.session[f"{self.session_key}_step"] = 1
            messages.info(
                request,
                "Assistant réinitialisé. Vous pouvez commencer un nouveau modèle.",
            )
            return redirect(reverse(self.get_url_name()))

        # Laisser la classe parente gérer la requête normalement
        return super().dispatch(request, *args, **kwargs)

    def finish_wizard(self):
        """Action finale lorsque le wizard est terminé."""
        self.reset_wizard()
        # Rediriger vers la liste des modèles de l'utilisateur
        return redirect("beadmodels:my_models")
