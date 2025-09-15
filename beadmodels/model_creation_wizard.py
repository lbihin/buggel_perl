"""
Module pour le wizard de création de modèle.

Ce module contient l'implémentation du wizard à 3 étapes pour
la création de modèles de perles à repasser.
"""

import base64
import io

import numpy as np
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.urls import reverse
from PIL import Image
from sklearn.cluster import KMeans

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

            # Traitement de l'image
            img = Image.open(image)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img.format or "PNG")
            img_bytes.seek(0)

            # Sauvegarder l'image en base64 pour la réutiliser dans les étapes suivantes
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

            # Mettre à jour les données du wizard
            self.wizard.update_data(
                {
                    "image_data": {
                        "image_base64": img_base64,
                        "image_format": img.format or "PNG",
                    }
                }
            )

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

        # Vérifier si c'est une requête HTMX pour une prévisualisation
        if self.wizard.request.headers.get("HX-Request") == "true":
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
        image_base64 = image_data.get("image_base64", "")

        if not image_base64:
            return ""

        # Paramètres de configuration
        shape_id = data.get("shape_id", "")
        color_reduction = data.get("color_reduction", 16)
        use_available_colors = data.get("use_available_colors", False)

        # Décoder l'image base64
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

        # Préserver le ratio de l'image originale
        orig_width, orig_height = img.size
        orig_ratio = orig_width / orig_height

        # Calculer les nouvelles dimensions qui préservent le ratio
        target_width = grid_width
        target_height = grid_height

        if orig_width > orig_height:
            # Image horizontale
            target_height = int(grid_width / orig_ratio)
            if target_height > grid_height:
                target_height = grid_height
                target_width = int(target_height * orig_ratio)
        else:
            # Image verticale ou carrée
            target_width = int(grid_height * orig_ratio)
            if target_width > grid_width:
                target_width = grid_width
                target_height = int(target_width / orig_ratio)

        # S'assurer que les dimensions sont au moins 1
        target_width = max(1, target_width)
        target_height = max(1, target_height)

        # Centrer l'image dans la grille
        offset_x = (grid_width - target_width) // 2
        offset_y = (grid_height - target_height) // 2

        # Redimensionner l'image
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Créer une nouvelle image avec les dimensions de la grille
        grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        # Coller l'image redimensionnée au centre
        grid_img.paste(img, (offset_x, offset_y))

        # Appliquer un masque circulaire si nécessaire
        if use_circle_mask and circle_diameter > 0:
            # Créer un masque circulaire
            mask_array = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            center_x = grid_width // 2
            center_y = grid_height // 2
            radius = circle_diameter // 2

            for y in range(grid_height):
                for x in range(grid_width):
                    # Calculer la distance au centre
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist > radius:
                        # Mettre en blanc les pixels en dehors du cercle
                        mask_array[y, x] = [255, 255, 255]

            # Convertir en image PIL
            mask_img = Image.fromarray(mask_array)

            # Appliquer le masque à l'image redimensionnée
            grid_img = Image.composite(grid_img, mask_img, mask_img.convert("L"))

        # Assurons-nous que l'image est exactement de la taille de la grille pour le clustering
        img_resized = grid_img.resize(
            (grid_width, grid_height), Image.Resampling.LANCZOS
        )
        img_array = np.array(img_resized)
        pixels = img_array.reshape(-1, 3)

        # Réduction de couleurs avec K-means
        kmeans = KMeans(n_clusters=color_reduction, random_state=0, n_init="auto")
        kmeans.fit(pixels)

        # Remplacer chaque pixel par la couleur du centroïde
        new_colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        reduced_pixels = new_colors[labels].reshape(img_array.shape)

        # Si on utilise les couleurs disponibles
        if use_available_colors:
            # Récupérer les couleurs disponibles pour l'utilisateur
            user_beads = Bead.objects.filter(creator=self.wizard.request.user)
            if user_beads.exists():
                user_colors = np.array(
                    [[bead.red, bead.green, bead.blue] for bead in user_beads]
                )

                # Pour chaque couleur réduite, trouver la couleur disponible la plus proche
                for i, color in enumerate(new_colors):
                    # Calculer la distance euclidienne entre cette couleur et les couleurs disponibles
                    distances = np.sqrt(np.sum((user_colors - color) ** 2, axis=1))
                    closest_color_idx = np.argmin(distances)
                    new_colors[i] = user_colors[closest_color_idx]

                # Recréer l'image avec les couleurs disponibles
                reduced_pixels = new_colors[labels].reshape(img_array.shape)

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
                                draw_pixels[y * cell_size + py, x * cell_size + px] = (
                                    200,
                                    200,
                                    200,
                                )
                            else:
                                # Pixel de couleur
                                draw_pixels[y * cell_size + py, x * cell_size + px] = (
                                    color
                                )
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
            # Approximation du nombre de perles dans un cercle: π × r²
            shape = BeadShape.objects.get(pk=shape_id)
            radius = shape.diameter / 2 if shape.diameter else 0
            total_beads = int(3.14159 * radius * radius)
        else:
            total_beads = grid_width * grid_height

        # Calculer la palette de couleurs
        image_bytes = base64.b64decode(preview_base64)
        img = Image.open(io.BytesIO(image_bytes))

        # S'assurer que l'image est en mode RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img)

        # Aplatir l'image pour identifier les couleurs uniques
        pixels = img_array.reshape(-1, 3)
        unique_colors, counts = np.unique(
            pixels.reshape(-1, 3), axis=0, return_counts=True
        )

        # Créer la palette sous forme de liste de dictionnaires
        palette = []
        for i, color in enumerate(unique_colors):
            r, g, b = color
            count = counts[i]
            percentage = (count / total_beads) * 100
            palette.append(
                {
                    "color": f"rgb({r}, {g}, {b})",
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "count": int(count),
                    "percentage": round(percentage, 1),
                }
            )

        # Trier par nombre de perles
        palette = sorted(palette, key=lambda x: x["count"], reverse=True)

        return {
            "image_base64": preview_base64,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "shape_id": wizard_data.get("shape_id"),
            "color_reduction": color_reduction,
            "total_beads": total_beads,
            "palette": palette,
        }


class ResultStep(WizardStep):
    """Troisième étape: Affichage et téléchargement du modèle."""

    name = "Résultat final"
    template = "beadmodels/model_creation/result.html"
    position = 3

    def handle_get(self, **kwargs):
        """Affiche le résultat final."""
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

        # Construire le contexte
        context = {
            "image_base64": final_model.get("image_base64", ""),
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "shape_id": final_model.get("shape_id"),
            "shape_name": shape_name,
            "shape_type": shape_type,
            "total_beads": final_model.get("total_beads", 0),
            "palette": final_model.get("palette", []),
            "wizard_step": self.position,
            "total_steps": 3,
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Gère les actions finales."""
        if "download" in self.wizard.request.POST:
            return self.download_model()

        return self.render_template()

    def download_model(self):
        """Prépare le téléchargement du modèle."""
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

        # Préparer la réponse HTTP pour le téléchargement
        response = HttpResponse(image_bytes, content_type="image/png")
        response["Content-Disposition"] = 'attachment; filename="modele_perles.png"'

        return response


class ModelCreationWizard(LoginRequiredWizard):
    """Wizard complet de création de modèle à 3 étapes."""

    name = "Création de modèle"
    steps = [ImageUploadStep, ConfigurationStep, ResultStep]
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
        return redirect("home")
