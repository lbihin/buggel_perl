"""
Module pour le wizard de création de modèle.

Ce module contient l'implémentation du wizard à 3 étapes pour
la création de modèles de perles à repasser.
"""

import base64
import io
import logging
from datetime import datetime

import numpy as np
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from PIL import Image

from beads.models import Bead

from ..forms import ImageUploadForm, ModelConfigurationForm
from ..models import BeadBoard
from .wizard_helpers import LoginRequiredWizard, WizardStep

logger = logging.getLogger(__name__)


class UploadImage(WizardStep):
    """Première étape: Chargement de l'image."""

    name = "Chargement de l'image"
    template = "beadmodels/wizard/upload_image.html"
    form_class = ImageUploadForm
    position = 1

    def handle_get(self, **kwargs):
        """Gère l'affichage du formulaire de chargement d'image."""

        # Initialiser le formulaire
        form = self.form_class()

        context = {"form": form, "wizard_step": self.position, "total_steps": 3}
        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite le formulaire de chargement d'image."""
        form = self.form_class(self.wizard.request.POST, self.wizard.request.FILES)

        if form.is_valid():
            from ..services.image_processing import save_temp_image

            image = form.cleaned_data["image"]
            # Sauvegarde temporaire sur disque pour éviter surcharge session
            stored_path = save_temp_image(image)
            self.wizard.update_session_data({"image_data": {"image_path": stored_path}})

            # Passer à l'étape suivante
            return self.wizard.go_to_next_step()

        # En cas d'erreur, réafficher le formulaire avec les erreurs
        context = {"form": form, "wizard_step": self.position, "total_steps": 3}
        return self.render_template(context)


class ConfigureModel(WizardStep):
    """Deuxième étape: Configuration et prévisualisation du modèle."""

    name = "Configuration du modèle"
    template = "beadmodels/wizard/configure_model.html"
    form_class = ModelConfigurationForm
    position = 2

    def handle_get(self, **kwargs):
        """Gère l'affichage du formulaire de configuration."""

        from shapes.models import BeadShape

        from ..services.image_processing import file_to_base64

        wizard_data = self.wizard.get_session_data()
        image_data = wizard_data.get("image_data", {})

        if not image_data:
            messages.error(self.wizard.request, "Veuillez d'abord charger une image.")
            return redirect("beadmodels:create", kwargs={"q": "reset"})

        # Initialiser le formulaire avec les données existantes ou valeurs par défaut
        initial_data = {
            "color_reduction": wizard_data.get("color_reduction", 16),
            "use_available_colors": wizard_data.get("use_available_colors", False),
        }

        form = self.form_class(initial=initial_data)

        # Récupérer les formes de l'utilisateur

        usr_shapes = BeadShape.objects.filter(
            creator=self.wizard.request.user
        ).order_by("name")

        # Trouver les formes par défaut si aucune n´est par défaut la première est choisie
        selected_shape_id = wizard_data.get("shape_id")

        if not selected_shape_id:
            selected_shape_id = (
                usr_shapes.filter(is_default=True).first() or usr_shapes.first()
            ).pk
            self.wizard.update_session_data({"shape_id": selected_shape_id})

        # Définir les valeurs de couleurs disponibles
        color_values = [2, 4, 6, 8, 16, 24, 32]

        # Générer une prévisualisation avec les paramètres par défaut
        preview_image_base64 = self.generate_preview(wizard_data)

        # Fallback pour afficher l'image originale si seulement path présent
        original_image_base64 = file_to_base64(image_data.get("image_path"))

        current_step_context = self.get_context_data()

        # Construire le contexte
        context = {
            "form": form,
            "image_base64": original_image_base64,
            "preview_image_base64": preview_image_base64,
            "user_shapes": usr_shapes,
            "selected_shape_id": wizard_data.get("shape_id"),
            "color_values": color_values,
            "step_nr": self.position,
            "step_name": self.name,
            "total_steps": current_step_context.get("total_steps"),
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite le formulaire de configuration."""
        wizard_data = self.wizard.get_session_data()

        # Gestion des boutons de navigation
        direction = self.wizard.request.POST.get("q")
        if direction == "previous":
            return self.wizard.go_to_previous_step()
        if direction == "next":
            return self.wizard.go_to_next_step()

        # Prévisualisation déclenchée via HTMX
        if getattr(self.wizard.request, "htmx", False):
            # Mise à jour des paramètres
            shape_id = self.wizard.request.POST.get("shape_id", "")
            posted_color = self.wizard.request.POST.get("color_reduction")

            # Log pour debug
            logger.info(
                f"HTMX preview request - shape_id: {shape_id}, color: {posted_color}"
            )

            try:
                color_reduction = (
                    int(posted_color)
                    if posted_color
                    else int(wizard_data.get("color_reduction", 16))
                )
            except (ValueError, TypeError):
                color_reduction = 16
            use_available_colors = (
                self.wizard.request.POST.get("use_available_colors") == "on"
            )

            # Mettre à jour les données du wizard
            self.wizard.update_session_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": color_reduction,
                    "use_available_colors": use_available_colors,
                }
            )

            # Générer la prévisualisation
            preview_image_base64 = self.generate_preview(self.wizard.get_session_data())

            # Renvoyer uniquement la partie prévisualisation
            context = {"preview_image_base64": preview_image_base64}

            html = render_to_string("beadmodels/partials/preview.html", context)
            return HttpResponse(html)

        # Soumission normale (non HTMX) : deux cas
        # 1. Bouton "previous_step" géré plus haut
        # 2. Bouton "generate" doit forcer passage à l'étape 3 même si le champ hidden n'est pas synchronisé

        if "generate" in self.wizard.request.POST:
            shape_id = self.wizard.request.POST.get("shape_id", "")
            posted_color = self.wizard.request.POST.get("color_reduction")
            try:
                color_reduction = (
                    int(posted_color)
                    if posted_color
                    else int(wizard_data.get("color_reduction", 16))
                )
            except (ValueError, TypeError):
                color_reduction = 16
            use_available_colors = (
                self.wizard.request.POST.get("use_available_colors") == "on"
            )

            # Mettre à jour les données (écraser toujours pour cohérence)
            self.wizard.update_session_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": color_reduction,
                    "use_available_colors": use_available_colors,
                }
            )

            # Générer le modèle final directement
            final_model = self.generate_model(self.wizard.get_session_data())

            # Stocker image finale sur disque (réduction session)
            if final_model.get("image_base64"):
                try:
                    import base64

                    from django.core.files.base import ContentFile

                    from .services.image_processing import save_temp_image

                    image_bytes = base64.b64decode(final_model["image_base64"])
                    temp_file = ContentFile(image_bytes, name="final_preview.png")
                    stored_path = save_temp_image(temp_file)
                    final_model["image_path"] = stored_path
                    del final_model["image_base64"]
                except Exception as e:
                    logger.warning(
                        f"Impossible de sauvegarder l'image finale temporaire (generate bypass): {e}"
                    )

            self.wizard.update_session_data({"final_model": final_model})
            # Si requête HTMX (hx-boost précédemment), retourner directement le template final
            if getattr(self.wizard.request, "htmx", False):
                self.wizard.set_current_step_number(3)
                save_step = self.wizard.get_step_by_number(3)
                return save_step.handle_get()
            # Sinon redirection standard
            return self.wizard.go_to_next_step()

        # Si c'est une soumission normale du formulaire (sans bouton generate explicite)
        form = self.form_class(self.wizard.request.POST)

        if form.is_valid():
            # Récupérer l'ID de la forme sélectionnée
            shape_id = self.wizard.request.POST.get("shape_id", "")

            # Mettre à jour les données du wizard
            safe_color_reduction = (
                form.cleaned_data.get("color_reduction")
                or wizard_data.get("color_reduction")
                or 16
            )
            try:
                safe_color_reduction = int(safe_color_reduction)
            except (ValueError, TypeError):
                safe_color_reduction = 16
            self.wizard.update_session_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": safe_color_reduction,
                    "use_available_colors": form.cleaned_data["use_available_colors"],
                }
            )

            # Générer le modèle final
            final_model = self.generate_model(self.wizard.get_session_data())

            # Optimisation: sauvegarder l'image finale en fichier temporaire plutôt que garder base64 en session
            if final_model.get("image_base64"):
                try:
                    import base64
                    import io

                    from django.core.files.base import ContentFile

                    from .services.image_processing import save_temp_image

                    image_bytes = base64.b64decode(final_model["image_base64"])
                    temp_file = ContentFile(image_bytes, name="final_preview.png")
                    stored_path = save_temp_image(temp_file)
                    # Remplacer base64 par chemin
                    final_model["image_path"] = stored_path
                    # Optionnel: conserver une miniature très réduite si besoin futur
                    del final_model["image_base64"]
                except Exception as e:
                    logger.warning(
                        f"Impossible de sauvegarder l'image finale temporaire: {e}"
                    )

            self.wizard.update_session_data({"final_model": final_model})

            # Passer à l'étape suivante explicitement
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

    def generate_preview(self, data, return_reduced_pixels=False):
        """Génère une prévisualisation pixelisée de l'image.

        Args:
            data: Wizard data dictionary
            return_reduced_pixels: If True, returns tuple (base64_image, reduced_pixels_array, content_mask)

        Returns:
            base64 string or tuple (base64_image, reduced_pixels, content_mask) if return_reduced_pixels=True
        """
        image_data = data.get("image_data", {})
        image_base64 = image_data.get("image_base64")
        image_path = image_data.get("image_path")
        if not image_base64 and not image_path:
            return ("", None, None) if return_reduced_pixels else ""

        # Paramètres de configuration
        shape_id = data.get("shape_id", "")
        raw_color_reduction = data.get("color_reduction", 16)
        try:
            color_reduction = int(raw_color_reduction) if raw_color_reduction else 16
        except (ValueError, TypeError):
            color_reduction = 16
        use_available_colors = data.get("use_available_colors", False)

        # Décoder l'image base64
        if image_path and not image_base64:
            # Charger depuis stockage en mémoire pour éviter "seek of closed file"
            from django.core.files.storage import default_storage

            with default_storage.open(image_path, "rb") as f:
                raw_bytes = f.read()
            img = Image.open(io.BytesIO(raw_bytes))
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
        # Initialiser le masque de contenu par défaut (tout est contenu)
        image_content_mask = None

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
                    # rectangle dimensions applied
                elif shape_type == "square" and shape.size is not None:
                    grid_width = shape.size
                    grid_height = shape.size
                    # square dimensions applied
                elif shape_type == "circle" and shape.diameter is not None:
                    grid_width = shape.diameter
                    grid_height = shape.diameter
                    use_circle_mask = True
                    circle_diameter = shape.diameter
                    # circle dimensions applied
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

            # Pour les cercles aussi, créer un masque du contenu réel
            # Le masque sera True pour les pixels de l'image source, False pour le fond blanc
            image_content_mask = np.zeros((grid_height, grid_width), dtype=bool)
            # Pour un cercle, marquer tous les pixels dans le rayon
            center_x_circle, center_y_circle = grid_width // 2, grid_height // 2
            radius_mask = circle_diameter // 2
            y_circle, x_circle = np.ogrid[:grid_height, :grid_width]
            dist_squared_mask = (x_circle - center_x_circle) ** 2 + (
                y_circle - center_y_circle
            ) ** 2
            image_content_mask = dist_squared_mask <= radius_mask**2

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

            # Créer un masque pour identifier les zones d'image réelle (non-blanc de remplissage)
            # Le masque sera True pour les pixels de l'image source, False pour le fond blanc
            image_content_mask = np.zeros((grid_height, grid_width), dtype=bool)
            image_content_mask[
                offset_y : offset_y + target_height, offset_x : offset_x + target_width
            ] = True

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
        from ..services.image_processing import reduce_colors

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

        if return_reduced_pixels:
            # Retourner aussi le masque du contenu pour exclure les zones de remplissage blanc
            return preview_base64, reduced_pixels, image_content_mask
        return preview_base64

    def generate_model(self, wizard_data):
        """Génère le modèle final."""
        # Générer l'image ET récupérer les pixels réduits + masque pour palette précise
        preview_base64, reduced_pixels, content_mask = self.generate_preview(
            wizard_data, return_reduced_pixels=True
        )

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

        # Palette via service - utiliser les pixels réduits pour précision
        from ..services.image_processing import compute_palette

        palette = compute_palette(
            reduced_pixels=reduced_pixels,
            total_beads=total_beads,
            content_mask=content_mask,
        )

        return {
            "image_base64": preview_base64,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "shape_id": wizard_data.get("shape_id"),
            "color_reduction": color_reduction,
            "total_beads": total_beads,
            "palette": palette,
        }


class SaveModel(WizardStep):
    """Troisième étape: Sauvegarde du modèle."""

    name = "Finalisation"
    template = "beadmodels/wizard/result.html"
    position = 3

    def handle_get(self, **kwargs):
        """Affiche le résultat final et les options de sauvegarde."""
        wizard_data = self.wizard.get_session_data()
        final_model = wizard_data.get("final_model", {})

        # TODO: désactivation temporaire. On vérifie que les boutons suivant et précédent fonctionnent correctement.
        # if not final_model:
        #     messages.error(self.wizard.request, "Veuillez configurer votre modèle.")
        #     return redirect("beadmodels:create")

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

        # Formulaire dédié à la finalisation
        # Obtenir la date actuelle formatée en français pour le nom par défaut
        import locale

        from ..forms import BeadModelFinalizeForm

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

        form = BeadModelFinalizeForm(
            initial=initial_data, user=self.wizard.request.user
        )

        # Charger l'image finale depuis le chemin si disponible
        final_image_base64 = final_model.get("image_base64")
        if not final_image_base64 and final_model.get("image_path"):
            try:
                from ..services.image_processing import file_to_base64

                final_image_base64 = file_to_base64(final_model.get("image_path"))
            except Exception as e:
                logger.warning(
                    f"Impossible de charger l'image finale depuis le chemin: {e}"
                )
                final_image_base64 = ""

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
        excluded_colors = final_model.get("excluded_colors", [])
        context = {
            "image_base64": final_image_base64 or "",
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
            "excluded_colors": excluded_colors,
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Gère les actions finales avec une meilleure expérience utilisateur."""
        from django.contrib import messages

        from ..forms import BeadModelFinalizeForm

        # Récupérer les données du modèle final
        wizard_data = self.wizard.get_session_data()
        final_model = wizard_data.get("final_model", {})

        # TODO: désactivation temporaire. On vérifie que les boutons suivant et précédent fonctionnent correctement.
        # if not final_model:
        #     messages.error(self.wizard.request, "Erreur: données du modèle manquantes.")
        #     return redirect("beadmodels:create")

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
            form = BeadModelFinalizeForm(
                self.wizard.request.POST, user=self.wizard.request.user
            )

            if form.is_valid():
                # Delegate the heavy save logic to a service function
                try:
                    from .services.save_model import save_bead_model_from_wizard

                    new_model = save_bead_model_from_wizard(
                        self.wizard.request.user, wizard_data, form
                    )
                except Exception as e:
                    logger.exception("Erreur lors de la sauvegarde du modèle: %s", e)
                    messages.error(
                        self.wizard.request,
                        "Une erreur est survenue lors de la sauvegarde du modèle.",
                    )
                    # Re-render the form with an error message
                    context = {
                        "image_base64": final_model.get("image_base64", ""),
                        "grid_width": final_model.get("grid_width", 29),
                        "grid_height": final_model.get("grid_height", 29),
                        "shape_id": final_model.get("shape_id"),
                        "total_beads": final_model.get("total_beads", 0),
                        "palette": final_model.get("palette", []),
                        "beads_count": len(final_model.get("palette", [])),
                        "wizard_step": self.position,
                        "total_steps": 3,
                        "form": form,
                        "boards": BeadBoard.objects.all(),
                        "excluded_colors": final_model.get("excluded_colors", []),
                    }

                    return self.render_template(context)

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
                    "excluded_colors": final_model.get("excluded_colors", []),
                }

                return self.render_template(context)

        # Par défaut, réafficher simplement le template
        return self.render_template()

    def download_model(self, format="png"):
        """Prépare le téléchargement du modèle dans différents formats."""
        wizard_data = self.wizard.get_session_data()
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
        wizard_data = self.wizard.get_session_data()
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


class ModelCreatorWizard(LoginRequiredWizard):
    """Wizard complet de création de modèle à 3 étapes."""

    name = "Création de modèle de perles"
    steps = [UploadImage, ConfigureModel, SaveModel]  # Étapes du wizard par ordre
    session_key = "model_creation_wizard"

    def get_url_name(self):
        """Renvoie le nom d'URL du wizard."""
        return "beadmodels:create"

    def get_redirect_kwargs(self):
        """Récupère les paramètres à conserver lors des redirections."""
        model_id = self.get_session_data()
        return {"model_id": model_id} if model_id else {}

    def start_wizard(self):
        self.reset_wizard()
        return self.go_to_step(1)

    def finish_wizard(self):
        """Action finale lorsque le wizard est terminé."""
        self.reset_wizard()
        return redirect("beadmodels:my_models")
