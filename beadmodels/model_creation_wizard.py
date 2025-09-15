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

        # Réinitialiser les données si on revient à la première étape depuis une étape suivante
        if self.wizard.get_current_step_number() > 1 and not reset_wizard:
            self.wizard.reset_wizard()
            # Mais conserver l'ID du modèle si présent
            if model_id:
                self.wizard.update_data({"model_id": model_id})

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
            "grid_width": wizard_data.get("grid_width", 29),
            "grid_height": wizard_data.get("grid_height", 29),
            "color_reduction": wizard_data.get("color_reduction", 16),
            "use_available_colors": wizard_data.get("use_available_colors", False),
        }

        form = self.form_class(initial=initial_data)

        # Récupérer les boards disponibles
        boards = BeadBoard.objects.all()

        # Générer une prévisualisation avec les paramètres par défaut
        preview_image_base64 = self.generate_preview(wizard_data)

        # Construire le contexte
        context = {
            "form": form,
            "image_base64": image_data.get("image_base64", ""),
            "preview_image_base64": preview_image_base64,
            "boards": boards,
            "wizard_step": self.position,
            "total_steps": 3,
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite le formulaire de configuration."""
        wizard_data = self.wizard.get_data()

        # Vérifier si c'est une requête HTMX pour une prévisualisation
        if self.wizard.request.headers.get("HX-Request") == "true":
            # Mise à jour des paramètres
            grid_width = int(self.wizard.request.POST.get("grid_width", 29))
            grid_height = int(self.wizard.request.POST.get("grid_height", 29))
            color_reduction = int(self.wizard.request.POST.get("color_reduction", 16))
            use_available_colors = (
                self.wizard.request.POST.get("use_available_colors") == "on"
            )

            # Mettre à jour les données du wizard
            self.wizard.update_data(
                {
                    "grid_width": grid_width,
                    "grid_height": grid_height,
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
            # Mettre à jour les données du wizard
            self.wizard.update_data(
                {
                    "grid_width": form.cleaned_data["grid_width"],
                    "grid_height": form.cleaned_data["grid_height"],
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

    def generate_preview(self, wizard_data):
        """Génère une prévisualisation du modèle."""
        image_data = wizard_data.get("image_data", {})
        image_base64 = image_data.get("image_base64", "")

        if not image_base64:
            return ""

        # Récupérer les paramètres
        grid_width = wizard_data.get("grid_width", 29)
        grid_height = wizard_data.get("grid_height", 29)
        color_reduction = wizard_data.get("color_reduction", 16)
        use_available_colors = wizard_data.get("use_available_colors", False)

        # Décoder l'image base64
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes))

        # S'assurer que l'image est en mode RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Redimensionner l'image
        img = img.resize((grid_width, grid_height), Image.Resampling.LANCZOS)

        # Convertir en array numpy pour le traitement
        img_array = np.array(img)  # Aplatir l'image pour le clustering
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

        # Convertir en base64
        preview_bytes = io.BytesIO()
        reduced_img.save(preview_bytes, format="PNG")
        preview_bytes.seek(0)
        preview_base64 = base64.b64encode(preview_bytes.read()).decode("utf-8")

        return preview_base64

    def generate_model(self, wizard_data):
        """Génère le modèle final."""
        # Cette méthode est similaire à generate_preview mais avec la taille finale
        preview_base64 = self.generate_preview(wizard_data)

        # Extraire les informations supplémentaires nécessaires
        color_reduction = wizard_data.get("color_reduction", 16)
        grid_width = wizard_data.get("grid_width", 29)
        grid_height = wizard_data.get("grid_height", 29)

        # Calculer le nombre total de perles
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

        # Construire le contexte
        context = {
            "image_base64": final_model.get("image_base64", ""),
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
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
        if "reset" in request.GET and request.GET.get("reset") == "true":
            self.reset_wizard()
        return super().dispatch(request, *args, **kwargs)

    def finish_wizard(self):
        """Action finale lorsque le wizard est terminé."""
        self.reset_wizard()
        return redirect("home")
