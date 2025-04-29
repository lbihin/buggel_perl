"""
Module pour le wizard de pixelisation.

Ce module contient l'implémentation du wizard de pixelisation,
utilisant le framework de wizard modulaire.
"""

from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse

from shapes.models import BeadShape

from .forms import PixelizationWizardForm
from .models import BeadModel
from .wizards import LoginRequiredWizard, WizardStep


class ConfigurationStep(WizardStep):
    """Première étape: Configuration du modèle de pixelisation."""

    name = "Configuration du modèle"
    template = "beadmodels/pixelization/pixelization_wizard.html"
    form_class = PixelizationWizardForm
    position = 1

    def handle_get(self, **kwargs):
        """Gère l'affichage du formulaire de configuration."""
        # Récupérer le modèle si un ID est fourni
        model_id = self.wizard.request.GET.get("model_id")
        model = None

        if model_id:
            try:
                model = BeadModel.objects.get(pk=model_id)
                # Vérifier l'accès au modèle
                if model.creator != self.wizard.request.user and not model.is_public:
                    messages.error(
                        self.wizard.request, "Vous n'avez pas accès à ce modèle."
                    )
                    return redirect("beadmodels:home")

                # Mettre à jour les données du wizard
                self.wizard.update_data({"model_id": model_id})
            except BeadModel.DoesNotExist:
                messages.error(self.wizard.request, "Le modèle spécifié n'existe pas.")
                return redirect("beadmodels:home")

        # Récupérer les données existantes du wizard
        wizard_data = self.wizard.get_data()

        # Initialiser le formulaire avec les données existantes
        initial_data = {
            "grid_width": wizard_data.get("grid_width", 29),
            "grid_height": wizard_data.get("grid_height", 29),
            "color_reduction": wizard_data.get("color_reduction", 16),
            "use_available_colors": wizard_data.get("use_available_colors", False),
        }

        model_has_image = model and model.original_image
        form = self.form_class(initial=initial_data, model_provided=model_has_image)

        # Récupérer les formes de l'utilisateur
        user_shapes = BeadShape.objects.filter(
            creator=self.wizard.request.user
        ).order_by("name")

        # Construire le contexte
        context = {
            "form": form,
            "model": model,
            "wizard_step": self.position,  # Pour compatibilité avec template existant
            "grid_type": wizard_data.get("grid_type", "square"),
            "user_shapes": user_shapes,
            "selected_shape_id": wizard_data.get("shape_id"),
            "has_grid_options": bool(user_shapes.exists()),
            "color_values": [2, 4, 6, 8, 16, 24, 32],
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite le formulaire soumis."""
        # Récupérer le modèle si un ID est fourni
        wizard_data = self.wizard.get_data()
        model_id = self.wizard.request.GET.get("model_id") or wizard_data.get(
            "model_id"
        )
        model = None

        if model_id:
            try:
                model = BeadModel.objects.get(pk=model_id)
                if model.creator != self.wizard.request.user and not model.is_public:
                    messages.error(
                        self.wizard.request, "Vous n'avez pas accès à ce modèle."
                    )
                    return redirect("beadmodels:home")
            except BeadModel.DoesNotExist:
                messages.error(self.wizard.request, "Le modèle spécifié n'existe pas.")
                return redirect("beadmodels:home")

        # Vérifier si on a une image à traiter (modèle avec image)
        model_has_image = model and model.original_image

        # Traiter le formulaire
        form = self.form_class(
            self.wizard.request.POST,
            self.wizard.request.FILES,
            model_provided=model_has_image,
            initial={"use_model_image": model_has_image},
        )

        if form.is_valid():
            # Stocker les données du formulaire
            form_data = {
                "grid_width": form.cleaned_data["grid_width"],
                "grid_height": form.cleaned_data["grid_height"],
                "color_reduction": form.cleaned_data["color_reduction"],
                "use_available_colors": form.cleaned_data["use_available_colors"],
                "grid_type": self.wizard.request.POST.get("grid_type", "square"),
                "shape_id": self.wizard.request.POST.get("shape_id"),
            }

            if model_id:
                form_data["model_id"] = model_id

            self.wizard.update_data(form_data)

            # Traiter l'image si disponible
            if model_has_image:
                # Traiter l'image du modèle
                from .views import process_image_for_wizard, process_image_pixelization

                image_data = process_image_for_wizard(model.original_image)
                form_data["uploaded_image"] = False

                # Traiter l'image pour la pixelisation
                processed_image = process_image_pixelization(
                    image_data["image_array"],
                    form_data["grid_width"],
                    form_data["grid_height"],
                    form_data["color_reduction"],
                    form_data["use_available_colors"],
                    (
                        self.wizard.request.user
                        if form_data["use_available_colors"]
                        else None
                    ),
                )

                # Mettre à jour les données d'image
                image_data.update(
                    {
                        "image_base64": processed_image["image_base64"],
                        "palette": processed_image["palette"],
                    }
                )

                form_data["image_data"] = image_data
                self.wizard.update_data(form_data)

                # Passer à l'étape suivante
                return self.wizard.go_to_next_step(
                    {"model_id": model_id} if model_id else {}
                )
            else:
                messages.error(
                    self.wizard.request,
                    "Aucune image disponible. Veuillez sélectionner un modèle avec une image.",
                )
                return redirect("beadmodels:home")
        else:
            # Récupérer les formes de l'utilisateur pour le rendu
            user_shapes = BeadShape.objects.filter(
                creator=self.wizard.request.user
            ).order_by("name")

            # Construire le contexte en cas d'erreur
            context = {
                "form": form,
                "model": model,
                "wizard_step": self.position,
                "user_shapes": user_shapes,
                "has_grid_options": bool(user_shapes.exists()),
                "color_values": [2, 4, 6, 8, 16, 24, 32],
                "selected_shape_id": self.wizard.request.POST.get("shape_id"),
            }

            return self.render_template(context)


class ResultStep(WizardStep):
    """Deuxième étape: Affichage du résultat de pixelisation."""

    name = "Visualisation du modèle"
    template = "beadmodels/pixelization/pixelization_result.html"
    position = 2

    def handle_get(self, **kwargs):
        """Affiche le résultat de la pixelisation."""
        # Récupérer les données du wizard
        wizard_data = self.wizard.get_data()

        # Vérifier que les données nécessaires sont présentes
        if "image_data" not in wizard_data:
            messages.warning(
                self.wizard.request, "Veuillez d'abord configurer votre modèle."
            )
            return self.wizard.go_to_step(
                1, {"model_id": wizard_data.get("model_id", "")}
            )

        # Récupérer le modèle si un ID est fourni
        model_id = wizard_data.get("model_id")
        model = None

        if model_id:
            try:
                model = BeadModel.objects.get(pk=model_id)
            except BeadModel.DoesNotExist:
                pass

        # Récupérer les données d'image
        image_data = wizard_data.get("image_data", {})

        # Construire le contexte
        context = {
            "image_base64": image_data.get("image_base64", ""),
            "grid_width": wizard_data.get("grid_width", 29),
            "grid_height": wizard_data.get("grid_height", 29),
            "palette": image_data.get("palette", []),
            "total_beads": wizard_data.get("grid_width", 29)
            * wizard_data.get("grid_height", 29),
            "wizard_step": self.position,  # Pour compatibilité avec template existant
            "wizard_data": wizard_data,
            "model": model,
        }

        return self.render_template(context)

    def handle_post(self, **kwargs):
        """Traite les actions finales du wizard."""
        # Ici on peut gérer des actions comme la sauvegarde du modèle
        # ou le téléchargement de l'image pixelisée

        # Pour l'instant, nous redirigerons simplement vers une action de téléchargement
        if "download" in self.wizard.request.POST:
            return redirect(reverse("beadmodels:download_pixelized_image"))

        return self.render_template()


class PixelizationWizard(LoginRequiredWizard):
    """Wizard complet de pixelisation."""

    name = "Assistant de pixelisation"
    steps = [ConfigurationStep, ResultStep]
    session_key = "pixelization_wizard"

    def get_url_name(self):
        """Renvoie le nom d'URL du wizard."""
        return "beadmodels:pixelization_wizard"

    def finish_wizard(self):
        """Action finale lorsque le wizard est terminé."""
        # Ici, on pourrait sauvegarder le modèle final ou rediriger vers une vue finale
        return redirect(reverse("beadmodels:home"))

    def get_redirect_kwargs(self):
        """Récupère les paramètres à conserver lors des redirections."""
        # Conserver l'ID du modèle dans les redirections
        model_id = self.get_data().get("model_id")
        if model_id:
            return {"model_id": model_id}
        return {}

    def dispatch(self, request, *args, **kwargs):
        """Point d'entrée principal pour le traitement des requêtes."""
        self.request = request

        # Vérifier si on doit réinitialiser le wizard
        model_id = request.GET.get("model_id")
        current_data = request.session.get(self.session_key, {})
        current_model_id = current_data.get("model_id")

        # On force le reset du wizard dans deux cas :
        # 1. Si explicitement demandé par ?reset=true
        # 2. Si un nouvel ID de modèle est fourni qui diffère de celui en session
        # 3. Si on est à l'étape 2 et qu'on demande à accéder à l'étape 1
        reset_wizard = (
            request.GET.get("reset") == "true"
            or (model_id and str(current_model_id) != str(model_id))
            or (request.GET.get("step") == "1" and self.get_current_step_number() == 2)
        )

        if reset_wizard:
            self.reset_wizard()
            if model_id:
                # Réinitialiser mais conserver l'ID du modèle
                self.update_data({"model_id": model_id})

            # Revenir à l'étape 1
            self.set_current_step_number(1)

            # Afficher un message seulement si demandé explicitement
            if request.GET.get("reset") == "true":
                messages.info(request, f"Le {self.name.lower()} a été réinitialisé.")

        # Continuer avec la logique standard
        # Gestion des boutons précédent/suivant
        if request.method == "POST":
            if "previous_step" in request.POST:
                return self.go_to_previous_step(self.get_redirect_kwargs())

        # Déléguer à l'étape courante
        step = self.get_current_step()
        if request.method == "GET":
            return step.handle_get(**kwargs)
        elif request.method == "POST":
            return step.handle_post(**kwargs)
