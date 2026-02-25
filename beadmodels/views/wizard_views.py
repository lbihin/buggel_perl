"""
Module pour le wizard de creation de modele.

Ce module contient l'implementation du wizard a 3 etapes pour
la creation de modeles de perles a repasser.
Le traitement d'image lourd est delegue a services/image_processing.py.
"""

import base64
import logging
import re
from dataclasses import asdict
from datetime import datetime

import numpy as np
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string

from beads.models import Bead

from ..forms import ImageUploadForm, ModelConfigurationForm
from ..models import BeadBoard
from ..services.image_processing import (
    ModelResult,
    file_to_base64,
    generate_model,
    generate_preview,
    save_temp_image,
)
from .wizard_helpers import LoginRequiredWizard, WizardStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_user_bead_colors(user) -> np.ndarray | None:
    """Return an Nx3 numpy array of the user's bead RGB colours, or None."""
    beads = Bead.objects.filter(creator=user)
    if not beads.exists():
        return None
    return np.array([[b.red, b.green, b.blue] for b in beads])


def _safe_int(value, default: int = 16) -> int:
    try:
        return int(value) if value else default
    except (ValueError, TypeError):
        return default


def _build_preview_kwargs(wizard_data: dict, user) -> dict:
    """Build keyword arguments for services.generate_preview from wizard session data."""
    image_data = wizard_data.get("image_data", {})
    use_available = wizard_data.get("use_available_colors", False)
    return {
        "image_path": image_data.get("image_path"),
        "image_base64": image_data.get("image_base64"),
        "shape_id": wizard_data.get("shape_id"),
        "color_reduction": _safe_int(wizard_data.get("color_reduction"), 16),
        "use_available_colors": use_available,
        "user_bead_colors": _get_user_bead_colors(user) if use_available else None,
    }


def _store_final_image(model_dict: dict) -> dict:
    """If model_dict contains image_base64, persist it to temp storage and replace with path."""
    b64 = model_dict.get("image_base64")
    if not b64:
        return model_dict
    try:
        from django.core.files.base import ContentFile

        image_bytes = base64.b64decode(b64)
        temp_file = ContentFile(image_bytes, name="final_preview.png")
        stored_path = save_temp_image(temp_file)
        model_dict["image_path"] = stored_path
        del model_dict["image_base64"]
    except Exception as e:
        logger.warning("Impossible de sauvegarder l'image finale temporaire: %s", e)
    return model_dict


def _model_result_to_dict(result: ModelResult) -> dict:
    """Convert a ModelResult dataclass to a plain dict for session storage."""
    d = asdict(result)
    # shape_id may be a string or int; keep it as-is
    return d


# ---------------------------------------------------------------------------
# Etape 1 : Upload
# ---------------------------------------------------------------------------


class UploadImage(WizardStep):
    """Premiere etape: Chargement de l'image."""

    name = "Chargement de l'image"
    template = "beadmodels/wizard/upload_image.html"
    form_class = ImageUploadForm
    position = 1

    def handle_get(self, **kwargs):
        form = self.form_class()
        return self.render_template(
            {"form": form, "wizard_step": self.position, "total_steps": 3}
        )

    def handle_post(self, **kwargs):
        form = self.form_class(self.wizard.request.POST, self.wizard.request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image"]
            stored_path = save_temp_image(image)
            self.wizard.update_session_data({"image_data": {"image_path": stored_path}})
            return self.wizard.go_to_next_step()

        return self.render_template(
            {"form": form, "wizard_step": self.position, "total_steps": 3}
        )


# ---------------------------------------------------------------------------
# Etape 2 : Configuration & Preview
# ---------------------------------------------------------------------------


class ConfigureModel(WizardStep):
    """Deuxieme etape: Configuration et previsualisation du modele."""

    name = "Configuration du modèle"
    template = "beadmodels/wizard/configure_model.html"
    form_class = ModelConfigurationForm
    position = 2

    def handle_get(self, **kwargs):
        from shapes.models import BeadShape

        wizard_data = self.wizard.get_session_data()
        image_data = wizard_data.get("image_data", {})

        if not image_data:
            messages.error(self.wizard.request, "Veuillez d'abord charger une image.")
            return redirect("beadmodels:create", kwargs={"q": "reset"})

        initial_data = {
            "color_reduction": wizard_data.get("color_reduction", 16),
            "use_available_colors": wizard_data.get("use_available_colors", False),
        }
        form = self.form_class(initial=initial_data)

        usr_shapes = BeadShape.objects.filter(
            creator=self.wizard.request.user
        ).order_by("name")

        selected_shape_id = wizard_data.get("shape_id")
        if not selected_shape_id:
            default = usr_shapes.filter(is_default=True).first() or usr_shapes.first()
            if default:
                selected_shape_id = default.pk
                self.wizard.update_session_data({"shape_id": selected_shape_id})

        color_values = [2, 4, 6, 8, 16, 24, 32]

        preview_kwargs = _build_preview_kwargs(wizard_data, self.wizard.request.user)
        preview_result = generate_preview(**preview_kwargs)
        original_image_base64 = file_to_base64(image_data.get("image_path"))

        step_ctx = self.get_context_data()
        context = {
            "form": form,
            "image_base64": original_image_base64,
            "preview_image_base64": preview_result.image_base64,
            "user_shapes": usr_shapes,
            "selected_shape_id": wizard_data.get("shape_id"),
            "color_values": color_values,
            "step_nr": self.position,
            "step_name": self.name,
            "total_steps": step_ctx.get("total_steps"),
        }
        return self.render_template(context)

    def handle_post(self, **kwargs):
        wizard_data = self.wizard.get_session_data()
        request = self.wizard.request

        # Navigation
        direction = request.POST.get("q")
        if direction == "previous":
            return self.wizard.go_to_previous_step()
        if direction == "next":
            return self.wizard.go_to_next_step()

        # ---------- HTMX preview ----------
        if getattr(request, "htmx", False) and "generate" not in request.POST:
            shape_id = request.POST.get("shape_id", "")
            color_reduction = _safe_int(
                request.POST.get("color_reduction"),
                _safe_int(wizard_data.get("color_reduction"), 16),
            )
            use_available = request.POST.get("use_available_colors") == "on"

            self.wizard.update_session_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": color_reduction,
                    "use_available_colors": use_available,
                }
            )

            preview_kwargs = _build_preview_kwargs(
                self.wizard.get_session_data(), request.user
            )
            preview_result = generate_preview(**preview_kwargs)
            html = render_to_string(
                "beadmodels/partials/preview.html",
                {"preview_image_base64": preview_result.image_base64},
            )
            return HttpResponse(html)

        # ---------- Generate button ----------
        if "generate" in request.POST:
            return self._handle_generate(wizard_data)

        # ---------- Standard form submit ----------
        form = self.form_class(request.POST)
        if form.is_valid():
            shape_id = request.POST.get("shape_id", "")
            color_reduction = _safe_int(
                form.cleaned_data.get("color_reduction")
                or wizard_data.get("color_reduction"),
                16,
            )
            self.wizard.update_session_data(
                {
                    "shape_id": shape_id,
                    "color_reduction": color_reduction,
                    "use_available_colors": form.cleaned_data["use_available_colors"],
                }
            )

            model_result = self._generate_final_model()
            model_dict = _store_final_image(_model_result_to_dict(model_result))
            self.wizard.update_session_data({"final_model": model_dict})
            return self.wizard.go_to_next_step()

        # Validation errors — re-render with preview
        preview_kwargs = _build_preview_kwargs(wizard_data, request.user)
        preview_result = generate_preview(**preview_kwargs)
        context = {
            "form": form,
            "image_base64": wizard_data.get("image_data", {}).get("image_base64", ""),
            "preview_image_base64": preview_result.image_base64,
            "wizard_step": self.position,
            "total_steps": 3,
        }
        return self.render_template(context)

    # -- private helpers ---

    def _handle_generate(self, wizard_data: dict):
        """Handle the 'generate' button: produce final model and go to step 3."""
        request = self.wizard.request
        shape_id = request.POST.get("shape_id", "")
        color_reduction = _safe_int(
            request.POST.get("color_reduction"),
            _safe_int(wizard_data.get("color_reduction"), 16),
        )
        use_available = request.POST.get("use_available_colors") == "on"

        self.wizard.update_session_data(
            {
                "shape_id": shape_id,
                "color_reduction": color_reduction,
                "use_available_colors": use_available,
            }
        )

        model_result = self._generate_final_model()
        model_dict = _store_final_image(_model_result_to_dict(model_result))
        self.wizard.update_session_data({"final_model": model_dict})

        if getattr(request, "htmx", False):
            self.wizard.set_current_step_number(3)
            save_step = self.wizard.get_step_by_number(3)
            return save_step.handle_get()

        return self.wizard.go_to_next_step()

    def _generate_final_model(self) -> ModelResult:
        """Call the service to generate the full model from current session data."""
        data = self.wizard.get_session_data()
        kwargs = _build_preview_kwargs(data, self.wizard.request.user)
        return generate_model(**kwargs)


# ---------------------------------------------------------------------------
# Etape 3 : Sauvegarde
# ---------------------------------------------------------------------------


class SaveModel(WizardStep):
    """Troisieme etape: Sauvegarde du modele."""

    name = "Finalisation"
    template = "beadmodels/wizard/result.html"
    position = 3

    def handle_get(self, **kwargs):
        wizard_data = self.wizard.get_session_data()
        final_model = wizard_data.get("final_model", {})

        # Shape info
        shape_name, shape_type = "Standard", "rectangle"
        if final_model.get("shape_id"):
            from shapes.models import BeadShape

            try:
                shape = BeadShape.objects.get(pk=final_model["shape_id"])
                shape_name, shape_type = shape.name, shape.shape_type
            except BeadShape.DoesNotExist:
                pass

        user_beads = Bead.objects.filter(creator=self.wizard.request.user)

        # Default name
        import locale

        from ..forms import BeadModelFinalizeForm

        try:
            locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
        except locale.Error:
            pass

        default_name = f"Modèle du {datetime.now().strftime('%d %B %Y')}"

        default_board = None
        board_id = final_model.get("board_id")
        if board_id:
            default_board = BeadBoard.objects.filter(pk=board_id).first()
        if not default_board:
            default_board = BeadBoard.objects.first()

        form = BeadModelFinalizeForm(
            initial={
                "name": default_name,
                "is_public": False,
                "board": default_board.pk if default_board else None,
            },
            user=self.wizard.request.user,
        )

        # Final image
        final_image_base64 = final_model.get("image_base64", "")
        if not final_image_base64 and final_model.get("image_path"):
            try:
                final_image_base64 = file_to_base64(final_model["image_path"])
            except Exception as e:
                logger.warning("Impossible de charger l'image finale: %s", e)

        # Enrich palette with user-bead matches
        palette = self._match_palette_to_beads(
            final_model.get("palette", []), user_beads
        )

        context = {
            "image_base64": final_image_base64,
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "shape_id": final_model.get("shape_id"),
            "shape_name": shape_name,
            "shape_type": shape_type,
            "total_beads": final_model.get("total_beads", 0),
            "palette": palette,
            "beads_count": len(final_model.get("palette", [])),
            "wizard_step": self.position,
            "total_steps": 3,
            "form": form,
            "boards": BeadBoard.objects.all(),
            "user_has_beads": user_beads.exists(),
            "default_board": default_board,
            "board_preview_url": (
                f"/media/board_previews/{default_board.pk}.png"
                if default_board
                else None
            ),
            "excluded_colors": final_model.get("excluded_colors", []),
        }
        return self.render_template(context)

    def handle_post(self, **kwargs):
        from ..forms import BeadModelFinalizeForm

        wizard_data = self.wizard.get_session_data()
        final_model = wizard_data.get("final_model", {})

        # Downloads
        if "download" in self.wizard.request.POST:
            fmt = self.wizard.request.POST.get("download_format", "png")
            return self._download_model(final_model, fmt)
        if "download_instructions" in self.wizard.request.POST:
            return self._generate_instructions(final_model)
        if "previous_step" in self.wizard.request.POST:
            return self.wizard.go_to_previous_step()

        # Save
        form = BeadModelFinalizeForm(
            self.wizard.request.POST, user=self.wizard.request.user
        )
        if form.is_valid():
            try:
                from ..services.save_model import save_bead_model_from_wizard

                new_model = save_bead_model_from_wizard(
                    self.wizard.request.user, wizard_data, form
                )
            except Exception as e:
                logger.exception("Erreur lors de la sauvegarde du modele: %s", e)
                messages.error(
                    self.wizard.request,
                    "Une erreur est survenue lors de la sauvegarde du modèle.",
                )
                return self._render_with_final_model(final_model, form)

            messages.success(
                self.wizard.request,
                f"Félicitations ! Votre modèle '{new_model.name}' a été créé avec succès ! "
                f"Il contient {final_model.get('total_beads', 0)} perles et "
                f"{len(final_model.get('palette', []))} couleurs différentes.",
            )
            self.wizard.reset_wizard()
            return redirect("beadmodels:details", pk=new_model.pk)

        return self._render_with_final_model(final_model, form)

    # -- private helpers ---

    def _render_with_final_model(self, final_model: dict, form):
        """Re-render step 3 with an existing form (possibly with errors)."""
        shape_name, shape_type = "Standard", "rectangle"
        if final_model.get("shape_id"):
            from shapes.models import BeadShape

            try:
                shape = BeadShape.objects.get(pk=final_model["shape_id"])
                shape_name, shape_type = shape.name, shape.shape_type
            except BeadShape.DoesNotExist:
                pass

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

    @staticmethod
    def _match_palette_to_beads(palette: list, user_beads) -> list:
        """Enrich each palette entry with matching bead suggestions."""
        if not user_beads.exists():
            return palette

        bead_list = list(user_beads)
        enriched = []
        for item in palette:
            rgb_match = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", item["color"])
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                matches = []
                for bead in bead_list:
                    dist = np.sqrt(
                        (bead.red - r) ** 2
                        + (bead.green - g) ** 2
                        + (bead.blue - b) ** 2
                    )
                    if dist < 30:
                        matches.append(
                            {
                                "name": bead.name,
                                "color": f"rgb({bead.red}, {bead.green}, {bead.blue})",
                                "hex": f"#{bead.red:02x}{bead.green:02x}{bead.blue:02x}",
                                "distance": int(dist),
                                "quantity": bead.quantity,
                            }
                        )
                matches.sort(key=lambda m: m["distance"])
                item["matches"] = matches[:3]
            enriched.append(item)
        return enriched

    @staticmethod
    def _download_model(final_model: dict, fmt: str = "png") -> HttpResponse:
        """Prepare an HTTP download response for the model image."""
        import io

        from PIL import Image

        image_b64 = final_model.get("image_base64", "")
        if not image_b64:
            return HttpResponse("Image introuvable.", status=404)

        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        output = io.BytesIO()

        if fmt.lower() in ("jpg", "jpeg"):
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(output, format="JPEG", quality=90)
            content_type, ext = "image/jpeg", "jpg"
        else:
            img.save(output, format="PNG")
            content_type, ext = "image/png", "png"

        output.seek(0)
        filename = f"modele_perles_{datetime.now():%Y%m%d_%H%M%S}.{ext}"
        response = HttpResponse(output.getvalue(), content_type=content_type)
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

    @staticmethod
    def _generate_instructions(final_model: dict) -> HttpResponse:
        """Generate an HTML instructions download."""
        image_b64 = final_model.get("image_base64", "")
        if not image_b64:
            return HttpResponse("Image introuvable.", status=404)

        context = {
            "image_base64": image_b64,
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "total_beads": final_model.get("total_beads", 0),
            "palette": final_model.get("palette", []),
            "date": datetime.now().strftime("%d/%m/%Y"),
            "model_name": "Nouveau modèle",
        }
        html = render_to_string("beadmodels/model_creation/instructions.html", context)
        response = HttpResponse(html, content_type="text/html")
        response["Content-Disposition"] = 'attachment; filename="instructions.html"'
        return response


# ---------------------------------------------------------------------------
# Wizard principal
# ---------------------------------------------------------------------------


class ModelCreatorWizard(LoginRequiredWizard):
    """Wizard complet de creation de modele a 3 etapes."""

    name = "Création de modèle de perles"
    steps = [UploadImage, ConfigureModel, SaveModel]
    session_key = "model_creation_wizard"

    def get_url_name(self):
        return "beadmodels:create"

    def get_redirect_kwargs(self):
        model_id = self.get_session_data()
        return {"model_id": model_id} if model_id else {}

    def start_wizard(self):
        self.reset_wizard()
        return self.go_to_step(1)

    def finish_wizard(self):
        self.reset_wizard()
        return redirect("beadmodels:my_models")
