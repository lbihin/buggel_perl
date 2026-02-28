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
from django.utils.translation import gettext as _

from beads.models import Bead

from ..forms import ImageUploadForm, ModelConfigurationForm
from ..models import BeadBoard
from ..services.image_processing import (ModelResult,
                                         analyze_image_suggestions,
                                         file_to_base64, generate_model,
                                         generate_preview, save_temp_image,
                                         suggest_color_count)
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


def _fibonacci_color_values(center: int, count: int = 7) -> list[int]:
    """Generate a Fibonacci-like sequence of color values centered on *center*.

    Produces *count* values spread around *center* using Fibonacci gaps so that
    values near the suggestion are close together and values further away are
    more spread out.  All values are clamped to [2, 64].

    Example with center=7, count=7 → [3, 5, 6, 7, 8, 9, 11]
    """
    # Fibonacci-like gaps: 1, 1, 2, 3, 5, 8, 13, …
    fibs = [1, 1, 2, 3, 5, 8, 13]

    half = count // 2
    values = [center]

    # Expand outward from center using Fibonacci gaps
    for i in range(half):
        gap = fibs[i] if i < len(fibs) else fibs[-1]
        lower = center - sum(fibs[: i + 1])
        upper = center + sum(fibs[: i + 1])
        values.append(lower)
        values.append(upper)

    # Clamp, deduplicate, sort, keep only valid values
    values = sorted({max(2, min(64, v)) for v in values})

    # If we still have more than count, trim from the edges
    while len(values) > count:
        # Remove the value furthest from center
        if abs(values[0] - center) >= abs(values[-1] - center):
            values.pop(0)
        else:
            values.pop()

    return values


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

    name = _("Chargement de l'image")
    template = "beadmodels/wizard/upload_image.html"
    form_class = ImageUploadForm
    position = 1

    def handle_get(self, **kwargs):
        form = self.form_class()
        # If returning from step 2, show the previously uploaded image
        wizard_data = self.wizard.get_session_data()
        image_data = wizard_data.get("image_data", {})
        existing_image_base64 = ""
        if image_data.get("image_path"):
            try:
                existing_image_base64 = file_to_base64(image_data["image_path"])
            except Exception:
                pass

        return self.render_template(
            {
                "form": form,
                "wizard_step": self.position,
                "total_steps": 3,
                "existing_image_base64": existing_image_base64,
            }
        )

    def handle_post(self, **kwargs):
        form = self.form_class(self.wizard.request.POST, self.wizard.request.FILES)

        # If no new file uploaded but session already has an image, skip re-upload
        if not self.wizard.request.FILES.get("image"):
            wizard_data = self.wizard.get_session_data()
            if wizard_data.get("image_data", {}).get("image_path"):
                return self.wizard.go_to_next_step()

        if form.is_valid():
            image = form.cleaned_data["image"]
            stored_path = save_temp_image(image)

            # Auto-detect optimal colour count for this image
            suggested_colors = suggest_color_count(image_path=stored_path)

            # Full image analysis for smart suggestions
            suggestions = analyze_image_suggestions(image_path=stored_path)

            from dataclasses import asdict

            self.wizard.update_session_data(
                {
                    "image_data": {"image_path": stored_path},
                    "suggested_colors": suggested_colors,
                    "color_reduction": suggested_colors,
                    "suggestions": asdict(suggestions),
                }
            )
            return self.wizard.go_to_next_step()

        return self.render_template(
            {"form": form, "wizard_step": self.position, "total_steps": 3}
        )


# ---------------------------------------------------------------------------
# Etape 2 : Configuration & Preview
# ---------------------------------------------------------------------------


class ConfigureModel(WizardStep):
    """Deuxieme etape: Configuration et previsualisation du modele."""

    name = _("Configuration du modèle")
    template = "beadmodels/wizard/configure_model.html"
    form_class = ModelConfigurationForm
    position = 2

    def handle_get(self, **kwargs):
        from shapes.models import BeadShape

        wizard_data = self.wizard.get_session_data()
        image_data = wizard_data.get("image_data", {})

        if not image_data:
            messages.error(
                self.wizard.request, _("Veuillez d'abord charger une image.")
            )
            return redirect("beadmodels:create")

        suggested_colors = wizard_data.get("suggested_colors", 16)
        suggestions = wizard_data.get("suggestions", {})
        initial_data = {
            "color_reduction": wizard_data.get("color_reduction", suggested_colors),
            "use_available_colors": wizard_data.get("use_available_colors", False),
        }
        form = self.form_class(initial=initial_data)

        usr_shapes = BeadShape.objects.filter(
            creator=self.wizard.request.user
        ).order_by("name")

        selected_shape_id = wizard_data.get("shape_id")
        recommended_shape_id = None
        if not selected_shape_id:
            # Try to match the AI suggestion to an existing shape
            suggested_shape = suggestions.get("suggested_shape", "")
            matched = None
            if suggested_shape:
                matched = usr_shapes.filter(shape_type=suggested_shape).first()
            if matched:
                recommended_shape_id = matched.pk
            if not matched:
                matched = (
                    usr_shapes.filter(is_default=True).first() or usr_shapes.first()
                )
            if matched:
                selected_shape_id = matched.pk
                self.wizard.update_session_data({"shape_id": selected_shape_id})
        else:
            # Even when shape_id is already set, find the recommended one for the star
            suggested_shape = suggestions.get("suggested_shape", "")
            if suggested_shape:
                rec = usr_shapes.filter(shape_type=suggested_shape).first()
                if rec:
                    recommended_shape_id = rec.pk

        color_values = _fibonacci_color_values(suggested_colors)

        # Count how many distinct bead colours the user has
        user_bead_count = Bead.objects.filter(creator=self.wizard.request.user).count()
        use_available = wizard_data.get("use_available_colors", False)

        # When "use my colours" is active, cap the colour choices
        if use_available and user_bead_count:
            color_values = [v for v in color_values if v <= user_bead_count]
            if not color_values:
                color_values = [user_bead_count]

        # Snap suggested_colors to the nearest available radio value
        suggested_snapped = min(color_values, key=lambda v: abs(v - suggested_colors))

        # If user hasn't manually changed the color count, use the suggestion
        current_color = wizard_data.get("color_reduction", suggested_snapped)
        if current_color == suggested_colors:
            current_color = suggested_snapped
            self.wizard.update_session_data({"color_reduction": current_color})

        preview_kwargs = _build_preview_kwargs(wizard_data, self.wizard.request.user)
        preview_result = generate_preview(**preview_kwargs)
        original_image_base64 = file_to_base64(image_data.get("image_path"))

        # Build suggestion labels for the UI
        shape_labels = {
            "circle": _("Rond"),
            "square": _("Carré"),
            "rectangle": _("Rectangle"),
        }
        suggestion_shape_label = shape_labels.get(
            suggestions.get("suggested_shape", ""), ""
        )

        step_ctx = self.get_context_data()
        context = {
            "form": form,
            "image_base64": original_image_base64,
            "preview_image_base64": preview_result.image_base64,
            "user_shapes": usr_shapes,
            "selected_shape_id": wizard_data.get("shape_id"),
            "recommended_shape_id": recommended_shape_id,
            "color_values": color_values,
            "suggested_colors": suggested_snapped,
            "suggestions": suggestions,
            "suggestion_shape_label": suggestion_shape_label,
            "user_bead_count": user_bead_count,
            "step_nr": self.position,
            "step_name": self.name,
            "total_steps": step_ctx.get("total_steps"),
        }
        return self.render_template(context)

    def handle_post(self, **kwargs):
        wizard_data = self.wizard.get_session_data()
        request = self.wizard.request

        # Navigation: retour
        direction = request.POST.get("q")
        if direction == "previous":
            return self.wizard.go_to_previous_step()

        # ---------- HTMX live-preview (checkbox "use_available_colors") ----------
        if getattr(request, "htmx", False) and direction != "next":
            # This HTMX path is only reached by the checkbox toggle
            # (shape/color radios use dedicated endpoints).
            # An unchecked checkbox is simply absent from POST → False.
            shape_id = request.POST.get("shape_id") or wizard_data.get("shape_id", "")
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
                "beadmodels/wizard/partials/preview.html",
                {"preview_image_base64": preview_result.image_base64},
            )
            return HttpResponse(html)

        # ---------- Générer le modèle final → étape 3 ----------
        if direction == "next" or "generate" in request.POST:
            return self._handle_generate(wizard_data)

        # Fallback: re-render step 2
        preview_kwargs = _build_preview_kwargs(wizard_data, request.user)
        preview_result = generate_preview(**preview_kwargs)
        form = self.form_class(request.POST)
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
        shape_id = request.POST.get("shape_id") or wizard_data.get("shape_id", "")
        color_reduction = _safe_int(
            request.POST.get("color_reduction"),
            _safe_int(wizard_data.get("color_reduction"), 16),
        )
        use_available = (
            request.POST.get("use_available_colors") == "on"
            if "use_available_colors" in request.POST
            else wizard_data.get("use_available_colors", False)
        )

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

    name = _("Finalisation")
    template = "beadmodels/wizard/final_step.html"
    position = 3

    def handle_get(self, **kwargs):
        wizard_data = self.wizard.get_session_data()
        final_model = wizard_data.get("final_model", {})

        # Shape info
        shape_name, shape_type = _("Standard"), "rectangle"
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

        default_name = _("Modèle du %(date)s") % {
            "date": datetime.now().strftime("%d %B %Y")
        }

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

        # Compute bead counts excluding background
        total_beads = final_model.get("total_beads", 0)
        bg_beads = sum(c["count"] for c in palette if c.get("is_background"))
        useful_beads = total_beads - bg_beads
        useful_colors = sum(1 for c in palette if not c.get("is_background"))

        context = {
            "image_base64": final_image_base64,
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "shape_id": final_model.get("shape_id"),
            "shape_name": shape_name,
            "shape_type": shape_type,
            "total_beads": useful_beads,
            "total_beads_raw": total_beads,
            "fill_ratio": final_model.get("fill_ratio", 1.0),
            "palette": palette,
            "beads_count": useful_colors,
            "wizard_step": self.position,
            "total_steps": 3,
            "form": form,
            "user_has_beads": user_beads.exists(),
        }
        return self.render_template(context)

    def handle_post(self, **kwargs):
        from ..forms import BeadModelFinalizeForm

        wizard_data = self.wizard.get_session_data()
        final_model = wizard_data.get("final_model", {})

        # Downloads
        if "download" in self.wizard.request.POST:
            fmt = self.wizard.request.POST.get("download", "png")
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
                # Set board from session (not in form anymore)
                board_id = wizard_data.get("board_id")
                if board_id:
                    new_model.board_id = board_id
                    new_model.save(update_fields=["board"])
            except Exception as e:
                logger.exception("Erreur lors de la sauvegarde du modele: %s", e)
                messages.error(
                    self.wizard.request,
                    _("Une erreur est survenue lors de la sauvegarde du modèle."),
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

        palette = final_model.get("palette", [])
        total_beads = final_model.get("total_beads", 0)
        bg_beads = sum(c["count"] for c in palette if c.get("is_background"))
        useful_beads = total_beads - bg_beads
        useful_colors = sum(1 for c in palette if not c.get("is_background"))

        context = {
            "image_base64": final_model.get("image_base64", ""),
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "shape_id": final_model.get("shape_id"),
            "shape_name": shape_name,
            "shape_type": shape_type,
            "total_beads": useful_beads,
            "palette": palette,
            "beads_count": useful_colors,
            "wizard_step": self.position,
            "total_steps": 3,
            "form": form,
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
        # Reload from disk if base64 was stripped by _store_final_image
        if not image_b64 and final_model.get("image_path"):
            try:
                image_b64 = file_to_base64(final_model["image_path"])
            except Exception:
                pass
        if not image_b64:
            return HttpResponse(_("Image introuvable."), status=404)

        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        output = io.BytesIO()

        if fmt.lower() == "pdf":
            # Convert to PDF using Pillow
            rgb_img = img.convert("RGB") if img.mode == "RGBA" else img
            rgb_img.save(output, format="PDF", resolution=150)
            content_type, ext = "application/pdf", "pdf"
        elif fmt.lower() in ("jpg", "jpeg"):
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
        if not image_b64 and final_model.get("image_path"):
            try:
                image_b64 = file_to_base64(final_model["image_path"])
            except Exception:
                pass
        if not image_b64:
            return HttpResponse("Image introuvable.", status=404)

        context = {
            "image_base64": image_b64,
            "grid_width": final_model.get("grid_width", 29),
            "grid_height": final_model.get("grid_height", 29),
            "total_beads": final_model.get("total_beads", 0),
            "palette": final_model.get("palette", []),
            "date": datetime.now().strftime("%d/%m/%Y"),
            "model_name": _("Nouveau modèle"),
        }
        html = render_to_string("beadmodels/wizard/instructions.html", context)
        response = HttpResponse(html, content_type="text/html")
        response["Content-Disposition"] = 'attachment; filename="instructions.html"'
        return response


# ---------------------------------------------------------------------------
# Wizard principal
# ---------------------------------------------------------------------------


class ModelCreatorWizard(LoginRequiredWizard):
    """Wizard complet de creation de modele a 3 etapes."""

    name = _("Création de modèle de perles")
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

    def dispatch(self, request, *args, **kwargs):
        """Override dispatch to handle model_id → skip to step 2.

        On a fresh GET (no ``w`` or ``model_id`` query param), reset the wizard
        so the user always starts at step 1 when navigating from the navbar.
        Internal redirects via ``go_to_step`` include ``?w=1``.
        """
        self.request = request

        # Fresh GET without internal wizard marker → reset to step 1
        if (
            request.method == "GET"
            and not request.GET.get("model_id")
            and not request.GET.get("q")
            and not request.GET.get("w")
        ):
            self.reset_wizard()

        model_id = request.GET.get("model_id")
        if model_id and request.method == "GET":
            try:
                from ..models import BeadModel

                obj = BeadModel.objects.get(pk=model_id, creator=request.user)
                # Reset wizard and pre-populate with existing image
                self.reset_wizard()
                image_b64 = file_to_base64(obj.original_image.path)
                stored_path = save_temp_image(
                    __import__(
                        "django.core.files.base", fromlist=["ContentFile"]
                    ).ContentFile(base64.b64decode(image_b64), name="original.png")
                )
                session_data = {
                    "image_data": {"image_path": stored_path},
                    "source_model_id": int(model_id),
                }
                # Restore generation settings from metadata if available
                meta = obj.metadata or {}
                if meta.get("shape_id"):
                    session_data["shape_id"] = meta["shape_id"]
                if meta.get("final_color_reduction"):
                    session_data["color_reduction"] = meta["final_color_reduction"]
                self.set_session_data(session_data)
                self.set_current_step_number(2)
                step = self.get_current_step()
                return step.handle_get(**kwargs)
            except BeadModel.DoesNotExist:
                messages.error(request, _("Modèle introuvable."))
                return redirect("beadmodels:my_models")

        return super().dispatch(request, *args, **kwargs)
