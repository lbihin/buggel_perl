import io

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
    UpdateView,
)
from PIL import Image

from ..forms import BeadModelForm, TransformModelForm
from ..models import BeadModel


class BeadModelListView(LoginRequiredMixin, ListView):
    model = BeadModel
    template_name = "beadmodels/models/my_models.html"
    context_object_name = "models"

    def get_queryset(self):
        return BeadModel.objects.filter(creator=self.request.user).order_by(
            "-created_at"
        )


class BeadModelDetailView(DetailView):
    model = BeadModel
    template_name = "beadmodels/details.html"
    context_object_name = "model"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["transform_form"] = TransformModelForm()
        return context

    def dispatch(self, request, *args, **kwargs):
        obj = self.get_object()
        is_owner = request.user.is_authenticated and obj.creator == request.user
        if not obj.is_public and not is_owner:
            messages.error(request, "Vous n'avez pas accès à ce modèle.")
            return redirect("beadmodels:my_models")
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
        return reverse_lazy("beadmodels:details", kwargs={"pk": self.object.pk})


class BeadModelUpdateView(LoginRequiredMixin, UpdateView):
    model = BeadModel
    form_class = BeadModelForm
    template_name = "beadmodels/edit.html"

    def dispatch(self, request, *args, **kwargs):
        model = self.get_object()
        if model.creator != request.user:
            messages.error(request, "Vous n'avez pas le droit de modifier ce modèle.")
            return redirect("beadmodels:details", pk=model.pk)
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        # Supprimer l'ancienne image si nécessaire est géré ailleurs
        messages.success(self.request, "Votre modèle a été modifié avec succès!")
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("beadmodels:details", kwargs={"pk": self.object.pk})


class BeadModelDeleteView(LoginRequiredMixin, DeleteView):
    model = BeadModel
    template_name = "beadmodels/partials/delete.html"
    success_url = reverse_lazy("beadmodels:my_models")

    def dispatch(self, request, *args, **kwargs):
        obj = self.get_object()
        if obj.creator != request.user:
            messages.error(request, "Vous n'avez pas le droit de supprimer ce modèle.")
            return redirect("beadmodels:details", pk=obj.pk)
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        messages.success(self.request, "Votre modèle a été supprimé avec succès!")
        return super().form_valid(form)


class BeadModelDownloadView(LoginRequiredMixin, View):
    """Download the bead pattern image as PDF or PNG."""

    def get(self, request, pk, fmt="pdf"):
        model = get_object_or_404(BeadModel, pk=pk)

        # Access control
        is_owner = request.user.is_authenticated and model.creator == request.user
        if not model.is_public and not is_owner:
            raise Http404

        if not model.bead_pattern:
            messages.error(request, "Ce modèle n'a pas encore de motif en perles.")
            return redirect("beadmodels:details", pk=pk)

        img = Image.open(model.bead_pattern.path)
        output = io.BytesIO()
        safe_name = model.name.replace(" ", "_")[:50]

        if fmt == "png":
            img.save(output, format="PNG")
            content_type = "image/png"
            filename = f"{safe_name}.png"
        else:
            # PDF via Pillow
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(output, format="PDF", resolution=150)
            content_type = "application/pdf"
            filename = f"{safe_name}.pdf"

        output.seek(0)
        response = HttpResponse(output.getvalue(), content_type=content_type)
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response
