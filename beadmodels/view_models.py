from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
    UpdateView,
)

from .forms import BeadModelForm, TransformModelForm
from .models import BeadModel


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


def my_models(request):
    user_models = BeadModel.objects.filter(creator=request.user).order_by("-created_at")
    return render(request, "beadmodels/models/my_models.html", {"models": user_models})
    return render(request, "beadmodels/models/my_models.html", {"models": user_models})
