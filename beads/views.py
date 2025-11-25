# Create your views here.
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, DeleteView, ListView, UpdateView

from .forms import BeadForm
from .models import Bead


class BeadListView(LoginRequiredMixin, ListView):
    model = Bead
    template_name = "beads/table.html"
    context_object_name = "beads"

    def get_queryset(self):
        return Bead.objects.filter(creator=self.request.user).order_by("name")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        from django.conf import settings

        context["threshold"] = getattr(settings, "BEAD_LOW_QUANTITY_THRESHOLD", 20)
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
    template_name = "beads/create_or_update.html"
    success_url = reverse_lazy("beads:list")

    def form_valid(self, form):
        form.instance.creator = self.request.user
        messages.success(self.request, "Perle ajoutée avec succès!")
        return super().form_valid(form)


class BeadUpdateView(LoginRequiredMixin, UpdateView):
    model = Bead
    form_class = BeadForm
    template_name = "beads/create_or_update.html"
    success_url = reverse_lazy("beads:list")

    def get_queryset(self):
        return Bead.objects.filter(creator=self.request.user)

    def form_valid(self, form):
        messages.success(self.request, "Perle mise à jour avec succès!")
        return super().form_valid(form)


class BeadDeleteView(LoginRequiredMixin, DeleteView):
    model = Bead
    success_url = reverse_lazy("beads:list")

    def get_queryset(self):
        return Bead.objects.filter(creator=self.request.user)

    def delete(self, request, *args, **kwargs):
        messages.success(request, "Perle supprimée avec succès!")
        return super().delete(request, *args, **kwargs)
