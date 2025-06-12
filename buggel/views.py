from django.shortcuts import render

from beadmodels.models import BeadModel


def home(request):
    public_models = BeadModel.objects.filter(is_public=True).order_by("-created_at")[
        :12
    ]
    return render(request, "home.html", {"models": public_models})
