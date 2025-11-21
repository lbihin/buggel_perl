from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .forms import BeadModelForm
from .models import BeadBoard


@login_required
@require_GET
def test_save_form(request):
    """Vue de test pour vérifier le formulaire de sauvegarde."""
    # Créer une instance du formulaire BeadModelForm
    form = BeadModelForm(initial={"name": "Test de formulaire", "is_public": False})

    # Récupérer la liste des supports de perles
    boards = BeadBoard.objects.all()

    # Rendu du template avec le formulaire
    context = {
        "form": form,
        "boards": boards,
    }
    return render(request, "beadmodels/model_creation/test_save_form.html", context)
