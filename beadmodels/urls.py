from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect
from django.urls import path

from . import views
from .model_creation_wizard import ModelCreationWizard


# Redirection vers la page d'accueil principale
def redirect_to_home(request):
    return redirect("home")


app_name = "beadmodels"

urlpatterns = [
    path("create/", views.BeadModelCreateView.as_view(), name="create_model"),
    path("model/<int:pk>/", views.BeadModelDetailView.as_view(), name="model_detail"),
    path(
        "model/<int:pk>/edit/", views.BeadModelUpdateView.as_view(), name="edit_model"
    ),
    path(
        "model/<int:pk>/delete/",
        views.BeadModelDeleteView.as_view(),
        name="delete_model",
    ),
    path("my-models/", views.BeadModelListView.as_view(), name="my_models"),
    # Route d'accueil locale
    path("home/", redirect_to_home, name="home"),
    # Gestion des perles avec des vues basées sur des classes
    path("beads/", views.BeadListView.as_view(), name="bead_list"),
    path("beads/add/", views.BeadCreateView.as_view(), name="bead_create"),
    path("beads/<int:pk>/edit/", views.BeadUpdateView.as_view(), name="bead_update"),
    path("beads/<int:pk>/delete/", views.BeadDeleteView.as_view(), name="bead_delete"),
    # Routes AJAX et traitement d'image
    path("model/<int:pk>/transform/", views.transform_image, name="transform_image"),
    path(
        "model/<int:pk>/save-transformation/",
        views.save_transformation,
        name="save_transformation",
    ),
    # Nouveau wizard de création de modèle (à 3 étapes)
    path(
        "model-creation/", ModelCreationWizard.as_view(), name="model_creation_wizard"
    ),
    # Routes HTMX pour les perles
    path(
        "beads/<int:pk>/edit-quantity/",
        views.bead_edit_quantity_htmx,
        name="bead_edit_quantity_htmx",
    ),
    path(
        "beads/<int:pk>/update-quantity/",
        views.bead_update_quantity_htmx,
        name="bead_update_quantity_htmx",
    ),
    path(
        "beads/<int:pk>/edit-color/",
        views.bead_edit_color_htmx,
        name="bead_edit_color_htmx",
    ),
    path(
        "beads/<int:pk>/update-color/",
        views.bead_update_color_htmx,
        name="bead_update_color_htmx",
    ),
    # Routes HTMX pour les formes
]

# Ajouter les chemins statiques seulement en développement
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
