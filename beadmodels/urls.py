from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views
from .pixelization_wizard import PixelizationWizard

app_name = "beadmodels"

urlpatterns = [
    # Utilisation des vues basées sur des classes pour les modèles
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
    # Wizard de pixelisation (nouveau, basé sur des classes)
    path(
        "pixelization-wizard/", PixelizationWizard.as_view(), name="pixelization_wizard"
    ),
    # Routes utilisateur
    # path("settings/", views.user_settings, name="user_settings"),
    path("shapes/<int:shape_id>/edit/", views.edit_shape, name="edit_shape"),
    # Routes HTMX pour les formes
]

# Ajouter les chemins statiques seulement en développement
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
