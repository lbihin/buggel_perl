from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import shapes_views, views
from .pixelization_wizard import PixelizationWizard

app_name = "beadmodels"

urlpatterns = [
    path("", views.home, name="home"),
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
    path("settings/", views.user_settings, name="user_settings"),
    path(
        "settings/delete-shape/<int:shape_id>/", views.delete_shape, name="delete_shape"
    ),
    path("shapes/<int:shape_id>/edit/", views.edit_shape, name="edit_shape"),
    # Routes HTMX pour les formes
    path("shapes/", shapes_views.shape_list, name="shape_list"),
    path("shapes/form/", shapes_views.shape_form, name="shape_form_new"),
    path(
        "shapes/<int:shape_id>/form/", shapes_views.shape_form, name="shape_form_edit"
    ),
    path("shapes/save/", shapes_views.shape_save, name="shape_save_new"),
    path(
        "shapes/<int:shape_id>/save/", shapes_views.shape_save, name="shape_save_edit"
    ),
    path(
        "shapes/<int:shape_id>/delete/", shapes_views.shape_delete, name="shape_delete"
    ),
    path(
        "shapes/<int:shape_id>/detail/", shapes_views.shape_detail, name="shape_detail"
    ),
    # Route pour l'aperçu des formes
    path(
        "shapes/<int:shape_id>/preview/",
        shapes_views.shape_preview,
        name="shape_preview",
    ),
    # Routes pour l'édition en ligne des dimensions
    path(
        "shapes/<int:shape_id>/inline-edit/",
        shapes_views.shape_inline_edit,
        name="shape_inline_edit",
    ),
    path(
        "shapes/<int:shape_id>/inline-update/",
        shapes_views.shape_inline_update,
        name="shape_inline_update",
    ),
    path(
        "shapes/<int:shape_id>/dimensions/",
        shapes_views.shape_dimensions,
        name="shape_dimensions",
    ),
    # Nouvelles routes pour l'édition en ligne du nom
    path("shapes/<int:shape_id>/name/", shapes_views.shape_name, name="shape_name"),
    path(
        "shapes/<int:shape_id>/name/edit/",
        shapes_views.shape_name_edit,
        name="shape_name_edit",
    ),
    path(
        "shapes/<int:shape_id>/name/update/",
        shapes_views.shape_name_update,
        name="shape_name_update",
    ),
    # Nouvelles routes pour l'édition en ligne du type
    path("shapes/<int:shape_id>/type/", shapes_views.shape_type, name="shape_type"),
    path(
        "shapes/<int:shape_id>/type/edit/",
        shapes_views.shape_type_edit,
        name="shape_type_edit",
    ),
    path(
        "shapes/<int:shape_id>/type/update/",
        shapes_views.shape_type_update,
        name="shape_type_update",
    ),
]

# Ajouter les chemins statiques seulement en développement
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
