from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect
from django.urls import path

from . import views
from .views import wizard_htmx_views as htmx_views


# Redirection vers la page d'accueil principale
def redirect_to_home(request):
    return redirect("home")


app_name = "beadmodels"

urlpatterns = [
    # path("create/", views.BeadModelCreateView.as_view(), name="create_model"),
    path("", views.BeadModelListView.as_view(), name="my_models"),
    path("model/<int:pk>/", views.BeadModelDetailView.as_view(), name="details"),
    path("model/<int:pk>/edit/", views.BeadModelUpdateView.as_view(), name="edit"),
    path("model/<int:pk>/delete/", views.BeadModelDeleteView.as_view(), name="delete"),
    # # Route d'accueil locale
    # path("home/", redirect_to_home, name="home"),
    # # Gestion des perles avec des vues basées sur des classes
    # path("beads/", views.BeadListView.as_view(), name="bead_list"),
    # path("beads/add/", views.BeadCreateView.as_view(), name="bead_create"),
    # path("beads/<int:pk>/edit/", views.BeadUpdateView.as_view(), name="bead_update"),
    # path("beads/<int:pk>/delete/", views.BeadDeleteView.as_view(), name="bead_delete"),
    # # Routes AJAX et traitement d'image
    # path("model/<int:pk>/transform/", views.transform_image, name="transform_image"),
    # path(
    #     "model/<int:pk>/save-transformation/",
    #     views.save_transformation,
    #     name="save_transformation",
    # ),
    # Nouveau wizard de création de modèle (à 3 étapes)
    path("create/", views.ModelCreatorWizard.as_view(), name="create"),
    # # Routes HTMX pour les perles
    # # Routes HTMX pour les formes
]

urlpatterns_htmx = [
    path(
        "hx/shape/<int:pk>/change/",
        htmx_views.change_shape_hx_view,
        name="hx-config-change-shape",
    ),
]
urlpatterns += urlpatterns_htmx

# Ajouter les chemins statiques seulement en développement
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
