from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views
from .views import wizard_htmx_views as htmx_views

app_name = "beadmodels"

urlpatterns = [
    path("", views.BeadModelListView.as_view(), name="my_models"),
    path("model/<int:pk>/", views.BeadModelDetailView.as_view(), name="details"),
    path("model/<int:pk>/edit/", views.BeadModelUpdateView.as_view(), name="edit"),
    path("model/<int:pk>/delete/", views.BeadModelDeleteView.as_view(), name="delete"),
    # Wizard de création de modèle (à 3 étapes)
    path("create/", views.ModelCreatorWizard.as_view(), name="create"),
    # Routes HTMX pour le wizard
    path(
        "hx/shape/<int:pk>/change/",
        htmx_views.change_shape_hx_view,
        name="hx-config-change-shape",
    ),
    path(
        "hx/colors/<int:color_reduction>/change/",
        htmx_views.change_max_colors_hx_view,
        name="hx-config-change-max-colors",
    ),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
