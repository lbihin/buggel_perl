from django.urls import path

from .view_htmx import (
    bead_edit_color_htmx,
    bead_edit_quantity_htmx,
    bead_update_color_htmx,
    bead_update_quantity_htmx,
    stock_alert_htmx,
)
from .views import BeadCreateView, BeadListView, BeadUpdateView

app_name = "beads"

urlpatterns = [
    path("", BeadListView.as_view(), name="list"),
    path("create/", BeadCreateView.as_view(), name="create"),
    path("<int:pk>/edit/", BeadUpdateView.as_view(), name="edit"),
]

htmx_urlpatterns = [
    path(
        "<int:pk>/edit-quantity/",
        bead_edit_quantity_htmx,
        name="bead_edit_quantity_htmx",
    ),
    path(
        "<int:pk>/update-quantity/",
        bead_update_quantity_htmx,
        name="bead_update_quantity_htmx",
    ),
    path(
        "<int:pk>/edit-color/",
        bead_edit_color_htmx,
        name="bead_edit_color_htmx",
    ),
    path(
        "<int:pk>/update-color/",
        bead_update_color_htmx,
        name="bead_update_color_htmx",
    ),
    path(
        "stock-alert/",
        stock_alert_htmx,
        name="stock_alert_htmx",
    ),
]
urlpatterns += htmx_urlpatterns
