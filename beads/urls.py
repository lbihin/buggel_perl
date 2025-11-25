from django.urls import path

from .view_htmx import *
from .views import *

app_name = "beads"

urlpatterns = [
    path("", BeadListView.as_view(), name="list"),
    path("create/", BeadCreateView.as_view(), name="create"),
    path("edit/", BeadUpdateView.as_view(), name="edit"),
]

htmx_urlpatterns = [
    path(
        "beads/<int:pk>/edit-quantity/",
        bead_edit_quantity_htmx,
        name="bead_edit_quantity_htmx",
    ),
    path(
        "beads/<int:pk>/update-quantity/",
        bead_update_quantity_htmx,
        name="bead_update_quantity_htmx",
    ),
    path(
        "beads/<int:pk>/edit-color/",
        bead_edit_color_htmx,
        name="bead_edit_color_htmx",
    ),
    path(
        "beads/<int:pk>/update-color/",
        bead_update_color_htmx,
        name="bead_update_color_htmx",
    ),
]
urlpatterns += htmx_urlpatterns
