from django.urls import path

from . import views

app_name = "shapes"

urlpatterns = [
    # Vues principales
    path("", views.ShapeListView.as_view(), name="shape_list"),
    path("columns/", views.ShapeListColumnsView.as_view(), name="shape_list_columns"),
    # CRUD classique
    path("create/", views.create_shape, name="create_shape"),
    path("<int:shape_id>/update/", views.update_shape, name="update_shape"),
    path("<int:shape_id>/delete/", views.delete_shape, name="delete_shape"),
    # HTMX — formulaires de création/édition
    path("hx/form/", views.shape_form_hx_view, name="shape_create_hx"),
    path("<int:shape_id>/hx/form/", views.shape_form_hx_view, name="shape_edit_hx"),
    # HTMX — sauvegarde unifiée
    path("hx/save/", views.shape_save_or_update_hx_view, name="shape_save_new"),
    path(
        "<int:shape_id>/hx/save/",
        views.shape_save_or_update_hx_view,
        name="shape_save_edit",
    ),
    # HTMX — suppression
    path(
        "<int:shape_id>/hx/delete/",
        views.shape_delete_hx_view,
        name="shape_delete_hx",
    ),
    # HTMX — formulaire d'ajout inline
    path("hx/inline-add-form/", views.get_inline_add_form, name="get_inline_add_form"),
    # HTMX — édition inline des dimensions
    path(
        "<int:shape_id>/hx/inline-edit/",
        views.shape_inline_edit,
        name="shape_inline_edit",
    ),
    path(
        "<int:shape_id>/hx/inline-update/",
        views.shape_inline_update,
        name="shape_inline_update",
    ),
    path(
        "<int:shape_id>/hx/dimensions/",
        views.shape_dimensions,
        name="shape_dimensions",
    ),
    # HTMX — édition inline du nom
    path("<int:shape_id>/hx/name/", views.shape_name, name="shape_name"),
    path("<int:shape_id>/hx/name/edit/", views.shape_name_edit, name="shape_name_edit"),
    path(
        "<int:shape_id>/hx/name/update/",
        views.shape_name_update,
        name="shape_name_update",
    ),
    # HTMX — édition inline du type
    path("<int:shape_id>/hx/type/", views.shape_type, name="shape_type"),
    path("<int:shape_id>/hx/type/edit/", views.shape_type_edit, name="shape_type_edit"),
    path(
        "<int:shape_id>/hx/type/update/",
        views.shape_type_update,
        name="shape_type_update",
    ),
    # HTMX — Row-level inline editing (new table approach)
    path(
        "<int:shape_id>/hx/edit-row/",
        views.shape_edit_row_htmx,
        name="shape_edit_row_htmx",
    ),
    path(
        "<int:shape_id>/hx/save-row/",
        views.shape_save_row_htmx,
        name="shape_save_row_htmx",
    ),
    path(
        "<int:shape_id>/hx/delete-row/",
        views.shape_delete_row_htmx,
        name="shape_delete_row_htmx",
    ),
    path(
        "hx/new-row/",
        views.shape_new_row_htmx,
        name="shape_new_row_htmx",
    ),
    path(
        "hx/create-inline/",
        views.shape_create_inline_htmx,
        name="shape_create_inline_htmx",
    ),
]
