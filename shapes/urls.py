from django.urls import path

from . import views

app_name = "shapes"

urlpatterns = [
    path("", views.shape_list, name="shape_list"),
    path("new/", views.create_shape, name="shape_new"),
    path("<int:shape_id>/edit/", views.update_shape, name="edit_shape"),
    path("<int:shape_id>/delete/", views.delete_shape, name="delete_shape"),
    # Routes HTMX pour le formulaire de création/édition
    path("create/", views.shape_form_hx_view, name="shape_create_hx"),
    path("<int:shape_id>/form/", views.shape_form_hx_view, name="shape_edit_hx"),
    # Routes unifiées pour la sauvegarde/mise à jour
    path("save/", views.shape_save_or_update_hx_view, name="shape_save_new"),
    path(
        "<int:shape_id>/save/",
        views.shape_save_or_update_hx_view,
        name="shape_save_edit",
    ),
    # Route pour le formulaire d'ajout en ligne
    path("get-inline-add-form/", views.get_inline_add_form, name="get_inline_add_form"),
    # Route pour la suppression HTMX
    path(
        "<int:shape_id>/delete-hx/", views.shape_delete_hx_view, name="shape_delete_hx"
    ),
    # Routes pour l'édition en ligne des dimensions
    path(
        "<int:shape_id>/inline-edit/", views.shape_inline_edit, name="shape_inline_edit"
    ),
    path(
        "<int:shape_id>/inline-update/",
        views.shape_inline_update,
        name="shape_inline_update",
    ),
    path("<int:shape_id>/dimensions/", views.shape_dimensions, name="shape_dimensions"),
    # Routes pour l'édition en ligne du nom
    path("<int:shape_id>/name/", views.shape_name, name="shape_name"),
    path("<int:shape_id>/name/edit/", views.shape_name_edit, name="shape_name_edit"),
    path(
        "<int:shape_id>/name/update/", views.shape_name_update, name="shape_name_update"
    ),
    # Routes pour l'édition en ligne du type
    path("<int:shape_id>/type/", views.shape_type, name="shape_type"),
    path("<int:shape_id>/type/edit/", views.shape_type_edit, name="shape_type_edit"),
    path(
        "<int:shape_id>/type/update/", views.shape_type_update, name="shape_type_update"
    ),
]

htmx_urlpatterns = []
urlpatterns += htmx_urlpatterns
