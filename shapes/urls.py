from django.urls import path

from . import views

app_name = "shapes"

urlpatterns = [
    path("", views.shape_list, name="shape_list"),
    path("new/", views.create_shape, name="new_shape"),
    path("<int:shape_id>/edit/", views.update_shape, name="edit_shape"),
    path("<int:shape_id>/delete/", views.delete_shape, name="delete_shape"),
    # path("save/", views.shape_save, name="shape_save_new"),
    # path("<int:shape_id>/save/", views.shape_save, name="shape_save_edit"),
    # path("<int:shape_id>/delete/", views.shape_delete, name="shape_delete"),
    # path("<int:shape_id>/detail/", views.shape_detail, name="shape_detail"),
    # # Route pour l'aperçu des formes
    # path(
    #     "<int:shape_id>/preview/",
    #     views.shape_preview,
    #     name="shape_preview",
    # ),
    # # Routes pour l'édition en ligne des dimensions
    # path(
    #     "<int:shape_id>/inline-edit/",
    #     views.shape_inline_edit,
    #     name="shape_inline_edit",
    # ),
    # path(
    #     "<int:shape_id>/inline-update/",
    #     views.shape_inline_update,
    #     name="shape_inline_update",
    # ),
    # path(
    #     "<int:shape_id>/dimensions/",
    #     views.shape_dimensions,
    #     name="shape_dimensions",
    # ),
    # # Nouvelles routes pour l'édition en ligne du nom
    # path("<int:shape_id>/name/", views.shape_name, name="shape_name"),
    # path(
    #     "<int:shape_id>/name/edit/",
    #     views.shape_name_edit,
    #     name="shape_name_edit",
    # ),
    # path(
    #     "<int:shape_id>/name/update/",
    #     views.shape_name_update,
    #     name="shape_name_update",
    # ),
    # # Nouvelles routes pour l'édition en ligne du type
    # path("<int:shape_id>/type/", views.shape_type, name="shape_type"),
    # path(
    #     "<int:shape_id>/type/edit/",
    #     views.shape_type_edit,
    #     name="shape_type_edit",
    # ),
    # path(
    #     "<int:shape_id>/type/update/",
    #     views.shape_type_update,
    #     name="shape_type_update",
    # ),
]
