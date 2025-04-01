from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

app_name = "beadmodels"

urlpatterns = [
    path("", views.home, name="home"),
    path("create/", views.create_model, name="create_model"),
    path("model/<int:pk>/", views.model_detail, name="model_detail"),
    path("model/<int:pk>/edit/", views.edit_model, name="edit_model"),
    path("model/<int:pk>/delete/", views.delete_model, name="delete_model"),
    path("model/<int:pk>/transform/", views.transform_image, name="transform_image"),
    path(
        "model/<int:pk>/save-transformation/",
        views.save_transformation,
        name="save_transformation",
    ),
    path("my-models/", views.my_models, name="my_models"),
    path("register/", views.register, name="register"),
    path("settings/", views.user_settings, name="user_settings"),
    path("settings/save-shape/", views.save_shape, name="save_shape"),
    path(
        "settings/delete-shape/<int:shape_id>/", views.delete_shape, name="delete_shape"
    ),
    path("shapes/create/", views.create_shape, name="create_shape"),
    path("shapes/<int:shape_id>/edit/", views.edit_shape, name="edit_shape"),
    path("shapes/<int:shape_id>/delete/", views.delete_shape, name="delete_shape"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
