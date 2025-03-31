from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

app_name = "beadmodels"

urlpatterns = [
    path("", views.home, name="home"),
    path("create/", views.create_model, name="create_model"),
    path("model/<int:pk>/", views.model_detail, name="model_detail"),
    path("my-models/", views.my_models, name="my_models"),
    path("register/", views.register, name="register"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
