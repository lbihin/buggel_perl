from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("logout/", views.logout, name="logout"),
    path("register/", views.register, name="register"),
    path("settings/", views.user_settings, name="user_settings"),
]

htmx_urlpatterns = []
urlpatterns += htmx_urlpatterns
