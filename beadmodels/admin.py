from django.contrib import admin

from .models import AppPreference, BeadBoard, BeadModel


@admin.register(BeadBoard)
class BeadBoardAdmin(admin.ModelAdmin):
    list_display = ("name", "width_pegs", "height_pegs")
    search_fields = ("name", "description")


@admin.register(BeadModel)
class BeadModelAdmin(admin.ModelAdmin):
    list_display = ("name", "creator", "created_at", "is_public")
    list_filter = ("is_public", "created_at", "creator", "board")
    search_fields = ("name", "description", "creator__username")
    readonly_fields = ("created_at", "updated_at")
    date_hierarchy = "created_at"


@admin.register(AppPreference)
class AppPreferenceAdmin(admin.ModelAdmin):
    """Interface d'administration pour les préférences de l'application."""

    fieldsets = (
        (
            "Préférences des perles",
            {
                "fields": ("bead_low_quantity_threshold",),
                "description": "Configuration des seuils d'alerte pour les perles",
            },
        ),
    )

    def has_add_permission(self, request):
        # Ne permet d'ajouter une instance que s'il n'en existe pas déjà une active
        return not AppPreference.objects.filter(is_active=True).exists()

    def has_delete_permission(self, request, obj=None):
        # Empêche la suppression des préférences
        return False
