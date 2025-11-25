from django.contrib import admin
from django.utils.html import format_html

from .models import Bead


# Register your models here.
@admin.register(Bead)
class BeadAdmin(admin.ModelAdmin):
    list_display = ("name", "creator", "color_preview", "quantity")
    list_filter = (
        "creator",
    )  # Suppression de color_category car c'est une propriété, pas un champ
    search_fields = ("name", "notes", "creator__username")
    readonly_fields = ("created_at", "updated_at", "color_preview")

    def color_preview(self, obj):
        """Affiche un carré coloré avec la couleur de la perle."""
        return format_html(
            '<div style="width:20px;height:20px;background-color:{}"></div>',
            obj.get_rgb_color(),
        )

    color_preview.short_description = "Couleur"
