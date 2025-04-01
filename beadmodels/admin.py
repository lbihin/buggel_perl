from django.contrib import admin

from .models import BeadBoard, BeadModel


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
