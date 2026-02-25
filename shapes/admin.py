from django.contrib import admin

from .models import BeadShape


@admin.register(BeadShape)
class BeadShapeAdmin(admin.ModelAdmin):
    list_display = ("name", "shape_type", "creator", "is_default", "is_shared")
    list_filter = ("shape_type", "is_default", "is_shared")
    search_fields = ("name",)
