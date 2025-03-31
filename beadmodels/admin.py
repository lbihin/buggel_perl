from django.contrib import admin

from .models import BeadModel


@admin.register(BeadModel)
class BeadModelAdmin(admin.ModelAdmin):
    list_display = ('title', 'creator', 'created_at', 'is_public')
    list_filter = ('is_public', 'created_at', 'creator')
    search_fields = ('title', 'description', 'creator__username')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
