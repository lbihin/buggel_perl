from django.contrib.auth.models import User
from django.db import models


class BeadModel(models.Model):
    title = models.CharField(max_length=200, verbose_name="Titre")
    description = models.TextField(blank=True, verbose_name="Description")
    original_image = models.ImageField(upload_to='originals/', verbose_name="Image originale")
    bead_pattern = models.ImageField(upload_to='patterns/', verbose_name="Motif en perles", blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Créé le")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Modifié le")
    creator = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Créateur")
    grid_size = models.IntegerField(default=32, verbose_name="Taille de la grille")
    is_public = models.BooleanField(default=True, verbose_name="Public")

    class Meta:
        verbose_name = "Modèle de perles"
        verbose_name_plural = "Modèles de perles"
        ordering = ['-created_at']

    def __str__(self):
        return self.title
