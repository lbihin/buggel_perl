from django.contrib.auth.models import User
from django.db import models


class BeadModel(models.Model):
    name = models.CharField(max_length=200, verbose_name="Nom")
    description = models.TextField(blank=True, verbose_name="Description")
    creator = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Créateur")
    created_at = models.DateTimeField(
        auto_now_add=True, verbose_name="Date de création"
    )
    updated_at = models.DateTimeField(
        auto_now=True, verbose_name="Date de modification"
    )
    is_public = models.BooleanField(default=False, verbose_name="Public")
    original_image = models.ImageField(
        upload_to="originals/", verbose_name="Image originale"
    )
    bead_pattern = models.ImageField(
        upload_to="patterns/", null=True, blank=True, verbose_name="Motif en perles"
    )
    board = models.ForeignKey(
        "BeadBoard",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="Support de perles",
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Métadonnées",
        help_text="Paramètres de génération (palette, dimensions, etc.)",
    )

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Modèle de perles"
        verbose_name_plural = "Modèles de perles"
        ordering = ["-created_at"]





class BeadBoard(models.Model):
    name = models.CharField(max_length=100, verbose_name="Nom")
    width_pegs = models.IntegerField(verbose_name="Nombre de picots en largeur")
    height_pegs = models.IntegerField(verbose_name="Nombre de picots en hauteur")
    description = models.TextField(verbose_name="Description", blank=True)

    def __str__(self):
        return f"{self.name} ({self.width_pegs}x{self.height_pegs})"

    class Meta:
        verbose_name = "Support de perles"
        verbose_name_plural = "Supports de perles"


class AppPreference(models.Model):
    """Modèle pour stocker les préférences globales de l'application."""

    bead_low_quantity_threshold = models.PositiveIntegerField(
        default=20,
        verbose_name="Seuil d'alerte pour les perles en faible quantité",
        help_text="Quantité minimale de perles en dessous de laquelle une alerte sera affichée",
    )

    # Champ pour limiter à une seule instance
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Préférence de l'application"
        verbose_name_plural = "Préférences de l'application"

    def __str__(self):
        return "Préférences de l'application"

    def save(self, *args, **kwargs):
        """Assure qu'il n'y a qu'une seule instance des préférences."""
        # Si l'instance est nouvelle, désactiver toutes les autres
        if not self.pk:
            AppPreference.objects.update(is_active=False)
            self.is_active = True
        super().save(*args, **kwargs)

    @classmethod
    def get_instance(cls):
        """Récupère l'instance active ou crée une nouvelle instance avec les valeurs par défaut."""
        try:
            return cls.objects.get(is_active=True)
        except cls.DoesNotExist:
            # Créer une nouvelle instance avec les valeurs par défaut
            prefs = cls()
            prefs.save()
            return prefs
