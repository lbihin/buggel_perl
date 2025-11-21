from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
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


class Bead(models.Model):
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name="beads")
    name = models.CharField(max_length=100, verbose_name="Nom")
    red = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(255)], verbose_name="Rouge"
    )
    green = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(255)], verbose_name="Vert"
    )
    blue = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(255)], verbose_name="Bleu"
    )
    quantity = models.PositiveIntegerField(
        null=True, blank=True, verbose_name="Quantité"
    )
    notes = models.TextField(blank=True, verbose_name="Notes")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Créé le")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Modifié le")

    class Meta:
        unique_together = ["creator", "name"]
        ordering = ["name"]
        verbose_name = "Perle"
        verbose_name_plural = "Perles"

    def __str__(self):
        return f"{self.name} ({self.creator.username})"

    def get_color_display(self):
        return self.name

    def get_rgb_color(self):
        return f"rgb({self.red}, {self.green}, {self.blue})"

    def get_hex_color(self):
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"

    @property
    def color_category(self):
        # Convertir RGB en teinte (hue)
        r, g, b = self.red / 255, self.green / 255, self.blue / 255
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val

        # Si toutes les valeurs sont égales, c'est une teinte de gris
        if delta == 0:
            if max_val < 0.2:
                return "Noir"
            if max_val < 0.5:
                return "Gris foncé"
            if max_val < 0.8:
                return "Gris"
            return "Blanc"

        # Calculer la teinte
        if max_val == r:
            hue = 60 * ((g - b) / delta)
        elif max_val == g:
            hue = 60 * (2 + (b - r) / delta)
        else:
            hue = 60 * (4 + (r - g) / delta)

        # Normaliser la teinte
        if hue < 0:
            hue += 360

        # Ajouter une vérification spécifique pour les verts avec dominante verte claire
        # Si la composante verte est significativement plus élevée que les autres
        if g > 0.4 and g > 1.5 * r and g > 1.5 * b:
            return "Vert"

        # Catégoriser selon la teinte avec des plages ajustées
        if 0 <= hue < 30 or 330 <= hue <= 360:
            return "Rouge"
        elif 30 <= hue < 65:  # Réduit la plage de l'orange
            return "Orange"
        elif 65 <= hue < 120:  # Ajuste la plage du jaune
            return "Jaune"
        elif 120 <= hue < 180:  # Élargit la plage du vert vers le bas
            return "Vert"
        elif 180 <= hue < 240:  # Ajuste la plage du cyan
            return "Cyan"
        elif 240 <= hue < 300:  # Ajuste la plage du bleu
            return "Bleu"
        elif 300 <= hue <= 360:  # Élargit la plage du violet
            return "Violet"

        return "Autres"


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
