from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.translation import gettext_lazy as _


class Bead(models.Model):
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name="beads")
    name = models.CharField(max_length=100, verbose_name=_("Nom"))
    red = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(255)],
        verbose_name=_("Rouge"),
    )
    green = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(255)],
        verbose_name=_("Vert"),
    )
    blue = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(255)],
        verbose_name=_("Bleu"),
    )
    quantity = models.PositiveIntegerField(
        null=True, blank=True, verbose_name=_("Quantité")
    )
    notes = models.TextField(blank=True, verbose_name=_("Notes"))
    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Créé le"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Modifié le"))

    class Meta:
        unique_together = ["creator", "name"]
        ordering = ["name"]
        verbose_name = _("Perle")
        verbose_name_plural = _("Perles")

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
                return _("Noir")
            if max_val < 0.5:
                return _("Gris foncé")
            if max_val < 0.8:
                return _("Gris")
            return _("Blanc")

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
            return _("Vert")

        # Catégoriser selon la teinte avec des plages ajustées
        if 0 <= hue < 30 or 330 <= hue <= 360:
            return _("Rouge")
        elif 30 <= hue < 65:  # Réduit la plage de l'orange
            return _("Orange")
        elif 65 <= hue < 120:  # Ajuste la plage du jaune
            return _("Jaune")
        elif 120 <= hue < 180:  # Élargit la plage du vert vers le bas
            return _("Vert")
        elif 180 <= hue < 240:  # Ajuste la plage du cyan
            return _("Cyan")
        elif 240 <= hue < 300:  # Ajuste la plage du bleu
            return _("Bleu")
        elif 300 <= hue <= 360:  # Élargit la plage du violet
            return _("Violet")

        return _("Autres")
