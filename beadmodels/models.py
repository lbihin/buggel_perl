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

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Modèle de perles"
        verbose_name_plural = "Modèles de perles"
        ordering = ["-created_at"]


class BeadShape(models.Model):
    SHAPE_TYPES = [
        ("rectangle", "Rectangle"),
        ("square", "Carré"),
        ("circle", "Rond"),
    ]

    name = models.CharField(max_length=100, verbose_name="Nom")
    shape_type = models.CharField(
        max_length=20, choices=SHAPE_TYPES, verbose_name="Type de forme"
    )
    width = models.IntegerField(
        validators=[MinValueValidator(1)], null=True, blank=True, verbose_name="Largeur"
    )
    height = models.IntegerField(
        validators=[MinValueValidator(1)], null=True, blank=True, verbose_name="Hauteur"
    )
    size = models.IntegerField(
        validators=[MinValueValidator(1)], null=True, blank=True, verbose_name="Taille"
    )
    diameter = models.IntegerField(
        validators=[MinValueValidator(1)],
        null=True,
        blank=True,
        verbose_name="Diamètre",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Créé le")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Modifié le")
    is_default = models.BooleanField(default=False, verbose_name="Forme par défaut")
    is_shared = models.BooleanField(default=False, verbose_name="Forme partagée")
    creator = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_shapes",
        verbose_name="Créateur",
    )

    class Meta:
        verbose_name = "Forme de perles"
        verbose_name_plural = "Formes de perles"
        ordering = ["name"]
        unique_together = ["name", "creator"]

    def __str__(self):
        return self.name

    def get_dimensions_display(self):
        if self.shape_type == "rectangle":
            return f"{self.width}×{self.height}"
        elif self.shape_type == "square":
            return f"{self.size}×{self.size}"
        elif self.shape_type == "circle":
            return f"∅{self.diameter}"
        return ""

    def get_parameters(self):
        if self.shape_type == "rectangle":
            return {"width": self.width, "height": self.height}
        elif self.shape_type == "square":
            return {"size": self.size}
        elif self.shape_type == "circle":
            return {"diameter": self.diameter}
        return {}


class CustomShape(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="custom_shapes",
        verbose_name="Utilisateur",
    )
    base_shape = models.ForeignKey(
        BeadShape,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="Forme de base",
    )
    name = models.CharField(max_length=100, verbose_name="Nom")
    shape_type = models.CharField(
        max_length=20, choices=BeadShape.SHAPE_TYPES, verbose_name="Type de forme"
    )
    width = models.IntegerField(
        validators=[MinValueValidator(1)], null=True, blank=True, verbose_name="Largeur"
    )
    height = models.IntegerField(
        validators=[MinValueValidator(1)], null=True, blank=True, verbose_name="Hauteur"
    )
    size = models.IntegerField(
        validators=[MinValueValidator(1)], null=True, blank=True, verbose_name="Taille"
    )
    diameter = models.IntegerField(
        validators=[MinValueValidator(1)],
        null=True,
        blank=True,
        verbose_name="Diamètre",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Créé le")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Modifié le")

    class Meta:
        verbose_name = "Forme personnalisée"
        verbose_name_plural = "Formes personnalisées"
        unique_together = ["user", "name"]
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.get_shape_type_display()})"

    def get_parameters(self):
        if self.shape_type == "rectangle":
            return {"width": self.width, "height": self.height}
        elif self.shape_type == "square":
            return {"size": self.size}
        elif self.shape_type == "circle":
            return {"diameter": self.diameter}
        return {}


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

        # Catégoriser selon la teinte
        if 0 <= hue < 30 or 330 <= hue <= 360:
            return "Rouge"
        elif 30 <= hue < 90:
            return "Orange"
        elif 90 <= hue < 150:
            return "Jaune"
        elif 150 <= hue < 210:
            return "Vert"
        elif 210 <= hue < 270:
            return "Cyan"
        elif 270 <= hue < 330:
            return "Bleu"
        elif 330 <= hue <= 360:
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
