from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from django.db import models


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
        return f"{self.name} ({self.base_shape.get_dimensions_display()})"

    def get_parameters(self):
        if self.shape_type == "rectangle":
            return {"width": self.width, "height": self.height}
        elif self.shape_type == "square":
            return {"size": self.size}
        elif self.shape_type == "circle":
            return {"diameter": self.diameter}
        return {}
