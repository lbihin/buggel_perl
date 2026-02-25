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
    is_default = models.BooleanField(default=False, verbose_name="Forme par défaut")
    is_shared = models.BooleanField(default=False, verbose_name="Forme partagée")

    class Meta:
        verbose_name = "Forme de perles"
        verbose_name_plural = "Formes de perles"
        ordering = ["name"]
        unique_together = ["name", "creator"]
        indexes = [
            models.Index(fields=["creator"]),
            models.Index(fields=["shape_type"]),
            models.Index(fields=["is_default"]),
            models.Index(fields=["is_shared"]),
        ]

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

    def determine_shape_type(self):
        """Détermine automatiquement le type de forme basé sur les dimensions définies"""
        if self.diameter and not self.width and not self.height and not self.size:
            return "circle"
        elif self.size and not self.width and not self.height and not self.diameter:
            return "square"
        elif self.width and self.height and not self.size and not self.diameter:
            if self.width == self.height:
                return "square"  # C'est en fait un carré si largeur = hauteur
            else:
                return "rectangle"
        # Si les dimensions ne correspondent à aucun type connu, on garde le type actuel
        return self.shape_type

    def update_from_dimensions(self):
        """Met à jour le type en fonction des dimensions actuelles et normalise les données"""
        new_type = self.determine_shape_type()

        # Si le type a changé, mettre à jour et normaliser les données
        if new_type != self.shape_type:
            if (
                new_type == "square"
                and self.width
                and self.height
                and self.width == self.height
            ):
                # Convertir rectangle carré en carré
                self.size = self.width
                self.width = None
                self.height = None
                self.diameter = None
            elif new_type == "circle":
                # S'assurer que seul le diamètre est défini
                self.width = None
                self.height = None
                self.size = None

            self.shape_type = new_type
