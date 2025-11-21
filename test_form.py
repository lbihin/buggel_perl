#!/usr/bin/env python
"""
Script de test pour vérifier le formulaire de sauvegarde du modèle.
"""
import os

import django

# Configurer l'environnement Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buggel.settings")
django.setup()

from django import forms

from beadmodels.forms import BeadModelForm
from beadmodels.models import BeadBoard


# Créer une classe de formulaire spécifique pour le test
class WizardBeadModelForm(BeadModelForm):
    """Version du formulaire BeadModelForm pour le test."""

    class Meta(BeadModelForm.Meta):
        fields = ["name", "description", "board", "is_public"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "board": forms.Select(attrs={"class": "form-select"}),
            "is_public": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }


# Initialiser le formulaire avec des valeurs par défaut
initial_data = {"name": "Test du modèle", "is_public": False}

print("Création du formulaire de test...")
form = WizardBeadModelForm(initial=initial_data)

# Vérifier que le formulaire est bien initialisé
print(f"Formulaire initialisé: {form is not None}")
print(f"Champs du formulaire: {list(form.fields.keys())}")

# Vérifier les tableaux disponibles
boards = BeadBoard.objects.all()
print(f"Tableaux disponibles: {boards.count()}")
for board in boards:
    print(f" - {board.name}")

print("\nFormulaire HTML:")
print(form.as_p())
