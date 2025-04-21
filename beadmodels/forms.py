from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Bead, BeadBoard, BeadModel, BeadShape, CustomShape


class BeadModelForm(forms.ModelForm):
    class Meta:
        model = BeadModel
        fields = ["name", "description", "original_image", "is_public"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 4}),
        }


class TransformModelForm(forms.Form):
    board = forms.ModelChoiceField(
        queryset=BeadBoard.objects.all(),
        label="Support de perles",
        widget=forms.Select(attrs={"class": "form-select"}),
        help_text="Choisissez le support de perles à utiliser",
    )
    color_reduction = forms.IntegerField(
        label="Nombre de couleurs",
        initial=16,
        min_value=2,
        max_value=256,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
        help_text="Nombre de couleurs à utiliser (entre 2 et 256)",
    )
    edge_detection = forms.BooleanField(
        label="Détection des contours",
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Activer la détection des contours pour faciliter la reproduction",
    )


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name")


class BeadForm(forms.ModelForm):
    class Meta:
        model = Bead
        fields = ["name", "red", "green", "blue", "quantity", "notes"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "red": forms.NumberInput(
                attrs={"class": "form-control", "min": 0, "max": 255}
            ),
            "green": forms.NumberInput(
                attrs={"class": "form-control", "min": 0, "max": 255}
            ),
            "blue": forms.NumberInput(
                attrs={"class": "form-control", "min": 0, "max": 255}
            ),
            "quantity": forms.NumberInput(attrs={"class": "form-control", "min": 0}),
            "notes": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
        }

    def clean(self):
        cleaned_data = super().clean()

        # Vérifier les valeurs RGB sont dans la plage 0-255
        for color in ["red", "green", "blue"]:
            value = cleaned_data.get(color)
            if value is not None:
                if value < 0 or value > 255:
                    self.add_error(color, f"La valeur doit être entre 0 et 255.")

        return cleaned_data


class BeadShapeForm(forms.ModelForm):
    class Meta:
        model = BeadShape
        fields = ["name", "shape_type"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "shape_type": forms.Select(attrs={"class": "form-control"}),
        }

    # Champs dynamiques qui seront ajoutés en fonction du type de forme
    width = forms.IntegerField(
        label="Largeur",
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    height = forms.IntegerField(
        label="Hauteur",
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    size = forms.IntegerField(
        label="Taille",
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    diameter = forms.IntegerField(
        label="Diamètre",
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instance = kwargs.get("instance")
        if instance and instance.pk:
            # Pré-remplir les champs spécifiques en fonction du type de forme
            if instance.shape_type == "rectangle":
                self.fields["width"].initial = instance.width
                self.fields["height"].initial = instance.height
            elif instance.shape_type == "square":
                self.fields["size"].initial = instance.size
            elif instance.shape_type == "circle":
                self.fields["diameter"].initial = instance.diameter

    def clean(self):
        cleaned_data = super().clean()
        shape_type = cleaned_data.get("shape_type")

        # Valider les champs en fonction du type de forme
        if shape_type == "rectangle":
            if not cleaned_data.get("width"):
                self.add_error("width", "La largeur est requise pour un rectangle.")
            if not cleaned_data.get("height"):
                self.add_error("height", "La hauteur est requise pour un rectangle.")
        elif shape_type == "square":
            if not cleaned_data.get("size"):
                self.add_error("size", "La taille est requise pour un carré.")
        elif shape_type == "circle":
            if not cleaned_data.get("diameter"):
                self.add_error("diameter", "Le diamètre est requis pour un cercle.")

        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        shape_type = self.cleaned_data.get("shape_type")

        # Enregistrer les champs spécifiques en fonction du type de forme
        if shape_type == "rectangle":
            instance.width = self.cleaned_data.get("width")
            instance.height = self.cleaned_data.get("height")
        elif shape_type == "square":
            instance.size = self.cleaned_data.get("size")
        elif shape_type == "circle":
            instance.diameter = self.cleaned_data.get("diameter")

        if commit:
            instance.save()
        return instance


class UserPreferencesForm(forms.Form):
    default_grid_size = forms.IntegerField(
        label="Taille de grille par défaut",
        min_value=5,
        max_value=100,
        initial=29,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    public_by_default = forms.BooleanField(
        label="Rendre mes modèles publics par défaut",
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )
