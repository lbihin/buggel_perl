from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import BeadBoard, BeadModel


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
