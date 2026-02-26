from django import forms
from django.utils.translation import gettext_lazy as _

from shapes.models import BeadShape


class BeadShapeForm(forms.ModelForm):
    class Meta:
        model = BeadShape
        fields = ["name", "shape_type", "is_shared"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "shape_type": forms.Select(attrs={"class": "form-control"}),
        }

    # Champs dynamiques ajoutés en fonction du type de forme
    width = forms.IntegerField(
        label=_("Largeur"),
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    height = forms.IntegerField(
        label=_("Hauteur"),
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    size = forms.IntegerField(
        label=_("Taille"),
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    diameter = forms.IntegerField(
        label=_("Diamètre"),
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instance = kwargs.get("instance")
        if instance and instance.pk:
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

        if shape_type == "rectangle":
            if not cleaned_data.get("width"):
                self.add_error("width", _("La largeur est requise pour un rectangle."))
            if not cleaned_data.get("height"):
                self.add_error("height", _("La hauteur est requise pour un rectangle."))
        elif shape_type == "square":
            if not cleaned_data.get("size"):
                self.add_error("size", _("La taille est requise pour un carré."))
        elif shape_type == "circle":
            if not cleaned_data.get("diameter"):
                self.add_error("diameter", _("Le diamètre est requis pour un cercle."))

        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        shape_type = self.cleaned_data.get("shape_type")

        # Enregistrer les dimensions et nettoyer les champs non pertinents
        if shape_type == "rectangle":
            instance.width = self.cleaned_data.get("width")
            instance.height = self.cleaned_data.get("height")
            instance.size = None
            instance.diameter = None
        elif shape_type == "square":
            instance.size = self.cleaned_data.get("size")
            instance.width = None
            instance.height = None
            instance.diameter = None
        elif shape_type == "circle":
            instance.diameter = self.cleaned_data.get("diameter")
            instance.width = None
            instance.height = None
            instance.size = None

        if commit:
            instance.save()
        return instance
