from django import forms

from shapes.models import BeadShape


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


SHAPE_CHOICES = [
    ("rectangle", "Rectangle"),
    ("square", "Carré"),
    ("circle", "Rond"),
]


class ShapeForm(forms.Form):
    shape_type = forms.ChoiceField(
        choices=SHAPE_CHOICES,
        label="Type de forme",
        widget=forms.Select(attrs={"class": "form-select", "id": "shapeType"}),
    )
    width = forms.IntegerField(
        required=False,
        label="Largeur (pics)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 1}),
    )
    height = forms.IntegerField(
        required=False,
        label="Hauteur (pics)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 1}),
    )
    size = forms.IntegerField(
        required=False,
        label="Taille (pics)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 1}),
    )
    diameter = forms.IntegerField(
        required=False,
        label="Diamètre (pics)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 1}),
    )
