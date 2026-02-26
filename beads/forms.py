from django import forms
from django.utils.translation import gettext_lazy as _

from .models import Bead


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
                    self.add_error(color, _("La valeur doit être entre 0 et 255."))

        return cleaned_data
