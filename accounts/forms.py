from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from accounts.models import UserSettings


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name")


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


class UserSettingsForm(forms.ModelForm):
    class Meta:
        model = UserSettings
        fields = ("set_public",)
        widgets = {
            "set_public": forms.CheckboxInput(
                attrs={"class": "form-check-input", "id": "public_by_default"}
            ),
        }
        labels = {"set_public": "Rendre les modèles publics par défaut"}
