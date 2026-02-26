from django import forms
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.forms import PasswordChangeForm, UserCreationForm
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


class UserPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["old_password"].widget.attrs.update(
            {"class": "form-control", "id": "old_password"}
        )
        self.fields["new_password1"].widget.attrs.update(
            {"class": "form-control", "id": "new_password1"}
        )
        self.fields["new_password2"].widget.attrs.update(
            {"class": "form-control", "id": "new_password2"}
        )

        # labels pour les champs
        self.fields["old_password"].label = "Mot de passe actuel"
        self.fields["new_password1"].label = "Nouveau mot de passe"
        self.fields["new_password2"].label = "Confirmer le nouveau mot de passe"
