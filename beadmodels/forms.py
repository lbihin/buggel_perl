from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import BeadModel


class BeadModelForm(forms.ModelForm):
    class Meta:
        model = BeadModel
        fields = ['title', 'description', 'original_image', 'grid_size', 'is_public']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'grid_size': forms.NumberInput(attrs={'min': 16, 'max': 64}),
        }


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2') 