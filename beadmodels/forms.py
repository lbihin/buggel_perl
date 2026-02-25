from django import forms

from .models import AppPreference, BeadBoard, BeadModel


# Formulaires pour le nouveau wizard de création de modèle
class ImageUploadForm(forms.Form):
    """Formulaire pour l'étape 1: Chargement de l'image."""

    image = forms.ImageField(
        label="Image à transformer",
        help_text="Téléchargez une image pour créer un modèle de perles à repasser",
        widget=forms.FileInput(attrs={"class": "form-control", "accept": "image/*"}),
    )


class ModelConfigurationForm(forms.Form):
    """Formulaire pour configurer un modèle de perles."""

    color_reduction = forms.IntegerField(
        label="Nombre de couleurs",
        min_value=2,
        max_value=64,
        initial=16,
        required=False,  # Sera géré via les boutons radio
        widget=forms.HiddenInput(),  # Caché car géré par les boutons
        help_text="Nombre de couleurs à utiliser dans le modèle final",
    )
    use_available_colors = forms.BooleanField(
        label="Utiliser mes couleurs disponibles",
        required=False,
        initial=False,
        widget=forms.CheckboxInput(
            attrs={
                "class": "form-check-input",
                "hx-post": "",
                "hx-trigger": "change",
                "hx-target": "#preview-container",
            }
        ),
    )


class BeadModelFinalizeForm(forms.ModelForm):
    """Formulaire de finalisation du modèle (étape 3)."""

    tags = forms.CharField(
        required=False,
        label="Tags",
        help_text="Séparez les tags par des virgules",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "ex: animaux, mario, cadeau",
            }
        ),
    )

    class Meta:
        model = BeadModel
        fields = ["name", "description", "board", "is_public"]
        widgets = {
            "name": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Nom du modèle",
                    "hx-trigger": "keyup changed delay:300ms",
                    "hx-target": "#model-preview-title",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 3,
                    "placeholder": "Description (optionnelle)",
                }
            ),
            "board": forms.Select(
                attrs={
                    "class": "form-select",
                    "hx-trigger": "change",
                    "hx-target": "#board-preview",
                }
            ),
            "is_public": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }
        labels = {
            "name": "Nom du modèle",
            "description": "Description",
            "board": "Support",
            "is_public": "Rendre public",
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        # Pré-remplir is_public selon les préférences utilisateur si disponible
        if user and not self.instance.pk:
            from accounts.models import UserSettings

            settings = UserSettings.objects.for_user(user=user)
            self.fields["is_public"].initial = settings.set_public


class BeadModelForm(forms.ModelForm):
    class Meta:
        model = BeadModel
        fields = ["name", "description", "original_image", "is_public"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 4}),
        }

    def __init__(self, *args, **kwargs):
        # Extraire l'utilisateur des kwargs s'il est fourni
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)

        # Si c'est un nouveau modèle (pas d'instance) et qu'on a un utilisateur
        if not self.instance.pk and user:
            from accounts.models import UserSettings

            # Récupérer ou créer les paramètres utilisateur
            user_settings = UserSettings.objects.for_user(user=user)
            # Définir la valeur par défaut selon les préférences utilisateur
            self.fields["is_public"].initial = user_settings.set_public


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


class AppPreferenceForm(forms.ModelForm):
    """Formulaire pour les préférences de l'application."""

    class Meta:
        model = AppPreference
        fields = ["bead_low_quantity_threshold"]
        widgets = {
            "bead_low_quantity_threshold": forms.NumberInput(
                attrs={"class": "form-control", "min": "1", "max": "1000"}
            ),
        }
