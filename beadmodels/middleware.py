from django.conf import settings

from .models import AppPreference


class AppPreferencesMiddleware:
    """Middleware qui charge les préférences de l'application et les met à disposition dans la requête."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Récupérer les préférences de l'application
        try:
            preferences = AppPreference.get_instance()
            # Mettre à jour le paramètre dans les settings pour une utilisation directe
            settings.BEAD_LOW_QUANTITY_THRESHOLD = (
                preferences.bead_low_quantity_threshold
            )
            # Ajouter les préférences à la requête pour accès dans les templates
            request.app_preferences = preferences
        except Exception:
            # En cas d'erreur, utiliser les valeurs par défaut
            settings.BEAD_LOW_QUANTITY_THRESHOLD = 20
            request.app_preferences = None

        # Continuer avec la requête
        response = self.get_response(request)
        return response
