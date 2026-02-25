from .models import AppPreference


class AppPreferencesMiddleware:
    """Middleware qui charge les préférences de l'application et les met à disposition dans la requête."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            preferences = AppPreference.get_instance()
            request.app_preferences = preferences
        except Exception:
            request.app_preferences = None

        response = self.get_response(request)
        return response
