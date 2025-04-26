import os
import sys

import django
import pytest
from django.conf import settings

# Ajouter le répertoire du projet au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configurer les paramètres Django pour pytest
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Configuration pour pytest et Django"""
    # Configuration de Django
    settings.DEBUG = False
    # Désactiver les signaux Django pour les tests parallèles mais conserver les middleware essentiels
    settings.USE_TZ = False

    # Au lieu de supprimer les middleware, préserver l'ordre correct pour l'authentification
    if "django.contrib.auth.middleware.AuthenticationMiddleware" in settings.MIDDLEWARE:
        # Assurons-nous que le SessionMiddleware est présent pour AuthenticationMiddleware
        if (
            "django.contrib.sessions.middleware.SessionMiddleware"
            not in settings.MIDDLEWARE
        ):
            # Trouver l'index de AuthenticationMiddleware
            auth_index = settings.MIDDLEWARE.index(
                "django.contrib.auth.middleware.AuthenticationMiddleware"
            )
            # Insérer SessionMiddleware avant AuthenticationMiddleware
            settings.MIDDLEWARE.insert(
                auth_index, "django.contrib.sessions.middleware.SessionMiddleware"
            )

    # Utiliser un backend de cache en mémoire pour les tests
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }
    django.setup()

    # Configuration pour xdist (tests parallèles)
    if hasattr(config, "workerinput"):
        # Ce code s'exécute uniquement pour les workers xdist
        worker_id = config.workerinput.get("workerid", "gw0")
        settings.DATABASES["default"]["NAME"] = f"test_{worker_id}"


# Cette fonction sera exécutée avant chaque test
@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    """
    Ce fixture permet l'accès à la base de données pour tous les tests
    sans avoir besoin de décorer chaque test avec @pytest.mark.django_db
    """
    pass


# Ajoutez d'autres fixtures ici qui seront disponibles dans tous vos tests
