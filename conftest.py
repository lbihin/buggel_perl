import os
import shutil
import sys
import tempfile

import django
import pytest
from django.conf import settings
from django.test.utils import override_settings

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


# On retire le fixture django_db_setup personnalisé car pytest-django le gère déjà
# et setup_databases n'est pas disponible dans votre version de Django


@pytest.fixture(scope="class")
def temp_media_root(request):
    """
    Fixture qui crée un dossier temporaire pour MEDIA_ROOT
    et le supprime après les tests.

    Cette fixture peut être utilisée au niveau classe:

    @pytest.mark.usefixtures('temp_media_root')
    class TestMyMedia:
        def test_something(self):
            ...
    """
    # Créer un dossier temporaire
    temp_dir = tempfile.mkdtemp()

    # Créer les sous-dossiers nécessaires
    os.makedirs(os.path.join(temp_dir, "originals"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "patterns"), exist_ok=True)

    # Utiliser override_settings pour remplacer temporairement MEDIA_ROOT
    media_root_override = override_settings(MEDIA_ROOT=temp_dir)
    media_root_override.__enter__()

    # Ajouter un callback pour nettoyer à la fin des tests
    def cleanup():
        media_root_override.__exit__(None, None, None)
        shutil.rmtree(temp_dir, ignore_errors=True)

    request.addfinalizer(cleanup)

    return temp_dir


@pytest.fixture(scope="session", autouse=True)
def session_temp_media_root(request):
    """
    Version 'session' du fixture temp_media_root.
    Ce fixture est automatiquement utilisé pour tous les tests de la session.
    """
    # Créer un dossier temporaire pour toute la session de test
    temp_dir = tempfile.mkdtemp()

    # Créer les sous-dossiers nécessaires
    os.makedirs(os.path.join(temp_dir, "originals"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "patterns"), exist_ok=True)

    # Sauvegarder l'ancien MEDIA_ROOT
    old_media_root = settings.MEDIA_ROOT

    # Remplacer MEDIA_ROOT par notre dossier temporaire
    settings.MEDIA_ROOT = temp_dir

    # Nettoyer à la fin de la session
    def cleanup():
        settings.MEDIA_ROOT = old_media_root
        shutil.rmtree(temp_dir, ignore_errors=True)

    request.addfinalizer(cleanup)

    return temp_dir


# Activer l'accès à la DB pour tous les tests par défaut
@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    """
    Ce fixture permet l'accès à la base de données pour tous les tests
    sans avoir besoin de décorer chaque test avec @pytest.mark.django_db
    """
    pass
