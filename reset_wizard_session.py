#!/usr/bin/env python
"""
Script pour réinitialiser la session du wizard dans la base de données.
Ce script permet de forcer le reset des données de session qui pourraient être en cache.
"""
import os

import django

# Configurer l'environnement Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buggel.settings")
django.setup()


from django.contrib.sessions.models import Session

# Chercher toutes les sessions actives
active_sessions = Session.objects.all()
print(f"Sessions totales trouvées: {active_sessions.count()}")

wizard_keys = ["model_creation_wizard", "pixelization_wizard"]
modified_sessions = 0

# Parcourir chaque session
for session in active_sessions:
    # Décoder les données de session
    session_data = session.get_decoded()
    modified = False

    # Vérifier et supprimer les clés liées au wizard
    for key in list(session_data.keys()):
        for wizard_key in wizard_keys:
            if wizard_key in key:
                print(
                    f"Suppression de la clé {key} dans la session {session.session_key}"
                )
                del session_data[key]
                modified = True

    # Si des modifications ont été faites, sauvegarder
    if modified:
        session.session_data = Session.objects.encode(session_data)
        session.save()
        modified_sessions += 1

print(f"Sessions modifiées: {modified_sessions}")
