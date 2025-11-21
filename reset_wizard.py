#!/usr/bin/env python
import os

import django

# Configurer l'environnement Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buggel.settings")
django.setup()

from django.contrib.sessions.models import Session


# Supprimer toutes les sessions existantes
Session.objects.all().delete()

print("Sessions nettoyées. Le wizard sera réinitialisé à sa première utilisation.")
