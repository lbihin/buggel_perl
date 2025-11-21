#!/usr/bin/env python
"""
Script pour réinitialiser la session du wizard et vider le cache Django.
"""

import os
import sys
from pathlib import Path

import django

# Configuration Django
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buggel.settings")
django.setup()

from django.core.cache import cache


def reset_all():
    """Réinitialise toutes les sessions de wizard et vide le cache Django."""
    # Chemin vers le fichier de base de données SQLite
    db_path = BASE_DIR / "db.sqlite3"

    # Supprimer les cookies de session (si existants)
    session_files = list(BASE_DIR.glob("*.session"))
    for session_file in session_files:
        try:
            os.remove(session_file)
            print(f"Session supprimée: {session_file}")
        except Exception as e:
            print(f"Erreur lors de la suppression de {session_file}: {e}")

    # Vider le cache Django
    cache.clear()
    print("Cache Django vidé")

    # Si la base de données est SQLite, exécuter VACUUM pour optimiser
    if db_path.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            conn.execute("VACUUM;")
            conn.close()
            print(f"Base de données optimisée: {db_path}")
        except Exception as e:
            print(f"Erreur lors de l'optimisation de la base de données: {e}")

    # Message final
    print("\nSession réinitialisée avec succès.")
    print("Redémarrez le serveur Django et videz le cache du navigateur.")


if __name__ == "__main__":
    reset_all()
