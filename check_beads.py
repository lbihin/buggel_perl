import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buggel.settings")
django.setup()

from django.contrib.auth.models import User

from beadmodels.models import Bead

# Vérifier si nous avons des utilisateurs
users = User.objects.all()
print(f"Nombre d'utilisateurs: {users.count()}")

# Vérifier si nous avons des perles
beads = Bead.objects.all()
print(f"Nombre de perles: {beads.count()}")

if beads.exists():
    print("\nExemples de perles existantes:")
    for bead in beads[:5]:  # Affiche jusqu'à 5 perles
        print(f"- {bead.name} (RGB: {bead.red}, {bead.green}, {bead.blue})")
else:
    print("\nAucune perle n'existe actuellement dans la base de données.")

    # Si un utilisateur existe, créons une perle de test
    if users.exists():
        user = users.first()
        print(f"\nCréation d'une perle de test pour l'utilisateur {user.username}...")

        try:
            bead = Bead.objects.create(
                creator=user,
                name="Rouge Test",
                red=255,
                green=0,
                blue=0,
                quantity=100,
                notes="Perle de test créée par script",
            )
            print(f"Perle créée avec succès: {bead.name}")
        except Exception as e:
            print(f"Erreur lors de la création de la perle: {e}")
