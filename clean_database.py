import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buggel.settings")
django.setup()

from django.db.models import Q

from beadmodels.models import BeadModel

# Supprimer les modèles qui n'ont pas d'image associée
# (probablement des résidus de l'ancienne implémentation)
orphaned_models = BeadModel.objects.filter(
    Q(original_image="")
    | Q(original_image=None)
    | Q(bead_pattern="")
    | Q(bead_pattern=None)
)

count = orphaned_models.count()
if count > 0:
    print(f"Suppression de {count} modèles orphelins...")
    orphaned_models.delete()
    print("Suppression terminée.")
else:
    print("Aucun modèle orphelin trouvé.")

print("\nNettoyage terminé.")
