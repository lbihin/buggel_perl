# Wizard Création de Modèle – Étape Finalisation

## Changements Principaux
- Lien "Créer un modèle" pointe désormais directement vers `model_creation_wizard` (étape 1).
- Nouvelle vue de finalisation : template `beadmodels/wizard/finalize_step.html`.
- Formulaire dédié `BeadModelFinalizeForm` (nom, description, board, visibilité, tags).
- Métadonnées enrichies à l'enregistrement: `initial_color_reduction`, `final_color_reduction`, `excluded_colors`.
- Palette affichée avec correspondances aux perles utilisateur (top 3 plus proches par couleur).
- Support exclusion couleur (placeholder HTMX; stockage dans `excluded_colors`).

## Fichier Modifiés
- `beadmodels/model_creation_wizard.py` : refactor de `SaveStep`.
- `beadmodels/forms.py` : ajout `BeadModelFinalizeForm`.
- `templates/base.html` : mise à jour du lien navbar.
- `beadmodels/tests.py` : ajout tests `TestFinalizeStep`.

## Métadonnées Exemple
```json
{
  "grid_width": 29,
  "grid_height": 29,
  "shape_id": null,
  "initial_color_reduction": 16,
  "final_color_reduction": 16,
  "total_beads": 841,
  "palette": [...],
  "excluded_colors": []
}
```

## Pistes Futures
- Recalcul réel de la palette après exclusion/merge (reclusterisation).
- Ajustement dynamique du `color_reduction` depuis la finalisation.
- Export PDF avec instructions (intégrer WeasyPrint).
- Stats d'utilisation des couleurs (pour stock planning).
