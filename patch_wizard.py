#!/usr/bin/env python
"""
Script pour mettre à jour la classe BaseWizard et résoudre le problème de rendu des templates.
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


# Patch pour la classe WizardStep
def patch_wizard_step():
    # Sauvegarde de l'ancien fichier wizards.py
    wizards_path = BASE_DIR / "beadmodels" / "wizards.py"
    backup_path = BASE_DIR / "beadmodels" / "wizards.py.bak"

    # Créer une sauvegarde si elle n'existe pas déjà
    if not backup_path.exists():
        with open(wizards_path, "r") as f:
            content = f.read()

        with open(backup_path, "w") as f:
            f.write(content)

        print(f"Sauvegarde créée: {backup_path}")

    # Lire le contenu actuel
    with open(wizards_path, "r") as f:
        lines = f.readlines()

    # Identifier et modifier la méthode render_template
    found_method = False
    method_start = None
    method_end = None
    indent = ""

    for i, line in enumerate(lines):
        if "def render_template" in line:
            found_method = True
            method_start = i
            indent = line[: line.find("def")]
        elif found_method and line.strip() and not line.startswith(indent + " "):
            method_end = i
            break

    # Si nous avons trouvé la méthode, la remplacer
    if found_method and method_start is not None:
        if method_end is None:
            method_end = len(lines)

        # Nouvelle implémentation
        new_method = [
            f"{indent}def render_template(self, context=None):\n",
            f'{indent}    """Rend le template avec le contexte fourni."""\n',
            f"{indent}    if context is None:\n",
            f"{indent}        context = {{}}\n",
            f"\n",
            f"{indent}    # Debug: tracer les templates étendus\n",
            f'{indent}    print(f"Rendering template: {{self.template}} with context keys: {{list(context.keys())}}")\n',
            f"\n",
            f"{indent}    full_context = self.get_context_data(**context)\n",
            f"{indent}    # Force l'ajout des variables nécessaires pour le template base.html\n",
            f"{indent}    full_context['debug_info'] = {{\n",
            f"{indent}        'template': self.template,\n",
            f"{indent}        'extends': 'base.html',\n",
            f"{indent}        'wizard_name': self.wizard.name,\n",
            f"{indent}    }}\n",
            f"{indent}    return render(self.wizard.request, self.template, full_context)\n",
        ]

        # Remplacer l'ancienne méthode par la nouvelle
        new_lines = lines[:method_start] + new_method + lines[method_end:]

        # Écrire le contenu mis à jour
        with open(wizards_path, "w") as f:
            f.writelines(new_lines)

        print(f"Méthode render_template mise à jour dans {wizards_path}")
    else:
        print("Méthode render_template non trouvée.")


if __name__ == "__main__":
    patch_wizard_step()
    print("Redémarrez le serveur Django pour appliquer les modifications.")
