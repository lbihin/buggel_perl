#!/usr/bin/env python
"""
Script de diagnostic pour les problèmes de template Django.
Exécutez ce script pour vérifier les templates et générer un rapport.
"""

from pathlib import Path


def check_template_paths():
    """Vérifie les chemins de templates standard."""
    base_dir = Path(__file__).resolve().parent
    template_paths = [
        base_dir / "templates",
        base_dir / "beadmodels" / "templates",
        base_dir / "accounts" / "templates",
        base_dir / "shapes" / "templates",
    ]

    results = {}
    for path in template_paths:
        exists = path.exists()
        results[str(path)] = {
            "exists": exists,
            "is_dir": path.is_dir() if exists else False,
        }

    # Vérifie des templates spécifiques
    specific_templates = [
        base_dir / "templates" / "base.html",
        base_dir
        / "beadmodels"
        / "templates"
        / "beadmodels"
        / "model_creation"
        / "upload_image.html",
    ]

    for template_path in specific_templates:
        results[str(template_path)] = {
            "exists": template_path.exists(),
            "is_file": template_path.is_file() if template_path.exists() else False,
        }

    return results


def check_static_files():
    """Vérifie les fichiers statiques principaux."""
    base_dir = Path(__file__).resolve().parent
    static_paths = [
        base_dir / "static",
        base_dir / "beadmodels" / "static",
        base_dir / "static" / "css" / "style.css",
    ]

    results = {}
    for path in static_paths:
        exists = path.exists()
        results[str(path)] = {
            "exists": exists,
            "type": (
                "directory"
                if path.is_dir()
                else "file" if path.is_file() else "unknown"
            ),
        }

    return results


def print_report():
    """Imprime un rapport de diagnostic."""
    print("=" * 80)
    print("RAPPORT DE DIAGNOSTIC DES TEMPLATES DJANGO")
    print("=" * 80)

    print("\n1. VÉRIFICATION DES CHEMINS DE TEMPLATES\n" + "-" * 50)
    template_results = check_template_paths()
    for path, result in template_results.items():
        status = "TROUVÉ" if result["exists"] else "NON TROUVÉ"
        print(f"{path}: {status}")
        if result["exists"]:
            type_info = (
                "DOSSIER"
                if result.get("is_dir", False)
                else "FICHIER" if result.get("is_file", False) else "TYPE INCONNU"
            )
            print(f"  Type: {type_info}")

    print("\n2. VÉRIFICATION DES FICHIERS STATIQUES\n" + "-" * 50)
    static_results = check_static_files()
    for path, result in static_results.items():
        status = "TROUVÉ" if result["exists"] else "NON TROUVÉ"
        print(f"{path}: {status}")
        if result["exists"]:
            print(f"  Type: {result['type']}")

    print("\n3. SOLUTION PROPOSÉE\n" + "-" * 50)
    print(
        "1. Vérifiez que la vue du wizard utilise correctement render() avec le template"
    )
    print("2. Assurez-vous que le template étend correctement 'base.html'")
    print("3. Vérifiez que 'AppPreferencesMiddleware' n'interfère pas avec le rendu")
    print("4. Essayez de nettoyer la session avec la commande:")
    print("   python reset_wizard.py")
    print(
        "5. Vérifiez le formatage HTML des templates, cherchez les balises non fermées"
    )
    print("6. Redémarrez le serveur de développement Django avec:")
    print("   python manage.py runserver")


if __name__ == "__main__":
    print_report()
    print("\nFin du diagnostic. Consultez les résultats ci-dessus.")
