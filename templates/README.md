# Templates

Tous les templates Django du projet sont centralisés ici.
**Aucun template ne doit se trouver dans les dossiers `<app>/templates/`.**

## Structure

```
templates/
├── base.html                   # Layout principal (navbar, head, scripts)
├── home.html                   # Page d'accueil
├── accounts/                   # Authentification & profil
├── beadmodels/                 # Modèles de perles
│   ├── partials/               # Fragments HTMX (inline edit, delete, cards)
│   └── wizard/                 # Étapes du wizard de création
│       └── partials/           # Fragments HTMX du wizard (preview)
├── beads/                      # Collection de perles
│   └── partials/               # Fragments HTMX (lignes, édition inline)
└── shapes/                     # Plaques à picots
    └── partials/               # Fragments HTMX (lignes, édition inline)
```

## Conventions

- **`partials/`** : contient les fragments HTML utilisés par HTMX (swap `outerHTML`, `innerHTML`, etc.).
  Ces fichiers ne sont jamais rendus en tant que pages complètes.
- Les pages complètes héritent de `base.html` via `{% extends 'base.html' %}`.
- Les chemins dans le code Python utilisent le préfixe de l'app : `"beads/partials/bead_row_display.html"`.
