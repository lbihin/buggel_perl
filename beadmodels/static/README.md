# Organisation des ressources statiques

Ce document décrit l'organisation des fichiers CSS et JavaScript de l'application Buggel.

## Structure CSS

Les styles CSS sont organisés de façon modulaire pour faciliter la maintenance et éviter la duplication :

- **base.css** - Styles fondamentaux pour le corps de la page, typographie et conteneurs
- **components.css** - Styles pour les composants réutilisables (cartes, alertes, boutons, formulaires)
- **layout.css** - Styles pour la mise en page (navbar, footer, grille)
- **utilities.css** - Classes utilitaires simples
- **style.css** - Styles spécifiques à l'application qui ne correspondent pas aux catégories ci-dessus
- **pixelization_wizard.css** - Styles spécifiques à la fonctionnalité de pixelisation
- **user_settings.css** - Styles spécifiques aux pages de paramètres utilisateur
- **bead_selector.css** - Styles pour le sélecteur de perles

## Structure JavaScript

Les fichiers JavaScript sont organisés par fonctionnalité :

- **utils.js** - Fonctions utilitaires partagées entre plusieurs fichiers (par ex. loading overlay)
- **pixelization_wizard.js** - Fonctionnalités pour l'assistant de pixelisation
- **pixelization_result.js** - Gestion de l'affichage des résultats de pixelisation
- **shapes.js** - Gestion des formes de perles
- **user_settings.js** - Fonctionnalités pour les pages de paramètres utilisateur
- **bead_selector.js** - Fonctionnalités pour le sélecteur de perles

## Bonnes pratiques

1. **Éviter la duplication** - Ne pas dupliquer les styles ou fonctions entre les fichiers
2. **Utiliser utils.js** - Placer les fonctions communes dans utils.js
3. **Respect de la séparation des préoccupations** - Garder les styles dans les fichiers CSS appropriés selon leur fonction
4. **Maintenir la documentation** - Mettre à jour ce document lorsque de nouveaux fichiers sont ajoutés ou la structure change