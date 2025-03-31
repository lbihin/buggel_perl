# Buggel - Créateur de Modèles de Perles à Repasser

Buggel est une application web Django qui permet de créer des modèles de perles à repasser à partir d'images. L'application transforme vos images en grilles de perles à repasser, facilitant ainsi la création de motifs pixelisés.

## Fonctionnalités

- Création de modèles à partir d'images
- Choix de la taille de la grille
- Gestion des modèles publics et privés
- Interface utilisateur intuitive
- Système d'authentification

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-username/buggel.git
cd buggel
```

2. Installez les dépendances avec Poetry :
```bash
poetry install
```

3. Appliquez les migrations :
```bash
poetry run python manage.py migrate
```

4. Créez un superutilisateur :
```bash
poetry run python manage.py createsuperuser
```

5. Lancez le serveur de développement :
```bash
poetry run python manage.py runserver
```

## Technologies utilisées

- Python 3.11+
- Django 5.0+
- Pillow (traitement d'images)
- Bootstrap 5
- Crispy Forms

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails. 