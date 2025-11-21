from django.core.management.base import BaseCommand

from beadmodels.services.image_processing import cleanup_temp_images


class Command(BaseCommand):
    help = "Supprime les images temporaires du wizard plus anciennes que l'age spécifié (défaut 1h)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--max-age",
            type=int,
            default=3600,
            help="Age maximum en secondes (défaut 3600).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Lister les fichiers candidats sans les supprimer.",
        )

    def handle(self, *args, **options):
        max_age = options["max_age"]
        dry_run = options["dry_run"]

        if dry_run:
            # Exécuter la fonction pour obtenir la liste sans suppression: on fait une stratégie alternative
            # Ici on ne peut pas obtenir sans suppression car cleanup supprime directement.
            # On informera l'utilisateur d'utiliser --max-age avec une petite valeur si besoin.
            self.stdout.write(
                self.style.WARNING(
                    "Mode dry-run non supporté finement: les fichiers ne seront pas supprimés car nous ne les listons pas séparément."
                )
            )
            self.stdout.write(
                "Pour un aperçu, inspectez le dossier 'media/temp_wizard'."
            )
            return

        deleted = cleanup_temp_images(max_age_seconds=max_age)
        if deleted:
            self.stdout.write(
                self.style.SUCCESS(
                    f"{len(deleted)} fichier(s) supprimé(s):\n" + "\n".join(deleted)
                )
            )
        else:
            self.stdout.write(
                self.style.NOTICE(
                    "Aucun fichier temporaire à supprimer (ou dossier absent)."
                )
            )
