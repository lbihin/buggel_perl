from django.core.management.base import BaseCommand

from beadmodels.models import BeadShape


class Command(BaseCommand):
    help = "Crée les formes de perles par défaut"

    def handle(self, *args, **options):
        default_shapes = [
            {
                "name": "Petit carré",
                "shape_type": "square",
                "size": 10,
                "is_default": True,
            },
            {
                "name": "Grand carré",
                "shape_type": "square",
                "size": 20,
                "is_default": True,
            },
            {
                "name": "Petit rectangle",
                "shape_type": "rectangle",
                "width": 15,
                "height": 10,
                "is_default": True,
            },
            {
                "name": "Grand rectangle",
                "shape_type": "rectangle",
                "width": 30,
                "height": 20,
                "is_default": True,
            },
            {
                "name": "Petit rond",
                "shape_type": "circle",
                "diameter": 10,
                "is_default": True,
            },
            {
                "name": "Grand rond",
                "shape_type": "circle",
                "diameter": 20,
                "is_default": True,
            },
        ]

        for shape_data in default_shapes:
            shape, created = BeadShape.objects.get_or_create(
                name=shape_data["name"], defaults=shape_data
            )
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Forme "{shape.name}" créée avec succès')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'La forme "{shape.name}" existe déjà')
                )
