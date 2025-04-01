from django.db import migrations


def create_default_boards(apps, schema_editor):
    BeadBoard = apps.get_model("beadmodels", "BeadBoard")

    default_boards = [
        {
            "name": "Petit carré",
            "width_pegs": 29,
            "height_pegs": 29,
            "description": "Support carré standard pour petits projets",
        },
        {
            "name": "Grand carré",
            "width_pegs": 58,
            "height_pegs": 58,
            "description": "Support carré pour projets moyens",
        },
        {
            "name": "Rectangle standard",
            "width_pegs": 58,
            "height_pegs": 29,
            "description": "Support rectangulaire pour projets panoramiques",
        },
        {
            "name": "Grand rectangle",
            "width_pegs": 87,
            "height_pegs": 58,
            "description": "Support rectangulaire pour grands projets",
        },
    ]

    for board_data in default_boards:
        BeadBoard.objects.create(**board_data)


def remove_default_boards(apps, schema_editor):
    BeadBoard = apps.get_model("beadmodels", "BeadBoard")
    BeadBoard.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ("beadmodels", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_default_boards, remove_default_boards),
    ]
