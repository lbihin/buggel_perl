from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("beadmodels", "0004_apppreference"),
    ]

    operations = [
        migrations.AddField(
            model_name="beadmodel",
            name="metadata",
            field=models.JSONField(
                default=dict,
                blank=True,
                verbose_name="Métadonnées",
                help_text="Paramètres de génération (palette, dimensions, etc.)",
            ),
        ),
    ]
