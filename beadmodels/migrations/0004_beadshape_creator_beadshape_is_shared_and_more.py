# Generated by Django 5.1.7 on 2025-03-31 22:25

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('beadmodels', '0003_beadshape_alter_customshape_options_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='beadshape',
            name='creator',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='created_shapes', to=settings.AUTH_USER_MODEL, verbose_name='Créateur'),
        ),
        migrations.AddField(
            model_name='beadshape',
            name='is_shared',
            field=models.BooleanField(default=False, verbose_name='Forme partagée'),
        ),
        migrations.AlterUniqueTogether(
            name='beadshape',
            unique_together={('name', 'creator')},
        ),
    ]
