# Generated by Django 5.1.7 on 2025-03-31 23:19

import django.core.validators
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('beadmodels', '0004_beadshape_creator_beadshape_is_shared_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Bead',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, verbose_name='Nom')),
                ('red', models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(255)], verbose_name='Rouge')),
                ('green', models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(255)], verbose_name='Vert')),
                ('blue', models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(255)], verbose_name='Bleu')),
                ('quantity', models.PositiveIntegerField(blank=True, null=True, verbose_name='Quantité')),
                ('notes', models.TextField(blank=True, verbose_name='Notes')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Créé le')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='Modifié le')),
                ('creator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='beads', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Perle',
                'verbose_name_plural': 'Perles',
                'ordering': ['name'],
                'unique_together': {('creator', 'name')},
            },
        ),
    ]
