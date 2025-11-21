import os
import time

import pytest
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from beadmodels.services.image_processing import cleanup_temp_images, save_temp_image


@pytest.mark.django_db
def test_cleanup_temp_images_deletes_old_file(tmp_path):
    # Créer un fichier temporaire via save_temp_image
    cf = ContentFile(b"testdata", name="old.png")
    path = save_temp_image(cf)
    abs_path = default_storage.path(path)
    # Modifier son mtime pour simuler ancien fichier (> 2s)
    two_seconds_ago = time.time() - 5
    os.utime(abs_path, (two_seconds_ago, two_seconds_ago))
    deleted = cleanup_temp_images(max_age_seconds=2)
    assert path in deleted
    assert not default_storage.exists(path)


@pytest.mark.django_db
def test_cleanup_temp_images_keeps_recent_file(tmp_path):
    cf = ContentFile(b"recent", name="recent.png")
    path = save_temp_image(cf)
    deleted = cleanup_temp_images(max_age_seconds=9999)
    assert path not in deleted
    assert default_storage.exists(path)
