import io

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from PIL import Image


@pytest.mark.django_db
def test_upload_step_stores_path_not_base64(client):
    user = User.objects.create_user(username="sessuser", password="pass12345")
    client.login(username="sessuser", password="pass12345")
    url = reverse("beadmodels:model_creation_wizard")
    # GET to initialize
    client.get(url)
    # Create in-memory image
    img = Image.new("RGB", (10, 10), color="red")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    upload = SimpleUploadedFile("test.png", bio.read(), content_type="image/png")
    resp = client.post(url, {"image": upload})
    assert resp.status_code in (302, 200)
    data = client.session.get("model_creation_wizard")
    assert data is not None
    image_data = data.get("image_data", {})
    assert "image_path" in image_data
    assert "image_base64" not in image_data  # Should not store raw base64 now
