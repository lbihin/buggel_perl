import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
def test_model_creation_wizard_reset(client):
    user = User.objects.create_user(username="wizuser", password="pass12345")
    client.login(username="wizuser", password="pass12345")
    url = reverse("beadmodels:model_creation_wizard")
    # First access sets step 1
    resp1 = client.get(url)
    assert resp1.status_code == 200
    assert client.session.get("model_creation_wizard_step") == 1
    # Simulate adding data to session then reset
    session = client.session
    session["model_creation_wizard"] = {"dummy": True}
    session.save()
    resp_reset = client.get(url + "?reset=true")
    assert resp_reset.status_code == 302  # Redirect after reset
    # After redirect follow
    resp_follow = client.get(url)
    assert resp_follow.status_code == 200
    assert "model_creation_wizard" not in client.session
    assert client.session.get("model_creation_wizard_step") == 1


@pytest.mark.django_db
def test_pixelization_wizard_auto_redirect(client):
    """Pixelization wizard route currently redirects to model_creation; ensure redirect works."""
    user = User.objects.create_user(username="wizuser2", password="pass12345")
    client.login(username="wizuser2", password="pass12345")
    url = reverse("beadmodels:pixelization_wizard")
    resp = client.get(url)
    # Should redirect to model creation wizard
    assert resp.status_code in (302, 301)
    assert resp.headers.get("Location", "").endswith("/model-creation/")
