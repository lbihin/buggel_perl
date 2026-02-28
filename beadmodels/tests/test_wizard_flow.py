"""
Tests du flux wizard de creation de modele.

Focus : navigation entre etapes (previous, next, reset).
"""

import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
def test_wizard_step1_accessible(client):
    """L'etape 1 du wizard est accessible pour un utilisateur connecte."""
    User.objects.create_user(username="wiz", password="pass12345")
    client.login(username="wiz", password="pass12345")
    resp = client.get(reverse("beadmodels:create"), follow=True)
    assert resp.status_code == 200


@pytest.mark.django_db
def test_wizard_previous_button_from_step2(client):
    """Le bouton Retour depuis l'etape 2 ramene a l'etape 1."""
    User.objects.create_user(username="wizprev", password="pass12345")
    client.login(username="wizprev", password="pass12345")
    url = reverse("beadmodels:create")
    client.get(url)

    # Simulate step 2 session state
    session = client.session
    session["model_creation_wizard"] = {"image_data": {"image_path": "dummy/path.png"}}
    session["model_creation_wizard_step"] = 2
    session.save()

    resp = client.post(url, {"q": "previous"})
    assert resp.status_code in (200, 302)

    # After following redirect, should be on step 1
    # Use ?w=1 to mimick internal wizard navigation (bare GET resets)
    resp2 = client.get(url + "?w=1")
    assert resp2.status_code == 200
    assert client.session.get("model_creation_wizard_step") == 1
