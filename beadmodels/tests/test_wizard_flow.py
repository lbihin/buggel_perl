import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
def test_model_creation_wizard_reset(client):
    user = User.objects.create_user(username="wizuser", password="pass12345")
    client.login(username="wizuser", password="pass12345")
    url = reverse("beadmodels:create")
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
def test_model_creation_wizard_previous_button(client):
    """Vérifie que le bouton Retour de l'étape 2 ramène à l'étape 1."""
    user = User.objects.create_user(username="wizprev", password="pass12345")
    client.login(username="wizprev", password="pass12345")
    url = reverse("beadmodels:create")
    # Étape 1 GET
    client.get(url)
    # Étape 1 POST (upload simulé sans fichier en adaptant): on simule session image_data
    session = client.session
    session["model_creation_wizard"] = {"image_data": {"image_path": "dummy/path.png"}}
    session["model_creation_wizard_step"] = 2
    session.save()
    # Étape 2 POST avec previous_step
    resp = client.post(url, {"previous_step": "1"})
    assert resp.status_code in (200, 302)

    @pytest.mark.django_db
    def test_configuration_step_original_image_display(client, tmp_path):
        """Vérifie que l'image originale est affichée à l'étape 2 même si seulement path en session."""
        user = User.objects.create_user(username="wizimg", password="pass12345")
        client.login(username="wizimg", password="pass12345")
        url = reverse("beadmodels:create")
        client.get(url)
        # Créer une petite image temporaire
        from PIL import Image

        img_file = tmp_path / "orig.png"
        img = Image.new("RGB", (10, 10), color=(123, 50, 200))
        img.save(img_file)
        # Simuler path dans session
        session = client.session
        session["model_creation_wizard"] = {"image_data": {"image_path": str(img_file)}}
        session["model_creation_wizard_step"] = 2
        session.save()
        resp = client.get(url)
        assert resp.status_code == 200
        # Vérifier presence balise image base64
        assert b"data:image/png;base64" in resp.content

    @pytest.mark.django_db
    def test_configuration_step_missing_color_reduction_defaults(client, tmp_path):
        """POST sans color_reduction ne doit pas casser et doit utiliser 16 par défaut."""
        user = User.objects.create_user(username="wizcolor", password="pass12345")
        client.login(username="wizcolor", password="pass12345")
        url = reverse("beadmodels:create")
        client.get(url)
        # Image session
        from PIL import Image

        img_file = tmp_path / "orig2.png"
        Image.new("RGB", (8, 8), color=(10, 200, 30)).save(img_file)
        session = client.session
        session["model_creation_wizard"] = {"image_data": {"image_path": str(img_file)}}
        session["model_creation_wizard_step"] = 2
        session.save()
        # POST sans color_reduction
        resp = client.post(url, {"use_available_colors": "on"})
        # Devrait soit rester étape 2 (invalide) soit passer étape 3 mais ne doit pas 500
        assert resp.status_code in (200, 302)

    @pytest.mark.django_db
    def test_generate_transitions_to_finalize(client, tmp_path):
        """Vérifie que le bouton Générer passe à l'étape 3 et que l'image finale est chargée depuis un chemin."""
        user = User.objects.create_user(username="wizgen", password="pass12345")
        client.login(username="wizgen", password="pass12345")
        url = reverse("beadmodels:create")
        client.get(url)
        from PIL import Image

        img_file = tmp_path / "orig3.png"
        Image.new("RGB", (12, 12), color=(40, 80, 160)).save(img_file)
        session = client.session
        session["model_creation_wizard"] = {"image_data": {"image_path": str(img_file)}}
        session["model_creation_wizard_step"] = 2
        session.save()
        resp_post = client.post(url, {"color_reduction": 8, "generate": "1"})
        # Redirection vers étape 3
        assert resp_post.status_code in (302, 200)
        # Get finalize
        finalize = client.get(url)
        assert finalize.status_code == 200
        assert b"Finalisation" in finalize.content
        # Image doit être affichée
        assert b"data:image/png;base64" in finalize.content

    @pytest.mark.django_db
    def test_generate_without_color_field(client, tmp_path):
        """Le bouton Générer doit fonctionner même sans champ color_reduction dans POST."""
        user = User.objects.create_user(username="wizgennocolor", password="pass12345")
        client.login(username="wizgennocolor", password="pass12345")
        url = reverse("beadmodels:create")
        client.get(url)
        from PIL import Image

        img_file = tmp_path / "orig4.png"
        Image.new("RGB", (10, 10), color=(200, 40, 60)).save(img_file)
        session = client.session
        session["model_creation_wizard"] = {"image_data": {"image_path": str(img_file)}}
        session["model_creation_wizard_step"] = 2
        session.save()
        # POST sans color_reduction
        resp_post = client.post(url, {"generate": "1"})
        assert resp_post.status_code in (302, 200)
        finalize = client.get(url)
        assert finalize.status_code == 200
        assert b"Finalisation" in finalize.content

    # Après retour, nouvelle requête GET devrait être sur step 1
    resp2 = client.get(url)
    assert resp2.status_code == 200
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
