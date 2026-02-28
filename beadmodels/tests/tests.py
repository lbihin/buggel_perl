"""
Tests pour l'application beadmodels.

Focus : comportement metier observable, pas implementation interne.
"""

import io

import numpy as np
import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse
from PIL import Image

from beadmodels.forms import BeadModelForm
from beadmodels.models import BeadBoard, BeadModel
from beads.forms import BeadForm
from beads.models import Bead
from shapes.forms import BeadShapeForm
from shapes.models import BeadShape

# ------------------- Fixtures -------------------


@pytest.fixture
def user():
    return User.objects.create_user(
        username="testuser", email="test@example.com", password="testpass123"
    )


@pytest.fixture
def authenticated_client(client, user):
    client.login(username="testuser", password="testpass123")
    return client


@pytest.fixture
def bead_board():
    return BeadBoard.objects.create(
        name="Support Test",
        width_pegs=29,
        height_pegs=29,
        description="Support de test",
    )


@pytest.fixture
def test_image():
    image = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return SimpleUploadedFile("test.jpg", buf.read(), content_type="image/jpeg")


@pytest.fixture
def bead_model(user, test_image, bead_board):
    model = BeadModel.objects.create(
        name="Modèle Test",
        description="Description test",
        creator=user,
        is_public=True,
        board=bead_board,
    )
    model.original_image.save("test.jpg", test_image, save=True)
    return model


@pytest.fixture
def bead_shape(user):
    return BeadShape.objects.create(
        name="Forme Test",
        shape_type="rectangle",
        width=10,
        height=15,
        creator=user,
        is_default=True,
    )


@pytest.fixture
def bead(user):
    return Bead.objects.create(
        creator=user,
        name="Perle Rouge",
        red=255,
        green=0,
        blue=0,
        quantity=100,
    )


# ------------------- Validation formulaires -------------------


@pytest.mark.django_db
class TestForms:
    """Valide que les formulaires acceptent/refusent les bons inputs."""

    def test_bead_model_form_requires_image(self):
        form = BeadModelForm(data={"name": "M", "description": "D", "is_public": True})
        assert not form.is_valid()
        assert "original_image" in form.errors

    def test_bead_form_rejects_out_of_range_color(self):
        form = BeadForm(
            data={
                "name": "Bad",
                "red": 300,
                "green": 0,
                "blue": 255,
                "quantity": 1,
            }
        )
        assert not form.is_valid()
        assert "red" in form.errors

    def test_shape_form_requires_dimensions_for_rectangle(self):
        form = BeadShapeForm(data={"name": "R", "shape_type": "rectangle"})
        form.is_valid()
        assert "width" in form.errors and "height" in form.errors


# ------------------- Vues & permissions -------------------


@pytest.mark.django_db
class TestViews:
    """Verifie les regles d'acces : public/prive, proprietaire/anonyme."""

    def test_my_models_requires_login(self, client):
        response = client.get(reverse("beadmodels:my_models"))
        assert response.status_code == 302  # redirect to login

    def test_public_model_visible_to_anonymous(self, client, bead_model):
        url = reverse("beadmodels:details", kwargs={"pk": bead_model.pk})
        response = client.get(url)
        assert response.status_code == 200

    def test_private_model_hidden_from_others(
        self, client, user, test_image, bead_board
    ):
        private = BeadModel.objects.create(
            name="Prive",
            creator=user,
            is_public=False,
            original_image=test_image,
            board=bead_board,
        )
        url = reverse("beadmodels:details", kwargs={"pk": private.pk})
        response = client.get(url, follow=True)
        assert len(response.redirect_chain) > 0

    def test_owner_can_see_private_model(self, authenticated_client, bead_model):
        bead_model.is_public = False
        bead_model.save()
        url = reverse("beadmodels:details", kwargs={"pk": bead_model.pk})
        response = authenticated_client.get(url)
        assert response.status_code == 200

    def test_owner_can_delete_model(self, authenticated_client, bead_model):
        url = reverse("beadmodels:delete", kwargs={"pk": bead_model.pk})
        authenticated_client.post(url, follow=True)
        assert not BeadModel.objects.filter(pk=bead_model.pk).exists()


# ------------------- Integration : workflows metier -------------------


@pytest.mark.django_db
class TestIntegration:
    """Tests de bout en bout simulant de vrais parcours utilisateur."""

    def test_basic_model_lifecycle(self, user, authenticated_client, bead_board):
        """Creer, voir, modifier, supprimer un modele."""
        img = Image.new("RGB", (100, 100), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        upload = SimpleUploadedFile("lc.jpg", buf.read(), content_type="image/jpeg")

        model = BeadModel.objects.create(
            name="Lifecycle",
            creator=user,
            is_public=True,
            board=bead_board,
        )
        model.original_image.save("lc.jpg", upload, save=True)

        # See it
        resp = authenticated_client.get(
            reverse("beadmodels:details", kwargs={"pk": model.pk})
        )
        assert resp.status_code == 200

        # Update
        model.name = "Updated"
        model.save()
        model.refresh_from_db()
        assert model.name == "Updated"

        # Delete
        pk = model.pk
        model.delete()
        assert not BeadModel.objects.filter(pk=pk).exists()

    def test_public_private_permissions(self, user, authenticated_client, bead_board):
        """Un autre utilisateur ne peut pas voir un modele prive."""
        other = User.objects.create_user("other", password="other123")
        other_client = Client()
        other_client.login(username="other", password="other123")

        img = Image.new("RGB", (50, 50), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        model = BeadModel.objects.create(
            name="Priv",
            creator=user,
            is_public=False,
            board=bead_board,
        )
        model.original_image.save(
            "p.jpg",
            SimpleUploadedFile("p.jpg", buf.read(), "image/jpeg"),
            save=True,
        )

        url = reverse("beadmodels:details", kwargs={"pk": model.pk})

        # Owner sees it
        assert authenticated_client.get(url).status_code == 200
        # Other gets redirected
        assert len(other_client.get(url, follow=True).redirect_chain) > 0


# ------------------- Service image_processing -------------------


@pytest.mark.django_db
class TestImageProcessingService:
    """Tests pour le service d'image extrait dans services/image_processing.py."""

    def test_reduce_colors_produces_correct_shape(self):
        from beadmodels.services.image_processing import reduce_colors

        arr = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        result = reduce_colors(arr, n_colors=4)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

    def test_reduce_colors_with_user_palette(self):
        from beadmodels.services.image_processing import reduce_colors

        arr = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        result = reduce_colors(arr, n_colors=3, user_colors=palette)
        unique = np.unique(result.reshape(-1, 3), axis=0)
        # All result colors must be from the user palette
        for color in unique:
            assert any(np.array_equal(color, p) for p in palette)

    def test_compute_palette_returns_sorted_entries(self):
        from beadmodels.services.image_processing import compute_palette

        # 3x3 image: 6 red pixels, 3 blue
        pixels = np.array(
            [[[255, 0, 0]] * 3, [[255, 0, 0]] * 3, [[0, 0, 255]] * 3],
            dtype=np.uint8,
        )
        palette = compute_palette(total_beads=9, reduced_pixels=pixels)
        assert len(palette) == 2
        assert palette[0]["count"] > palette[1]["count"]
        assert "rgb(" in palette[0]["color"]
        assert palette[0]["hex"].startswith("#")

    def test_resolve_shape_spec_defaults(self):
        from beadmodels.services.image_processing import resolve_shape_spec

        spec = resolve_shape_spec(None)
        assert spec.grid_width == 29
        assert spec.grid_height == 29
        assert not spec.use_circle_mask

    def test_resolve_shape_spec_with_shape(self, user):
        from beadmodels.services.image_processing import resolve_shape_spec

        shape = BeadShape.objects.create(
            name="Circle",
            shape_type="circle",
            diameter=20,
            creator=user,
        )
        spec = resolve_shape_spec(shape.pk)
        assert spec.grid_width == 20
        assert spec.grid_height == 20
        assert spec.use_circle_mask
        assert spec.circle_diameter == 20

    def test_generate_preview_returns_base64(self, user):
        from beadmodels.services.image_processing import generate_preview

        shape = BeadShape.objects.create(
            name="Rect",
            shape_type="rectangle",
            width=5,
            height=5,
            creator=user,
        )
        # Create a simple base64 image
        img = Image.new("RGB", (50, 50), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        import base64

        b64 = base64.b64encode(buf.read()).decode()

        result = generate_preview(
            image_base64=b64,
            shape_id=shape.pk,
            color_reduction=4,
        )
        assert result.image_base64  # non-empty string
        assert result.reduced_pixels is not None
        assert result.reduced_pixels.shape == (5, 5, 3)

    def test_generate_model_returns_complete_result(self, user):
        from beadmodels.services.image_processing import generate_model

        shape = BeadShape.objects.create(
            name="Sq",
            shape_type="square",
            size=4,
            creator=user,
        )
        img = Image.new("RGB", (40, 40), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        import base64

        b64 = base64.b64encode(buf.read()).decode()

        result = generate_model(
            image_base64=b64,
            shape_id=shape.pk,
            color_reduction=2,
        )
        assert result.image_base64
        assert result.grid_width == 4
        assert result.grid_height == 4
        assert result.total_beads == 16
        assert len(result.palette) >= 1


# ------------------- Wizard finalisation -------------------


@pytest.mark.django_db
class TestFinalizeStep:
    """Tests pour la derniere etape du wizard (sauvegarde en BDD)."""

    def test_finalize_step_renders_form(
        self, authenticated_client, test_image, bead_board
    ):
        url = reverse("beadmodels:create")
        authenticated_client.post(url, {"image": test_image}, follow=True)
        authenticated_client.post(
            url, {"color_reduction": 8, "use_available_colors": "on"}, follow=True
        )
        session = authenticated_client.session
        session["model_creation_wizard_step"] = 3
        session.save()

        response = authenticated_client.get(url + "?w=1")
        assert response.status_code == 200
        content = response.content.decode()
        assert "Finalisation" in content
        assert "Nom du modèle" in content

    def test_finalize_step_saves_model(
        self, authenticated_client, test_image, bead_board
    ):
        url = reverse("beadmodels:create")
        authenticated_client.post(url, {"image": test_image}, follow=True)
        authenticated_client.post(url, {"color_reduction": 8}, follow=True)
        session = authenticated_client.session
        session["model_creation_wizard_step"] = 3
        session.save()

        response = authenticated_client.post(
            url + "?w=1",
            {
                "name": "Modèle Finalisé",
                "description": "Via wizard",
                "board": bead_board.pk,
                "is_public": "on",
                "tags": "test",
            },
            follow=True,
        )
        assert response.status_code == 200
        assert BeadModel.objects.filter(name="Modèle Finalisé").exists()
        model = BeadModel.objects.get(name="Modèle Finalisé")
        assert model.metadata.get("final_color_reduction") is not None
