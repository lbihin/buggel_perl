import pytest
from django.contrib.auth.models import User


# ------------------- Fixtures -------------------
@pytest.fixture
def user():
    """Création d'un utilisateur de test."""
    user = User.objects.create_user(
        username="testuser", email="test@example.com", password="testpass123"
    )
    return user


@pytest.fixture
def authenticated_client(client, user):
    """Client authentifié avec l'utilisateur de test."""
    client.login(username="testuser", password="testpass123")
    return client


@pytest.mark.django_db
class TestUserProfileViews:
    def test_usr_settings_get(self, authenticated_client):
        """Test de la vue usr_settings pour une requête GET."""
        response = authenticated_client.get("/fr/account/settings/")
        assert response.status_code == 200
        assert "active_tab" in response.context
        assert response.context["active_tab"] == "profile"
        assert "form" in response.context
