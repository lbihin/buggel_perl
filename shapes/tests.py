from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from .models import BeadShape, CustomShape


class BeadShapeModelTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

    def test_create_rectangle_shape(self):
        shape = BeadShape.objects.create(
            name="Rectangle Test",
            shape_type="rectangle",
            width=10,
            height=20,
            creator=self.user,
        )
        self.assertEqual(shape.get_dimensions_display(), "10×20")
        self.assertEqual(shape.get_parameters(), {"width": 10, "height": 20})
        self.assertEqual(str(shape), "Rectangle Test")

    def test_create_square_shape(self):
        shape = BeadShape.objects.create(
            name="Carré Test",
            shape_type="square",
            size=15,
            creator=self.user,
        )
        self.assertEqual(shape.get_dimensions_display(), "15×15")
        self.assertEqual(shape.get_parameters(), {"size": 15})
        self.assertEqual(str(shape), "Carré Test")

    def test_create_circle_shape(self):
        shape = BeadShape.objects.create(
            name="Cercle Test",
            shape_type="circle",
            diameter=8,
            creator=self.user,
        )
        self.assertEqual(shape.get_dimensions_display(), "∅8")
        self.assertEqual(shape.get_parameters(), {"diameter": 8})
        self.assertEqual(str(shape), "Cercle Test")


class CustomShapeModelTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.base_shape = BeadShape.objects.create(
            name="Base Rectangle",
            shape_type="rectangle",
            width=5,
            height=10,
            creator=self.user,
        )

    def test_create_custom_shape(self):
        custom = CustomShape.objects.create(
            user=self.user,
            base_shape=self.base_shape,
            name="Custom Rectangle",
            shape_type="rectangle",
            width=7,
            height=12,
        )
        self.assertEqual(str(custom), "Custom Rectangle (5×10)")
        self.assertEqual(custom.get_parameters(), {"width": 7, "height": 12})


class BeadShapeBDDTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="bdduser", email="bdd@example.com", password="bddpass123"
        )
        self.client = Client()
        self.client.login(username="bdduser", password="bddpass123")

    def test_user_can_create_rectangle_shape_via_view(self):
        """
        Étant donné qu'un utilisateur authentifié souhaite créer une nouvelle forme rectangulaire,
        lorsqu'il soumet le formulaire avec un nom, une largeur et une hauteur,
        alors la forme est créée et visible dans la liste des formes personnalisées.
        """
        response = self.client.post(
            reverse("shapes:create_shape"),
            {
                "name": "Rectangle BDD",
                "shape_type": "rectangle",
                "width": 12,
                "height": 8,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            BeadShape.objects.filter(name="Rectangle BDD", creator=self.user).exists()
        )
        self.assertContains(response, "Rectangle BDD")

    def test_user_cannot_delete_default_shape(self):
        """
        Étant donné qu'une forme par défaut existe,
        lorsqu'un utilisateur tente de la supprimer,
        alors l'opération échoue et un message d'erreur est affiché.
        """
        default_shape = BeadShape.objects.create(
            name="Forme Défaut",
            shape_type="square",
            size=10,
            is_default=True,
        )
        response = self.client.post(
            reverse("shapes:delete_shape", args=[default_shape.pk]), follow=True
        )
        self.assertContains(
            response, "Les formes par défaut ne peuvent pas être supprimées"
        )
        self.assertTrue(BeadShape.objects.filter(pk=default_shape.pk).exists())

    def test_user_can_edit_own_shape(self):
        """
        Étant donné qu'un utilisateur a créé une forme,
        lorsqu'il modifie le nom de la forme via la vue d'édition,
        alors le nouveau nom est enregistré et visible dans la liste.
        """
        shape = BeadShape.objects.create(
            name="Ancien Nom",
            shape_type="circle",
            diameter=5,
            creator=self.user,
        )
        response = self.client.post(
            reverse("shapes:update_shape", args=[shape.pk]),
            {
                "name": "Nouveau Nom",
                "shape_type": "circle",
                "diameter": 5,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        shape.refresh_from_db()
        self.assertEqual(shape.name, "Nouveau Nom")
        self.assertContains(response, "Nouveau Nom")

    def test_user_cannot_edit_others_shape(self):
        """
        Étant donné qu'une forme appartient à un autre utilisateur,
        lorsqu'un utilisateur tente de la modifier,
        alors l'opération échoue et un message d'erreur est affiché.
        """
        other = get_user_model().objects.create_user(
            username="otheruser", email="other@example.com", password="otherpass"
        )
        shape = BeadShape.objects.create(
            name="Forme Autre",
            shape_type="rectangle",
            width=3,
            height=4,
            creator=other,
        )
        response = self.client.post(
            reverse("shapes:update_shape", args=[shape.pk]),
            {
                "name": "Tentative Modif",
                "shape_type": "rectangle",
                "width": 3,
                "height": 4,
            },
            follow=True,
        )
        self.assertContains(
            response, "Vous n'avez pas l'autorisation de modifier cette forme"
        )
        shape.refresh_from_db()
        self.assertEqual(shape.name, "Forme Autre")


# TODO: Ajouter des tests pour les vues et les formulaires (BeadShapeForm)
# TODO: Ajouter des tests d'intégration pour les workflows utilisateur
