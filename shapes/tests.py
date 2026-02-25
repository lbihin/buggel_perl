from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from .models import BeadShape


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
        self.assertContains(response, "autorisation de modifier cette forme")
        shape.refresh_from_db()
        self.assertEqual(shape.name, "Forme Autre")
