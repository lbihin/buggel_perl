"""
Framework de wizard modulaire pour l'application beadmodels.

Ce module fournit des classes de base pour créer des wizards (assistants) modulaires
et extensibles dans l'application Django. Le framework est conçu pour être facilement
adaptable à différents types de flux de formulaires multi-étapes.
"""

from abc import ABC, abstractmethod

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View


class BaseWizard(View):
    """Classe de base pour les wizards modulaires."""

    name = "Wizard"
    steps = []
    session_key = "wizard_data"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steps = []
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialise les instances d'étapes."""
        for step_class in self.steps:
            self._steps.append(step_class(self))

    def get_current_step_number(self):
        """Récupère le numéro de l'étape actuelle depuis la session."""
        return self.request.session.get(f"{self.session_key}_step", 1)

    def set_current_step_number(self, step_number):
        """Définit le numéro de l'étape actuelle dans la session."""
        self.request.session[f"{self.session_key}_step"] = step_number

    def get_current_step(self):
        """Récupère l'instance de l'étape actuelle."""
        step_number = self.get_current_step_number()
        # S'assurer que l'étape existe
        if step_number > len(self._steps):
            step_number = 1
            self.set_current_step_number(step_number)

        # Les index commencent à 0, les étapes à 1
        return self._steps[step_number - 1]

    def get_step_by_number(self, step_number):
        """Récupère une instance d'étape par son numéro."""
        if 1 <= step_number <= len(self._steps):
            return self._steps[step_number - 1]
        return None

    def get_total_steps(self):
        """Renvoie le nombre total d'étapes."""
        return len(self._steps)

    def get_session_data(self):
        """Récupère les données du wizard stockées en session."""
        return self.request.session.get(self.session_key, {})

    def set_session_data(self, data):
        """Met à jour les données du wizard en session."""
        self.request.session[self.session_key] = data

    def update_session_data(self, new_data):
        """Met à jour les données du wizard avec de nouvelles valeurs."""
        data = self.get_session_data()
        data.update(new_data)
        self.set_session_data(data)

    def reset_wizard(self):
        """Réinitialise complètement le wizard."""
        if self.session_key in self.request.session:
            del self.request.session[self.session_key]
        if f"{self.session_key}_step" in self.request.session:
            del self.request.session[f"{self.session_key}_step"]

    def go_to_step(self, step_number, redirect_kwargs=None):
        """Change l'étape courante et redirige."""
        if redirect_kwargs is None:
            redirect_kwargs = {}

        self.set_current_step_number(step_number)

        # Construire l'URL avec les paramètres
        url = reverse(self.get_url_name())
        if redirect_kwargs:
            url += "?" + "&".join([f"{k}={v}" for k, v in redirect_kwargs.items()])

        return redirect(url)

    def go_to_next_step(self, redirect_kwargs=None):
        """Passe à l'étape suivante."""
        current_step = self.get_current_step_number()
        if current_step < self.get_total_steps():
            return self.go_to_step(current_step + 1, redirect_kwargs)
        return self.finish_wizard()

    def go_to_previous_step(self, redirect_kwargs=None):
        """Retourne à l'étape précédente."""
        current_step = self.get_current_step_number()
        if current_step > 1:
            return self.go_to_step(current_step - 1, redirect_kwargs)
        return self.go_to_step(1, redirect_kwargs)

    @abstractmethod
    def get_url_name(self):
        """Renvoie le nom d'URL pour ce wizard."""
        pass

    @abstractmethod
    def start_wizard(self):
        """Action à effectuer quand le wizard démarre."""
        pass

    @abstractmethod
    def finish_wizard(self):
        """Action à effectuer quand le wizard est terminé."""
        pass

    def get_context_data(self, **kwargs):
        """Renvoie le contexte de base pour tous les templates du wizard."""
        step = self.get_current_step()

        context = {
            "wizard_name": self.name,
            "total_steps": self.get_total_steps(),
            "current_step_number": self.get_current_step_number(),
        }
        context.update(kwargs)

        return context

    def dispatch(self, request, *args, **kwargs):
        """Point d'entrée principal pour le traitement des requêtes."""
        self.request = request

        # Gestion du reset du wizard
        if self.request.GET.get("q") == "reset":
            self.reset_wizard()
            messages.info(self.request, f"Le {self.name.lower()} a été réinitialisé.")
            # return self.handle_reset()

        # Vérification de la disponibilité des données
        # data = self.get_session_data()
        # current_step = self.get_current_step_number()

        # # Si on est à une étape > 1 mais qu'on n'a pas de données d'image, on revient à l'étape 1
        # if current_step > 1 and (
        #     "image_data" not in data or not data.get("image_data")
        # ):
        #     messages.warning(request, "Veuillez d'abord charger une image.")
        #     self.set_current_step_number(1)
        #     return redirect(reverse(self.get_url_name()))

        # Gestion des boutons précédent/suivant
        # if request.method == "POST":
        #     if "previous_step" in request.POST:
        #         return self.go_to_previous_step(self.get_redirect_kwargs())

        # Déléguer à l'étape courante
        step = self.get_current_step()
        if request.method == "GET":
            return step.handle_get(**kwargs)
        elif request.method == "POST":
            return step.handle_post(**kwargs)

    def get_redirect_kwargs(self):
        """Récupère les paramètres à conserver lors des redirections."""
        return {}

    def handle_reset(self):
        """Gère la réinitialisation du wizard."""
        messages.info(self.request, f"Le {self.name.lower()} a été réinitialisé.")
        return redirect(reverse(self.get_url_name()))


@method_decorator(login_required, name="dispatch")
class LoginRequiredWizard(BaseWizard):
    """Version du wizard qui nécessite une connexion."""

    pass


class WizardStep(ABC):
    """Classe abstraite représentant une étape du wizard."""

    name = "Étape"
    template = None
    form_class = None
    position = 0

    def __init__(self, wizard: BaseWizard):
        self.wizard = wizard
        # Ne plus accéder à request ici car elle n'est pas encore disponible
        # lors de l'initialisation

    @abstractmethod
    def handle_get(self, **kwargs):
        """Gère les requêtes GET pour cette étape."""
        pass

    @abstractmethod
    def handle_post(self, **kwargs):
        """Gère les requêtes POST pour cette étape."""
        pass

    def get_context_data(self, **kwargs):
        """Renvoie le contexte pour le template."""
        context = {
            "wizard": self.wizard,
            "current_step": self,
            "step_name": self.name,
            "position": self.position,
            "total_steps": self.wizard.get_total_steps(),
        }
        context.update(kwargs)
        return context

    def render_template(self, context=None):
        """Rend le template avec le contexte fourni."""
        if context is None:
            context = {}

        full_context = self.get_context_data(**context)
        return render(self.wizard.request, self.template, full_context)
