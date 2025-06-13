from django.contrib import messages
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.urls import reverse

from accounts.forms import (
    UserPasswordChangeForm,
    UserProfileForm,
    UserRegistrationForm,
    UserSettingsForm,
)
from accounts.models import UserSettings


def logout(request):
    """
    Log out the user and redirect to the home page.
    """
    auth_logout(request)
    # Clear the session to ensure all user data is removed
    request.session.flush()
    # Optionally, you can add a message to inform the user
    # that they have been logged out successfully.
    messages.success(request, "Vous avez été déconnecté avec succès.")
    return redirect("home")


def register(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(
                request,
                "Votre compte a été créé avec succès ! Vous pouvez maintenant vous connecter.",
            )
            return redirect("login")
    else:
        form = UserRegistrationForm()
    return render(request, "accounts/register.html", {"form": form})


# Create your views here.
@login_required
def user_settings(request):
    active_tab = request.GET.get("tab", "profile")
    context = {"active_tab": active_tab}

    if request.method == "POST":
        result = traiter_formulaire_settings(request, active_tab, context)
        if result:
            return result

    remplir_contexte_pour_requete_get(active_tab, context, request)
    return render(request, "accounts/settings.html", context)


def traiter_formulaire_settings(request, active_tab, context):
    """Traite les formulaires selon l'onglet actif"""
    if active_tab == "profile":
        return gerer_mise_a_jour_profil(request, context)
    elif active_tab == "password":
        return gerer_mise_a_jour_mot_de_passe(request, context)
    elif active_tab == "preferences":
        return gerer_mise_a_jour_preferences(request, context)
    return None


def gerer_mise_a_jour_mot_de_passe(request, context):
    """Gère la mise à jour du mot de passe"""
    form = UserPasswordChangeForm(request.user, request.POST)
    if form.is_valid():
        user = form.save()
        update_session_auth_hash(request, user)
        messages.success(request, "Votre mot de passe a été mis à jour avec succès!")
        return redirect(reverse("accounts:user_settings") + "?tab=password")
    else:
        context["form"] = form
        return None


def gerer_mise_a_jour_profil(request, context):
    """Gère la mise à jour du profil utilisateur"""
    form = UserProfileForm(request.POST, instance=request.user)
    if form.is_valid():
        form.save()
        messages.success(request, "Votre profil a été mis à jour avec succès!")
        return redirect(reverse("accounts:user_settings") + "?tab=profile")
    context["form"] = form
    return None


def gerer_mise_a_jour_preferences(request, context=None):
    """Gère la mise à jour des préférences utilisateur"""
    # Récupérer ou créer les paramètres utilisateur
    user_settings, created = UserSettings.objects.get_or_create(user=request.user)

    form = UserSettingsForm(request.POST, instance=user_settings)
    if form.is_valid():
        form.save()
        messages.success(request, "Vos préférences ont été mises à jour avec succès!")
        return redirect(reverse("accounts:user_settings") + "?tab=preferences")
    else:
        # En cas d'erreur, retourner None et laisser le formulaire avec erreurs dans le contexte
        if context is not None:
            context["form"] = form
        return None


def remplir_contexte_pour_requete_get(active_tab, context, request):
    """Remplit le contexte avec le bon formulaire selon l'onglet actif"""

    # Configuration des formulaires par onglet
    form_configs = {
        "profile": {
            "form_class": UserProfileForm,
            "kwargs": {"instance": request.user},
        },
        "password": {
            "form_class": UserPasswordChangeForm,
            "kwargs": {"user": request.user},
        },
        "preferences": {
            "form_class": UserSettingsForm,
            "kwargs": {"instance": None},  # Sera rempli dynamiquement
        },
    }

    if active_tab in form_configs and "form" not in context:
        config = form_configs[active_tab]

        if active_tab == "preferences":
            # Cas spécial pour les préférences
            user_settings, _ = UserSettings.objects.get_or_create(user=request.user)
            context["form"] = config["form_class"](instance=user_settings)
        else:
            context["form"] = config["form_class"](**config["kwargs"])
