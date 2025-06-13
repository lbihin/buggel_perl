from django.contrib import messages
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

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
        if active_tab == "profile":
            result = gerer_mise_a_jour_profil(request, context)
            if result:
                return result
        elif active_tab == "password":
            form = UserPasswordChangeForm(request.user, request.POST)
            if form.is_valid():
                user = form.save()
                # Important: update session to prevent user from being logged out
                update_session_auth_hash(request, user)
                messages.success(
                    request, "Votre mot de passe a été mis à jour avec succès!"
                )
                return redirect("accounts:user_settings")
            else:
                messages.error(request, "Veuillez corriger les erreurs ci-dessous.")
                context["password_form"] = form
            pass
        elif active_tab == "preferences":
            return gerer_mise_a_jour_preferences(request)

    remplir_contexte_pour_requete_get(active_tab, context, request)
    return render(request, "accounts/settings.html", context)


def gerer_mise_a_jour_profil(request, context):
    form = UserProfileForm(request.POST, instance=request.user)
    if form.is_valid():
        form.save()
        messages.success(request, "Votre profil a été mis à jour avec succès!")
        return redirect("accounts:user_settings")
    context["form"] = form
    return None


def gerer_mise_a_jour_preferences(request):
    # Récupérer ou créer les paramètres utilisateur
    user_settings, created = UserSettings.objects.get_or_create(user=request.user)

    form = UserSettingsForm(request.POST, instance=user_settings)
    if form.is_valid():
        form.save()
        messages.success(request, "Vos préférences ont été mises à jour avec succès!")
        return redirect("accounts:user_settings")
    else:
        # En cas d'erreur, rediriger avec un message d'erreur
        messages.error(
            request,
            "Une erreur s'est produite lors de la mise à jour de vos préférences.",
        )
        return redirect("accounts:user_settings")


def remplir_contexte_pour_requete_get(active_tab, context, request):
    if active_tab == "profile":
        context["form"] = UserProfileForm(instance=request.user)
    elif active_tab == "password":
        context["form"] = UserPasswordChangeForm(request.user)
    elif active_tab == "preferences":
        # Récupérer ou créer les paramètres utilisateur
        user_settings, _ = UserSettings.objects.get_or_create(user=request.user)
        context["form"] = UserSettingsForm(instance=user_settings)
