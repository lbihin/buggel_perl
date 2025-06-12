from django.contrib import messages
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from accounts.forms import UserProfileForm, UserRegistrationForm


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
            return gerer_mise_a_jour_profil(request, context)
        elif active_tab == "password":
            # TODO: Implémenter la logique de changement de mot de passe
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
    # TODO: Sauvegarder les préférences utilisateur
    messages.success(request, "Vos préférences ont été mises à jour avec succès!")
    return redirect("accounts:user_settings")


def remplir_contexte_pour_requete_get(active_tab, context, request):
    if active_tab == "profile":
        context["form"] = UserProfileForm(instance=request.user)
    # elif active_tab == "shapes":
    #     context["saved_shapes"] = BeadShape.objects.filter(
    #         creator=request.user
    #     ).order_by("-created_at")
    # elif active_tab == "shapes_new":
    #     context["form"] = BeadShapeForm()
    # elif active_tab == "beads":
    #     context["beads"] = Bead.objects.filter(creator=request.user)
