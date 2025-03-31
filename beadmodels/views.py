import numpy as np
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from PIL import Image

from .forms import BeadModelForm, UserRegistrationForm
from .models import BeadModel


def home(request):
    public_models = BeadModel.objects.filter(is_public=True).order_by('-created_at')[:12]
    return render(request, 'beadmodels/home.html', {'models': public_models})


def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Votre compte a été créé avec succès ! Vous pouvez maintenant vous connecter.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})


@login_required
def create_model(request):
    if request.method == 'POST':
        form = BeadModelForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=False)
            model.creator = request.user
            model.save()
            messages.success(request, 'Votre modèle a été créé avec succès!')
            return redirect('model_detail', pk=model.pk)
    else:
        form = BeadModelForm()
    return render(request, 'beadmodels/create_model.html', {'form': form})


def model_detail(request, pk):
    model = get_object_or_404(BeadModel, pk=pk)
    if not model.is_public and model.creator != request.user:
        messages.error(request, "Vous n'avez pas accès à ce modèle.")
        return redirect('home')
    return render(request, 'beadmodels/model_detail.html', {'model': model})


@login_required
def my_models(request):
    user_models = BeadModel.objects.filter(creator=request.user).order_by('-created_at')
    return render(request, 'beadmodels/my_models.html', {'models': user_models})
