from django.shortcuts import render


def home(request):
    # If you're trying to render base.html directly
    return render(request, "base.html")

    # Or if base.html is meant to be extended by another template:
    # return render(request, 'home.html')
