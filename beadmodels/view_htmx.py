from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render

from .models import Bead


@login_required
def bead_edit_quantity_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.GET.get("cancel"):
        from django.conf import settings

        threshold = getattr(settings, "BEAD_LOW_QUANTITY_THRESHOLD", 20)
        context = {"bead": bead, "threshold": threshold}
        return render(
            request, "beadmodels/partials/bead_quantity_display.html", context
        )

    return render(
        request, "beadmodels/partials/bead_edit_quantity.html", {"bead": bead}
    )


@login_required
def bead_update_quantity_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.method == "POST":
        try:
            quantity_str = request.POST.get("quantity", "")
            if quantity_str.strip() == "":
                bead.quantity = None
            else:
                quantity = int(quantity_str)
                bead.quantity = max(0, quantity)
            bead.save(update_fields=["quantity"])
            from django.conf import settings

            threshold = getattr(settings, "BEAD_LOW_QUANTITY_THRESHOLD", 20)
            context = {"bead": bead, "threshold": threshold}
            return render(
                request, "beadmodels/partials/bead_quantity_display.html", context
            )
        except (ValueError, TypeError):
            context = {
                "bead": bead,
                "error": "La quantité doit être un nombre entier positif.",
            }
            return render(
                request, "beadmodels/partials/bead_edit_quantity.html", context
            )

    return bead_edit_quantity_htmx(request, pk)


@login_required
def bead_edit_color_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.GET.get("cancel"):
        return render(
            request, "beadmodels/partials/bead_color_display.html", {"bead": bead}
        )
    return render(request, "beadmodels/partials/bead_edit_color.html", {"bead": bead})


@login_required
def bead_update_color_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.method == "POST":
        try:
            red = max(0, min(255, int(request.POST.get("red", "0"))))
            green = max(0, min(255, int(request.POST.get("green", "0"))))
            blue = max(0, min(255, int(request.POST.get("blue", "0"))))
            bead.red = red
            bead.green = green
            bead.blue = blue
            bead.name = f"Perle #{red:02x}{green:02x}{blue:02x}"
            bead.save(update_fields=["red", "green", "blue", "name"])
            return render(
                request, "beadmodels/partials/bead_color_display.html", {"bead": bead}
            )
        except (ValueError, TypeError):
            return render(
                request,
                "beadmodels/partials/bead_edit_color.html",
                {
                    "bead": bead,
                    "error": "Les valeurs de couleur doivent être des nombres entiers entre 0 et 255.",
                },
            )

    return bead_edit_color_htmx(request, pk)
