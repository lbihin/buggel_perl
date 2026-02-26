import json

from django.contrib.auth.decorators import login_required
from django.db import IntegrityError
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render

from .models import Bead

# Common bead colors for quick-pick palette
PRESET_COLORS = [
    "#ff0000",
    "#ff6600",
    "#ffcc00",
    "#ffff00",
    "#00cc00",
    "#009933",
    "#00cccc",
    "#3399ff",
    "#0000ff",
    "#9933ff",
    "#ff00ff",
    "#ff66b2",
    "#a52a2a",
    "#ffc0cb",
    "#ffffff",
    "#808080",
    "#333333",
    "#000000",
]


def _get_threshold(request):
    preferences = getattr(request, "app_preferences", None)
    return (
        getattr(preferences, "bead_low_quantity_threshold", 20) if preferences else 20
    )


@login_required
def bead_edit_quantity_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.GET.get("cancel"):
        threshold = _get_threshold(request)
        return render(
            request,
            "beads/partials/bead_quantity_display.html",
            {"bead": bead, "threshold": threshold},
        )
    return render(request, "beads/partials/bead_edit_quantity.html", {"bead": bead})


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

            threshold = _get_threshold(request)
            response = render(
                request,
                "beads/partials/bead_quantity_display.html",
                {"bead": bead, "threshold": threshold},
            )
            # Trigger stock alert refresh + row class update
            is_low = bead.quantity is not None and bead.quantity <= threshold
            response["HX-Trigger"] = json.dumps(
                {
                    "quantityUpdated": {
                        "beadId": bead.pk,
                        "isLow": is_low,
                    }
                }
            )
            return response
        except (ValueError, TypeError):
            return render(
                request,
                "beads/partials/bead_edit_quantity.html",
                {
                    "bead": bead,
                    "error": "La quantité doit être un nombre entier positif.",
                },
            )

    return bead_edit_quantity_htmx(request, pk)


@login_required
def bead_edit_color_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.GET.get("cancel"):
        return render(request, "beads/partials/bead_color_display.html", {"bead": bead})
    return render(
        request,
        "beads/partials/bead_edit_color.html",
        {"bead": bead, "preset_colors": PRESET_COLORS},
    )


@login_required
def bead_update_color_htmx(request, pk):
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.method == "POST":
        try:
            hex_color = request.POST.get("hex_color", "").strip().lstrip("#")
            if len(hex_color) == 6:
                red = int(hex_color[0:2], 16)
                green = int(hex_color[2:4], 16)
                blue = int(hex_color[4:6], 16)
            else:
                # Fallback to individual RGB values
                red = max(0, min(255, int(request.POST.get("red", "0"))))
                green = max(0, min(255, int(request.POST.get("green", "0"))))
                blue = max(0, min(255, int(request.POST.get("blue", "0"))))

            bead.red = red
            bead.green = green
            bead.blue = blue
            bead.name = f"Perle #{red:02x}{green:02x}{blue:02x}"
            bead.save(update_fields=["red", "green", "blue", "name"])
            return render(
                request, "beads/partials/bead_color_display.html", {"bead": bead}
            )
        except (ValueError, TypeError):
            return render(
                request,
                "beads/partials/bead_edit_color.html",
                {
                    "bead": bead,
                    "preset_colors": PRESET_COLORS,
                    "error": "Couleur invalide.",
                },
            )

    return bead_edit_color_htmx(request, pk)


@login_required
def stock_alert_htmx(request):
    """Returns the stock alert banner HTML (called via HX-Trigger after quantity update)."""
    threshold = _get_threshold(request)
    beads = Bead.objects.filter(creator=request.user)
    low_quantity_beads = [
        b for b in beads if b.quantity is not None and b.quantity <= threshold
    ]
    return render(
        request,
        "beads/partials/stock_alert.html",
        {"low_quantity_beads": low_quantity_beads, "threshold": threshold},
    )


# ── Row-level inline editing ────────────────────────────────────────


@login_required
def bead_edit_row_htmx(request, pk):
    """GET: returns edit row; GET?cancel=true: returns display row."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    threshold = _get_threshold(request)
    if request.GET.get("cancel"):
        return render(
            request,
            "beads/partials/bead_row_display.html",
            {"bead": bead, "threshold": threshold},
        )
    return render(
        request,
        "beads/partials/bead_row_edit.html",
        {"bead": bead},
    )


@login_required
def bead_save_row_htmx(request, pk):
    """POST: saves all fields, returns display row."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.method != "POST":
        return bead_edit_row_htmx(request, pk)

    # Parse color
    hex_color = request.POST.get("hex_color", "").strip().lstrip("#")
    if len(hex_color) == 6:
        try:
            bead.red = int(hex_color[0:2], 16)
            bead.green = int(hex_color[2:4], 16)
            bead.blue = int(hex_color[4:6], 16)
        except ValueError:
            pass

    # Parse name
    name = request.POST.get("name", "").strip()
    if name:
        bead.name = name

    # Parse quantity
    quantity_str = request.POST.get("quantity", "").strip()
    if quantity_str == "":
        bead.quantity = None
    else:
        try:
            bead.quantity = max(0, int(quantity_str))
        except (ValueError, TypeError):
            pass

    # Parse notes
    bead.notes = request.POST.get("notes", "").strip()

    try:
        bead.save()
    except IntegrityError:
        return render(
            request,
            "beads/partials/bead_row_edit.html",
            {"bead": bead, "error": "Une perle avec ce nom existe déjà."},
        )

    threshold = _get_threshold(request)
    is_low = bead.quantity is not None and bead.quantity <= threshold
    response = render(
        request,
        "beads/partials/bead_row_display.html",
        {"bead": bead, "threshold": threshold},
    )
    response["HX-Trigger"] = json.dumps(
        {"quantityUpdated": {"beadId": bead.pk, "isLow": is_low}}
    )
    return response


@login_required
def bead_new_row_htmx(request):
    """GET: returns a new empty edit row for inline creation."""
    return render(request, "beads/partials/bead_row_new.html")


@login_required
def bead_create_inline_htmx(request):
    """POST: creates a new bead inline, returns display row."""
    if request.method != "POST":
        return bead_new_row_htmx(request)

    hex_color = request.POST.get("hex_color", "ff0000").strip().lstrip("#")
    if len(hex_color) == 6:
        try:
            red = int(hex_color[0:2], 16)
            green = int(hex_color[2:4], 16)
            blue = int(hex_color[4:6], 16)
        except ValueError:
            red, green, blue = 255, 0, 0
    else:
        red, green, blue = 255, 0, 0

    name = request.POST.get("name", "").strip()
    if not name:
        name = f"Perle #{hex_color}"

    quantity_str = request.POST.get("quantity", "").strip()
    quantity = None
    if quantity_str:
        try:
            quantity = max(0, int(quantity_str))
        except (ValueError, TypeError):
            pass

    notes = request.POST.get("notes", "").strip()

    try:
        bead = Bead.objects.create(
            creator=request.user,
            name=name,
            red=red,
            green=green,
            blue=blue,
            quantity=quantity,
            notes=notes,
        )
    except IntegrityError:
        return render(
            request,
            "beads/partials/bead_row_new.html",
            {
                "error": "Une perle avec ce nom existe déjà.",
                "name": name,
                "hex_color": f"#{hex_color}",
                "quantity": quantity_str,
                "notes": notes,
            },
        )

    threshold = _get_threshold(request)
    is_low = bead.quantity is not None and bead.quantity <= threshold
    response = render(
        request,
        "beads/partials/bead_row_display.html",
        {"bead": bead, "threshold": threshold},
    )
    response["HX-Trigger"] = json.dumps(
        {"quantityUpdated": {"beadId": bead.pk, "isLow": is_low}}
    )
    return response


@login_required
def bead_delete_inline_htmx(request, pk):
    """DELETE: deletes bead, returns empty to remove the row."""
    bead = get_object_or_404(Bead, pk=pk, creator=request.user)
    if request.method == "DELETE":
        bead.delete()
        response = HttpResponse("")
        response["HX-Trigger"] = json.dumps({"quantityUpdated": {}})
        return response
    return HttpResponse(status=405)
