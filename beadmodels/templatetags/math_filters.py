from django import template

register = template.Library()


@register.filter
def div(value, arg):
    """
    Divise la valeur par l'argument.
    """
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return ""


@register.filter
def mul(value, arg):
    """
    Multiplie la valeur par l'argument.
    """
    try:
        return float(value) * float(arg)
    except ValueError:
        return ""
