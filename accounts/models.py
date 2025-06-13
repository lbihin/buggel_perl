from django.db import models


class UserSettings(models.Model):
    user = models.OneToOneField("auth.User", on_delete=models.CASCADE)
    set_public = models.BooleanField(
        default=False, verbose_name="Set public by default"
    )

    def __str__(self):
        return f"{self.user.username}'s settings"
