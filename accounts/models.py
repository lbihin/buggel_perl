from django.db import models

class UserSettingsQuerySet(models.QuerySet):
    def for_user(self, user):
        """
        Returns the settings for a specific user.
        
        Args:
            user: A User instance or user ID
        
        Returns:
            The UserSettings instance for the specified user
        """
        obj, _ = self.get_or_create(user=user)
        return obj


class UserSettingsManager(models.Manager):
    def get_queryset(self):
        return UserSettingsQuerySet(self.model, using=self._db)
        
    def for_user(self, user):
        return self.get_queryset().for_user(user)


class UserSettings(models.Model):
    user = models.OneToOneField("auth.User", on_delete=models.CASCADE)
    set_public = models.BooleanField(
        default=False, verbose_name="Set public by default"
    )

    objects = UserSettingsManager()

    def __str__(self):
        return f"{self.user.username}'s settings"
