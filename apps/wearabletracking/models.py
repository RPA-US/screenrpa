from django.db import models
from django.contrib.auth.models import User

class FitbitToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    access_token = models.TextField()
    refresh_token = models.TextField()
    expires_in = models.IntegerField()
    token_type = models.CharField(max_length=20)
    scope = models.CharField(max_length=200)
    fitbit_user_id = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Token for user {self.user_id}"