from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    short = models.CharField(max_length=100)
    message = models.TextField()
    read = models.BooleanField(default=False)
    href = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)