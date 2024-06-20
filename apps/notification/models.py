import os
from django.conf import settings
from django.db import models
from enum import Enum
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.

class Status(Enum):
    SUCCESS = {
        'color': '#c0f4e4',
        'icon': 'fas fa-check'
    }
    INFO = {
        'color': '#b9d1ee',
        'icon': 'fas fa-info-circle'
    }
    WARNING = {
        'color': '#ffedb0',
        'icon': 'fas fa-exclamation-triangle'
    }
    ERROR = {
        'color': '#f2acac',
        'icon': 'fas fa-exclamation-circle'
    }
    PROCESSING = {
        'color': '#525f7f',
        'icon': 'fas fa-cogs'
    }

def get_default_status():
    return Status.INFO.value

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    short = models.CharField(max_length=100)
    message = models.TextField()
    read = models.BooleanField(default=False)
    href = models.CharField(max_length=100)
    created_at = models.DateTimeField(default=timezone.now)
    status = models.JSONField(default=get_default_status)

