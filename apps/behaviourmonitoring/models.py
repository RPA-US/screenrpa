from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from django.urls import reverse

# Create your models here.
class GazeAnalysis(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=25, default='attention-points')
    configurations = JSONField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("analyzer:casestudy_list")
    
    def __str__(self):
        return 'type: ' + self.type
