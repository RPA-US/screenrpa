from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from django.urls import reverse

class UIElementsDetection(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("analyzer:casestudy_list")
    
    def __str__(self):
        return 'type: ' + self.type + ' - skip? ' + str(self.skip)

class NoiseFiltering(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=25, default='attention-points')
    configurations = JSONField()
    # eyetracking_log_filename =  models.CharField(max_length=255, default="eyetracking_log.csv")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("analyzer:casestudy_list")
    
    def __str__(self):
        return 'type: ' + self.type

class UIElementsClassification(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    model = models.CharField(max_length=255, default="resources/models/custom-v2.h5")
    model_properties = models.CharField(max_length=255, default="resources/models/custom-v2-classes.json")
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("analyzer:casestudy_list")
        
    def __str__(self):
        return 'type: ' + self.type + ' - model: ' + self.model

class FeatureExtractionTechnique(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    technique_name = models.CharField(max_length=255, default='count')
    skip = models.BooleanField(default=False)
    identifier = models.CharField(max_length=25)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("analyzer:casestudy_list")
        
    def __str__(self):
        return 'type: ' + self.technique_name + ' - skip? ' + str(self.skip)