from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField

class UIElementsDetection(models.Model):
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)

class NoiseFiltering(models.Model):
    type = models.CharField(max_length=25, default='attention-points')
    configurations = JSONField()
    # eyetracking_log_filename =  models.CharField(max_length=255, default="eyetracking_log.csv")

class UIElementsClassification(models.Model):
    model = models.CharField(max_length=255, default="resources/models/custom-v2.h5")
    model_properties = models.CharField(max_length=255, default="resources/models/custom-v2-classes.json")
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)

class FeatureExtractionTechnique(models.Model):
    technique_name = models.CharField(max_length=255, default='count')
    skip = models.BooleanField(default=False)
    identifier = models.CharField(max_length=25)
