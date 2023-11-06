from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from apps.analyzer.models import CaseStudy
from django.urls import reverse

def default_prefilters_conf():
    return dict({
                "prefilter1":{
                    
                },
                "prefilter2":{
                
                }
                })
    
def default_filters_conf():
    return dict({
                "gaze":{
                    "UI_selector": "all",
                    "predicate": "(compo['row_min'] <= fixation_point_x) and (fixation_point_x <= compo['row_max']) and (compo['column_min'] <= fixation_point_y) and (fixation_point_y <= compo['column_max'])",
                    "only_leaf": True
                },
                "filter2":{

                }
                })

class Prefilters(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    configurations = JSONField(null=True, blank=True, default=default_prefilters_conf)
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:prefilters_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.technique_name + ' - skip? ' + str(self.skip)

class UIElementsDetection(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    type = models.CharField(max_length=25, default='rpa-us')
    input_filename = models.CharField(max_length=50, default='log.csv')
    configurations = JSONField(null=True, blank=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:ui_detection_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type + ' - skip? ' + str(self.skip)


class UIElementsClassification(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    model = models.CharField(max_length=255, default="resources/models/custom-v2.h5")
    model_properties = models.CharField(max_length=255, default="resources/models/custom-v2-classes.json")
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("featureextraction:ui_classification_list", args=[str(self.case_study_id)])
        
    def __str__(self):
        return 'type: ' + self.type + ' - model: ' + self.model

class Postfilters(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    configurations = JSONField(null=True, blank=True, default=default_filters_conf)
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:postfilters_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.technique_name + ' - skip? ' + str(self.skip)
    

class FeatureExtractionTechnique(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    identifier = models.CharField(max_length=25)
    type = models.CharField(max_length=255, default='SINGLE')
    technique_name = models.CharField(max_length=255, default='count')
    relevant_compos_predicate = models.CharField(max_length=255, default="compo['relevant'] == 'True'")
    consider_relevant_compos = models.BooleanField(default=False)
    configurations = JSONField(null=True, blank=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("featureextraction:fe_technique_list", args=[str(self.case_study_id)])
        
    def __str__(self):
        return 'technique: ' + self.technique_name + ' - skip? ' + str(self.skip)