from typing import Any
from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
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


UI_ELM_DET_TYPES = (
    ('rpa-us', 'Kevin Moran'),
    ('uied', 'UIED'),
    ('sam', 'SAM'),
    ('fast-sam', 'Fast-SAM'),
    ('screen2som', 'Screen2SOM'),
)

class Prefilters(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    configurations = JSONField(null=True, blank=True, default=default_prefilters_conf)
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:prefilters_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.technique_name + ' - skip? ' + str(self.skip)

class UIElementsDetection(models.Model):

    title = models.CharField(max_length=255)
    # TODO: Add optional OCR
    ocr = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    type = models.CharField(max_length=25, choices=UI_ELM_DET_TYPES, default='rpa-us')
    input_filename = models.CharField(max_length=50, default='log.csv')
    decision_point_activity = models.CharField(max_length=255, blank=True)
    configurations = JSONField(null=True, blank=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    ui_elements_classification = models.ForeignKey('UIElementsClassification', on_delete=models.SET_NULL, null=True)

    # Delete ui_elements_classification when deleting UIElementsDetection
    def delete(self, *args, **kwargs):
        self.ui_elements_classification.delete()
        super().delete(*args, **kwargs)

    def get_absolute_url(self):
        return reverse("featureextraction:ui_detection_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type + ' - skip? ' + str(self.skip)

def get_ui_elements_classification_image_shape():
    return [64, 64, 3]


def get_ui_elements_classification_old_classes():
    return 'x0_Button, x0_CheckBox, x0_CheckedTextView, x0_EditText, x0_ImageButton, x0_ImageView, x0_NumberPicker, x0_RadioButton', 
'x0_RatingBar, x0_SeekBar, x0_Spinner, x0_Switch, x0_TextView, x0_ToggleButton'.split(', ') # this returns a list

def get_ui_elements_classification_classes():
    return "Button, Checkbox, CheckedTextView, EditText, ImageButton, ImageView, NumberPicker, RadioButton, RatingBar, SeekBar, Spinner, Switch, TextView, ToggleButton".split(', ') # this returns a list

class CNNModels(models.Model):
    name = models.CharField(max_length=25, unique=True)
    path = models.CharField(max_length=255, unique=True)
    ui_elements_classification_image_shape = ArrayField(models.IntegerField(null=True, blank=True), default=get_ui_elements_classification_image_shape)
    ui_elements_classification_classes = ArrayField(models.CharField(max_length=50), default=get_ui_elements_classification_classes)
    text_classname = models.CharField(max_length=50, default="TextView")
   
    def clean(self):
        if (self.text_classname not in self.ui_elements_classification_classes):
            raise ValidationError("text_classname must be one of the ui_elements_classification_classes")

class UIElementsClassification(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    # TODO: Change this to a foreign key
    model = models.CharField(max_length=255)
    model_properties = models.CharField(max_length=255, default="resources/models/custom-v2-classes.json")
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("featureextraction:ui_classification_list", args=[str(self.case_study_id)])
        
    def __str__(self):
        return 'type: ' + self.type + ' - model: ' + self.model

class Postfilters(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    configurations = JSONField(null=True, blank=True, default=default_filters_conf)
    type = models.CharField(max_length=25, default='rpa-us')
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:postfilters_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.technique_name + ' - skip? ' + str(self.skip)
    

class FeatureExtractionTechnique(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    identifier = models.CharField(max_length=25)
    type = models.CharField(max_length=255, default='SINGLE')
    technique_name = models.CharField(max_length=255, default='count')
    relevant_compos_predicate = models.CharField(max_length=255, default="compo['relevant'] == 'True'")
    consider_relevant_compos = models.BooleanField(default=False)
    configurations = JSONField(null=True, blank=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def clean(self):
        cleaned_data = super().clean()
        if not UIElementsDetection.objects.exists(case_study__id=self.case_study.id):
            raise ValidationError("To be able to apply a feature extraction technique, UI Element Detection has to be done")
        if not UIElementsClassification.objects.exists(case_study__id=self.case_study.id):
            raise ValidationError("To be able to apply a feature extraction technique, UI Element Classification has to be done")
        return cleaned_data
    
    def get_absolute_url(self):
        return reverse("featureextraction:fe_technique_list", args=[str(self.case_study_id)])
        
    def __str__(self):
        return 'technique: ' + self.technique_name + ' - skip? ' + str(self.skip)