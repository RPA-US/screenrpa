from typing import Any
from django.db import models
import zipfile
import os
import time
import json

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.core.validators import RegexValidator
from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from private_storage.fields import PrivateFileField
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from django.urls import reverse
from core.settings import PRIVATE_STORAGE_ROOT, sep, SCREEN2SOM_CONFIG_PATH
from django.utils.translation import gettext_lazy as _
from django.db.models.signals import pre_delete
from django.dispatch import receiver

def unzip_file(zip_file_path, dest_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder_path)

def get_exp_foldername(exp_folder_complete_path):
    count = 0
    if "/" in exp_folder_complete_path:
        count+=1
        aux = "/"
    if "\\\\" in exp_folder_complete_path:
        count+=1
        aux = "\\\\"
    elif "\\" in exp_folder_complete_path:
        count+=1
        aux = "\\"
    if count>1:
         raise ValidationError(_("exp_folder_complete_path separators not coherent"))
    splitted_s = exp_folder_complete_path.split(aux)
    return splitted_s[len(splitted_s) - 1]

UI_ELM_DET_TYPES = (
    ('rpa-us', 'Kevin Moran'),
    ('uied', 'UIED'),
    ('sam', 'SAM'),
    ('fast-sam', 'Fast-SAM'),
    ('screen2som', 'Screen2SOM'),
)

class Prefilters(models.Model):
    title = models.CharField(max_length=255)
    preloaded = models.BooleanField(default=False, editable=True)
    preloaded_file = PrivateFileField("File", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    freeze = models.BooleanField(default=False, editable=True)
    # scale_factor = models.IntegerField(default=3, editable=True)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:prefilters_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.title + ' - active? ' + str(self.active)

@receiver(pre_delete, sender=Prefilters)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)

class UIElementsDetection(models.Model):
    preloaded = models.BooleanField(default=False, editable=True)
    preloaded_file = PrivateFileField("File", null=True, blank=True)
    freeze = models.BooleanField(default=False, editable=True)
    title = models.CharField(max_length=255)
    ocr = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    type = models.CharField(max_length=25, choices=UI_ELM_DET_TYPES, default='rpa-us', blank=True, null=True)
    input_filename = models.CharField(max_length=50, default='log.csv')
    configurations = JSONField(null=True, blank=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    ui_elements_classification = models.ForeignKey('UIElementsClassification', on_delete=models.SET_NULL, null=True)

    # Delete ui_elements_classification when deleting UIElementsDetection
    def delete_ui_elements_classification(self, *args, **kwargs):
        self.ui_elements_classification.delete()
        super().delete(*args, **kwargs)

    def get_absolute_url(self):
        return reverse("featureextraction:ui_detection_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type + '- active? ' + str(self.active)
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.type == "screen2som" and not self.ui_elements_classification:
            ui_elements_classification = UIElementsClassification.objects.create(
                preloaded=self.preloaded,
                preloaded_file=self.preloaded_file,
                freeze=self.freeze,
                active=self.active,
                executed=self.executed,
                model=CNNModels.objects.get_or_create(name="screen2som", path="NA", image_shape=[640, 360, 3], classes=get_ui_elements_classification_screen2som(), text_classname="Text")[0],
                type=self.type,
                skip=self.skip,
                case_study=self.case_study,
                user=self.user
            )

            self.ui_elements_classification = ui_elements_classification

            super().save(*args, **kwargs)

@receiver(pre_delete, sender=UIElementsDetection)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)

def get_ui_elements_classification_image_shape():
    return [64, 64, 3]

def get_ui_elements_classification_screen2som():
    classes_json = list(json.load(open(SCREEN2SOM_CONFIG_PATH))["classes"].keys())
    return  classes_json    

def get_ui_elements_classification_moran():
    return 'x0_Button,x0_CheckBox,x0_CheckedTextView,x0_EditText,x0_ImageButton,x0_ImageView,x0_NumberPicker,x0_RadioButton,x0_RatingBar,x0_SeekBar,x0_Spinner,x0_Switch,x0_TextView,x0_ToggleButton'.split(',') # this returns a list

def get_ui_elements_classification_uied():
    return "Button,Checkbox,CheckedTextView,EditText,ImageButton,ImageView,NumberPicker,RadioButton,RatingBar,SeekBar,Spinner,Switch,TextView,ToggleButton".split(',') # this returns a list

class CNNModels(models.Model):
    name = models.CharField(max_length=25)
    path = models.CharField(max_length=255, default="checkpoints/uied/custom-v2.h5")
    image_shape = ArrayField(models.IntegerField(blank=True), default=get_ui_elements_classification_image_shape)
    classes = ArrayField(models.CharField(max_length=50), default=get_ui_elements_classification_uied)
    text_classname = models.CharField(max_length=50, default="TextView")
    model_properties = models.JSONField(null=True, blank=True)

    # Unique name + classes
    class Meta:
        unique_together = ('name', 'classes')
   
    def clean(self):
        if (self.text_classname not in self.classes):
            raise ValidationError("text_classname must be one of the ui_elements_classification_classes")

    def __str__(self):
        return self.name

class UIElementsClassification(models.Model):
    preloaded = models.BooleanField(default=False, editable=False)
    preloaded_file = PrivateFileField("File", null=True)
    freeze = models.BooleanField(default=False, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    model = models.ForeignKey('CNNModels', on_delete=models.CASCADE, null=True)
    type = models.CharField(max_length=25, default='rpa-us', blank=True, null=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("featureextraction:ui_classification_list", args=[str(self.case_study_id)])
        
    def __str__(self):
        return 'type: ' + self.type

@receiver(pre_delete, sender=UIElementsClassification)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)

class Postfilters(models.Model):
    preloaded = models.BooleanField(default=False, editable=True)
    preloaded_file = PrivateFileField("File", null=True,blank=True)
    title = models.CharField(max_length=255)
    freeze = models.BooleanField(default=False, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    # scale_factor = models.IntegerField(default=5)
    # intersection_area_thresh = models.IntegerField(default=0)
    consider_nested_as_relevant = models.BooleanField(default=True)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return reverse("featureextraction:postfilters_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.title + ' - active? ' + str(self.active)
    
@receiver(pre_delete, sender=Postfilters)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)

class FeatureExtractionTechnique(models.Model):
    preloaded = models.BooleanField(default=False, editable=True)
    preloaded_file = PrivateFileField("File", null=True, blank=True)
    title = models.CharField(max_length=255)
    freeze = models.BooleanField(default=False, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    # quiero validar que solo contenga minusculas y mayusculas, ni numeros ni caracteres especiales
    identifier = models.CharField(max_length=25, default='rpa-us', null=True, blank=True, validators=[RegexValidator(r'^[a-zA-Z-]*$', _('Only lowercase, uppercase letters, and hyphen are allowed'))])
    type = models.CharField(max_length=255, null=True)
    technique_name = models.CharField(max_length=255, null=True)
    relevant_compos_predicate = models.CharField(max_length=255, null=True, blank=True)
    consider_relevant_compos = models.BooleanField(default=False)
    configurations = JSONField(null=True, blank=True)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data
    
    def get_absolute_url(self):
        return reverse("featureextraction:fe_technique_list", args=[str(self.case_study_id)])
        
    def __str__(self):
        return 'technique: ' + self.technique_name + ' - active? ' + str(self.active)

@receiver(pre_delete, sender=FeatureExtractionTechnique)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)

class Postprocessing(models.Model):
    """
    Postprocessing techniques to enriched log after process discovery.
    
    Ej. Normalize centroids based on image corresponding activity.
    """
    preloaded = models.BooleanField(default=False, editable=True)
    preloaded_file = PrivateFileField("File", null=True, blank=True)
    title = models.CharField(max_length=255)
    freeze = models.BooleanField(default=False, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    technique_name = models.CharField(max_length=255, null=True)

    def get_absolute_url(self):
        return reverse("featureextraction:postprocessing_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.title + ' - active? ' + str(self.active)

@receiver(pre_delete, sender=Postprocessing)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)