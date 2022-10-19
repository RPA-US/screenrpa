from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from decisiondiscovery.models import ExtractTrainingDataset,DecisionTreeTraining
from featureextraction.models import UIElementsDetection, UIElementsClassification, FeatureExtractionTechnique

def default_phases_to_execute():
    return {'ui_elements_detection': {}, 'ui_elements_classification': {}, 'extract_training_dataset': {}, 'decision_tree_training': {}}

# Create your models here.

class CaseStudy(models.Model):
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    exp_foldername = models.CharField(max_length=255)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True)
    drop = models.CharField(max_length=255, null=True)
    special_colnames = JSONField()
    # phases_to_execute = JSONField()
    decision_point_activity = models.CharField(max_length=255)
    gui_class_success_regex = models.CharField(max_length=255)
    gui_quantity_difference = models.IntegerField(default=1)
    ui_elements_detection = models.ForeignKey(UIElementsDetection, null=True, on_delete=models.CASCADE)
    ui_elements_classification = models.ForeignKey(UIElementsClassification, null=True, on_delete=models.CASCADE)
    feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, on_delete=models.CASCADE)
    extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, on_delete=models.CASCADE)
    decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, on_delete=models.CASCADE)

    # user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='CaseStudyExecuter')
    
    def __str__(self):
        return self.title