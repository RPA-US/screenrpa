from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField

def default_phases_to_execute():
    return {'gui_components_detection': {}, 'classify_image_components': {}, 'extract_training_dataset': {}, 'decision_tree_training': {}}

# Create your models here.

def get_default_extract_training_columns_to_ignore():
    return 'Coor_X, Coor_Y, Case'.split(', ') # this returns a list

def get_default_decision_tree_columns_to_ignore():
    return 'Timestamp_start, Timestamp_end'.split(', ') # this returns a list

def get_default_algorithms():
    return 'ID3, CART, CHAID, C4.5'.split(', ') # this returns a list

class GUIComponentDetection(models.Model):
    eyetracking_log_filename =  models.CharField(max_length=255, default="eyetracking_log.csv")
    add_words_columns = models.BooleanField(default=False)
    overwrite_npy = models.BooleanField(default=False)
    algorithm = models.CharField(max_length=25, default='legacy')

class ClassifyImageComponents(models.Model):
    model_json_file_name = models.CharField(max_length=255, blank=True, default="resources/models/model.json")
    model_weights = models.CharField(max_length=255, default="resources/models/custom-v2.h5")
    model_properties = models.CharField(max_length=255, default="resources/models/custom-v2-classes.json")
    overwrite_npy = models.BooleanField(default=False)
    algorithm = models.CharField(max_length=25, default='legacy')

class FeatureExtractionTechnique(models.Model):
    name = models.CharField(max_length=255, default='count')
    overwrite_npy = models.BooleanField(default=False)

class ExtractTrainingDataset(models.Model):
    columns_to_ignore = ArrayField(models.CharField(max_length=25), default=get_default_extract_training_columns_to_ignore)

class DecisionTreeTraining(models.Model):
    library = models.CharField(max_length=255, default='chefboost') # 'sklearn'
    algorithms = ArrayField(models.CharField(max_length=25), default=get_default_algorithms)
    mode = models.CharField(max_length=25, default='autogeneration')
    columns_to_ignore = ArrayField(models.CharField(max_length=50), default=get_default_decision_tree_columns_to_ignore)

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
    gui_components_detection = models.ForeignKey(GUIComponentDetection, null=True, on_delete=models.CASCADE)
    classify_image_components = models.ForeignKey(ClassifyImageComponents, null=True, on_delete=models.CASCADE)
    feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, on_delete=models.CASCADE)
    extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, on_delete=models.CASCADE)
    decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, on_delete=models.CASCADE)

    # user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='CaseStudyExecuter')
    
    def __str__(self):
        return self.title