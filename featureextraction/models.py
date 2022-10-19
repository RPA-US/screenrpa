from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField

def default_phases_to_execute():
    return {'ui_elements_detection': {}, 'ui_elements_classification': {}, 'extract_training_dataset': {}, 'decision_tree_training': {}}

# Create your models here.

def get_default_extract_training_columns_to_ignore():
    return 'Coor_X, Coor_Y, Case'.split(', ') # this returns a list

def get_default_decision_tree_columns_to_ignore():
    return 'Timestamp_start, Timestamp_end'.split(', ') # this returns a list

def get_ui_elements_classification_classes():
    return 'x0_Button, x0_CheckBox, x0_CheckedTextView, x0_EditText, x0_ImageButton, x0_ImageView, x0_NumberPicker, x0_RadioButton', 
'x0_RatingBar, x0_SeekBar, x0_Spinner, x0_Switch, x0_TextView, x0_ToggleButton'.split(', ') # this returns a list
    
def get_default_algorithms():
    return 'ID3, CART, CHAID, C4.5'.split(', ') # this returns a list

class UIElementsDetection(models.Model):
    eyetracking_log_filename =  models.CharField(max_length=255, default="eyetracking_log.csv")
    add_words_columns = models.BooleanField(default=False)
    overwrite_npy = models.BooleanField(default=False)
    algorithm = models.CharField(max_length=25, default='legacy')

class UIElementsClassification(models.Model):
    model_weights = models.CharField(max_length=255, default="resources/models/custom-v2.h5")
    model_properties = models.CharField(max_length=255, default="resources/models/custom-v2-classes.json")
    overwrite_npy = models.BooleanField(default=False)
    ui_elements_classification_classes = ArrayField(models.CharField(max_length=50), default=get_ui_elements_classification_classes)
    classifier = models.CharField(max_length=25, default='legacy')

class FeatureExtractionTechnique(models.Model):
    name = models.CharField(max_length=255, default='count')
    overwrite_npy = models.BooleanField(default=False)
