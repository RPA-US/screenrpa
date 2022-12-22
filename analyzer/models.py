from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from decisiondiscovery.models import ExtractTrainingDataset,DecisionTreeTraining
from featureextraction.models import UIElementsDetection, UIElementsClassification, FeatureExtractionTechnique, NoiseFiltering

def default_phases_to_execute():
    return {'ui_elements_detection': {}, 'ui_elements_classification': {}, 'extract_training_dataset': {}, 'decision_tree_training': {}}

def get_ui_elements_classification_image_shape():
    return [64, 64, 3]

def get_ui_elements_classification_classes():
    return 'x0_Button, x0_CheckBox, x0_CheckedTextView, x0_EditText, x0_ImageButton, x0_ImageView, x0_NumberPicker, x0_RadioButton', 
'x0_RatingBar, x0_SeekBar, x0_Spinner, x0_Switch, x0_TextView, x0_ToggleButton'.split(', ') # this returns a list


class CaseStudy(models.Model):
    title = models.CharField(max_length=255)
    # priority = models.IntegerField(default=1, editable=False)
    executed = models.BooleanField(default=False, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    exp_foldername = models.CharField(max_length=255)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True)
    special_colnames = JSONField()
    text_classname = models.CharField(max_length=50)
    # phases_to_execute = JSONField()
    decision_point_activity = models.CharField(max_length=255)
    gui_class_success_regex = models.CharField(max_length=255)
    ui_elements_classification_image_shape = ArrayField(models.IntegerField(null=True, blank=True), default=get_ui_elements_classification_image_shape)
    ui_elements_classification_classes = ArrayField(models.CharField(max_length=50), default=get_ui_elements_classification_classes)
    target_label = models.CharField(max_length=50, default='Variant')
    ui_elements_detection = models.ForeignKey(UIElementsDetection, null=True, on_delete=models.CASCADE)
    noise_filtering = models.ForeignKey(NoiseFiltering, null=True, on_delete=models.CASCADE)
    ui_elements_classification = models.ForeignKey(UIElementsClassification, null=True, on_delete=models.CASCADE)
    feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, on_delete=models.CASCADE)
    extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, on_delete=models.CASCADE)
    decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, on_delete=models.CASCADE)
    # user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='CaseStudyExecuter')

    def __str__(self):
        return self.title