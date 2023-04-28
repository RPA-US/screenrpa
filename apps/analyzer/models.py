from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.exceptions import ValidationError
from private_storage.fields import PrivateFileField
from apps.processdiscovery.models import ProcessDiscovery
from apps.decisiondiscovery.models import ExtractTrainingDataset,DecisionTreeTraining
from apps.featureextraction.models import Prefilters, UIElementsDetection, UIElementsClassification, Filters, FeatureExtractionTechnique
from apps.behaviourmonitoring.models import Monitoring
from apps.reporting.models import PDD

def get_ui_elements_classification_image_shape():
    return [64, 64, 3]

def default_special_colnames():
    return dict({
        "Case": "Case",
        "Activity": "Activity",
        "Screenshot": "Screenshot", 
        "Variant": "Variant",
        "Timestamp": "Timestamp",
        "eyetracking_recording_timestamp": "Recording timestamp",
        "eyetracking_gaze_point_x": "Gaze point X",
        "eyetracking_gaze_point_y": "Gaze point Y"
    })

def get_ui_elements_classification_old_classes():
    return 'x0_Button, x0_CheckBox, x0_CheckedTextView, x0_EditText, x0_ImageButton, x0_ImageView, x0_NumberPicker, x0_RadioButton', 
'x0_RatingBar, x0_SeekBar, x0_Spinner, x0_Switch, x0_TextView, x0_ToggleButton'.split(', ') # this returns a list

def get_ui_elements_classification_classes():
    return "Button, Checkbox, CheckedTextView, EditText, ImageButton, ImageView, NumberPicker, RadioButton, RatingBar, SeekBar, Spinner, Switch, TextView, ToggleButton".split(', ') # this returns a list

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
         raise ValidationError("exp_folder_complete_path separators not coherent")
    splitted_s = exp_folder_complete_path.split(aux)
    return splitted_s[len(splitted_s) - 1]

class CaseStudy(models.Model):
    title = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    executed = models.IntegerField(default=0, editable=True)
    active = models.BooleanField(default=True, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    exp_files = PrivateFileField("File")
    exp_foldername = models.CharField(max_length=255, null=True, blank=True)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True, blank=True)
    special_colnames = JSONField(default=default_special_colnames)
    text_classname = models.CharField(max_length=50)
    phases_to_execute = JSONField(null=True, blank=True)
    decision_point_activity = models.CharField(max_length=255)
    gui_class_success_regex = models.CharField(max_length=255, default="CheckBox_4_D or ImageView_4_D or TextView_4_D")
    ui_elements_classification_image_shape = ArrayField(models.IntegerField(null=True, blank=True), default=get_ui_elements_classification_image_shape)
    ui_elements_classification_classes = ArrayField(models.CharField(max_length=50), default=get_ui_elements_classification_classes)
    target_label = models.CharField(max_length=50, default='Variant')
    monitoring = models.ForeignKey(Monitoring, null=True, blank=True, on_delete=models.CASCADE)
    prefilters = models.ForeignKey(Prefilters, null=True, blank=True, on_delete=models.CASCADE)
    ui_elements_detection = models.ForeignKey(UIElementsDetection, null=True, blank=True, on_delete=models.CASCADE)
    ui_elements_classification = models.ForeignKey(UIElementsClassification, null=True, blank=True, on_delete=models.CASCADE)
    filters = models.ForeignKey(Filters, null=True, blank=True, on_delete=models.CASCADE)
    feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, blank=True, on_delete=models.CASCADE)
    process_discovery = models.ForeignKey(ProcessDiscovery, null=True, blank=True, on_delete=models.CASCADE)
    extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, blank=True, on_delete=models.CASCADE)
    decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, blank=True, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='CaseStudyExecuter')

    class Meta:
        verbose_name = "Case study"
        verbose_name_plural = "Case studies"

    def get_absolute_url(self):
        return reverse("home")

    def create(self, validated_data):
        CaseStudy.term_unique(self, validated_data.get("title"))
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        validated_data.update({"user": self.request.user})
        exp_fol = get_exp_foldername(validated_data.pop("exp_folder_complete_path", None))
        # items = validated_data.pop("formats_supported", None)
        case_study = CaseStudy.objects.create(**validated_data)
        case_study.exp_foldername = exp_fol
        # if items is not None:
        #     # items = [InputFormatSupported.objects.create(**item) for item in items]
        #     # '*' is the "splat" operator: It takes a list as input, and expands it into actual positional arguments in the function call.
        #     action.formats_supported.add(*items)
        return case_study
    
    def term_unique(self, title):
        if CaseStudy.objects.filter(term=title).exists():
            raise ValidationError('The title of the case study already exists')

    
    def __str__(self):
        return self.title + ' - id:' + str(self.id)
    
