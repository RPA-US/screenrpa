import os
import zipfile
import time
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.exceptions import ValidationError
from private_storage.fields import PrivateFileField
from core.settings import PRIVATE_STORAGE_ROOT, sep
# from apps.processdiscovery.models import ProcessDiscovery
# from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining
# from apps.featureextraction.models import Prefilters, UIElementsDetection, UIElementsClassification, Postfilters
# from apps.behaviourmonitoring.models import Monitoring
# from apps.reporting.models import PDD

def unzip_file(zip_file_path, dest_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder_path)


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
    description = models.CharField(max_length=255, default="This is a nice experiment...")
    executed = models.IntegerField(default=0, editable=True)
    active = models.BooleanField(default=True, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    exp_file = PrivateFileField("File", null=True)
    exp_foldername = models.CharField(max_length=255, null=True, blank=True)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True, blank=True)
    special_colnames = JSONField(default=default_special_colnames)
    phases_to_execute = JSONField(null=True, blank=True)
    gui_class_success_regex = models.CharField(max_length=255, default="CheckBox_4_D or ImageView_4_D or TextView_4_D")
    target_label = models.CharField(max_length=50, default='Variant')
    # monitoring = models.ForeignKey(Monitoring, null=True, blank=True, on_delete=models.CASCADE)
    # prefilters = models.ForeignKey(Prefilters, null=True, blank=True, on_delete=models.CASCADE)
    # ui_elements_detection = models.ForeignKey(UIElementsDetection, null=True, blank=True, on_delete=models.CASCADE)
    # ui_elements_classification = models.ForeignKey(UIElementsClassification, null=True, blank=True, on_delete=models.CASCADE)
    # postfilters = models.ForeignKey(Postfilters, null=True, blank=True, on_delete=models.CASCADE)
    # feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, blank=True, on_delete=models.CASCADE)
    # process_discovery = models.ForeignKey(ProcessDiscovery, null=True, blank=True, on_delete=models.CASCADE)
    # extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, blank=True, on_delete=models.CASCADE)
    # decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, blank=True, on_delete=models.CASCADE)
    # report = models.ForeignKey(PDD, null=True, blank=True, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='CaseStudyExecuter')

    class Meta:
        verbose_name = "Case study"
        verbose_name_plural = "Case studies"

    def get_absolute_url(self):
        return reverse("home")
    

    # def create(self, validated_data):
    #     CaseStudy.term_unique(self, validated_data.get("title"))
    #     if not self.request.user.is_authenticated:
    #         raise ValidationError("User must be authenticated.")
    #     validated_data.update({"user": self.request.user})
    #     exp_fol = get_exp_foldername(validated_data.pop("exp_folder_complete_path", None))
    #     case_study = CaseStudy.objects.create(**validated_data)
    #     case_study.exp_foldername = exp_fol
        
    #     return case_study

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.exp_file:
            # Generate unique folder name based on the uploaded file's name and current time
            folder_name = f"{self.exp_file.name.split('.')[0]}_{str(int(time.time()))}"
            # folder_path = os.path.join(PRIVATE_STORAGE_ROOT, 'unzipped', folder_name)
            folder_path = PRIVATE_STORAGE_ROOT + sep + 'unzipped' + sep + folder_name
            # Create the unzipped folder
            os.makedirs(folder_path)
            # Unzip the uploaded file to the unzipped folder
            unzip_file(self.exp_file.path, folder_path)
            # Save the unzipped folder path to the model instance
            self.exp_folder_complete_path = folder_path
            self.exp_foldername = get_exp_foldername(folder_path)
            super().save(*args, **kwargs)
            
    
    def term_unique(self, title):
        if CaseStudy.objects.filter(term=title).exists():
            raise ValidationError('The title of the case study already exists')

    
    def __str__(self):
        return self.title + ' - id:' + str(self.id)



# TODO: Implement execution model
# class Execution(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='CaseStudyExecuter')
#     case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, related_name='CaseStudy')
#     monitoring = models.ForeignKey(Monitoring, null=True, blank=True, on_delete=models.CASCADE)
#     prefilters = models.ForeignKey(Prefilters, null=True, blank=True, on_delete=models.CASCADE)
#     ui_elements_detection = models.ForeignKey(UIElementsDetection, null=True, blank=True, on_delete=models.CASCADE)
#     ui_elements_classification = models.ForeignKey(UIElementsClassification, null=True, blank=True, on_delete=models.CASCADE)
#     postfilters = models.ForeignKey(Postfilters, null=True, blank=True, on_delete=models.CASCADE)
#     feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, blank=True, on_delete=models.CASCADE)
#     process_discovery = models.ForeignKey(ProcessDiscovery, null=True, blank=True, on_delete=models.CASCADE)
#     extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, blank=True, on_delete=models.CASCADE)
#     decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, blank=True, on_delete=models.CASCADE)
