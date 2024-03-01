import os
import subprocess
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
from core.settings import PRIVATE_STORAGE_ROOT, sep, DEFAULT_EXECUTION_ATTRIBUTES_PHASES
from apps.processdiscovery.models import ProcessDiscovery
from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining
from apps.featureextraction.models import Prefilters, UIElementsDetection, UIElementsClassification, Postfilters, FeatureExtractionTechnique
from apps.behaviourmonitoring.models import Monitoring
from apps.reporting.models import PDD
from django.utils.translation import gettext_lazy as _

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
         raise ValidationError(_("exp_folder_complete_path separators not coherent"))
    splitted_s = exp_folder_complete_path.split(aux)
    return splitted_s[len(splitted_s) - 1]

class CaseStudy(models.Model):
    title = models.CharField(max_length=255)
    description = models.CharField(max_length=255, default=_("This is a nice experiment..."))
    executed = models.IntegerField(default=0, editable=True)
    active = models.BooleanField(default=True, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    exp_file = PrivateFileField("File", null=True)
    exp_foldername = models.CharField(max_length=255, null=True, blank=True)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True, blank=True)
    special_colnames = JSONField(default=default_special_colnames)
    phases_to_execute = JSONField(null=True, blank=True)
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
        verbose_name = _("Case study")
        verbose_name_plural = _("Case studies")

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
            raise ValidationError(_('The title of the case study already exists'))

    def any_active(self):
        """Returns True if any of the phases is active"""
        monitoring = Monitoring.objects.filter(case_study=self, active=True).first()
        prefilters = Prefilters.objects.filter(case_study=self, active=True).first()
        ui_elements_detection = UIElementsDetection.objects.filter(case_study=self, active=True).first()
        ui_elements_classification = UIElementsClassification.objects.filter(case_study=self, active=True).first()
        postfilters = Postfilters.objects.filter(case_study=self, active=True).first()
        feature_extraction_technique = FeatureExtractionTechnique.objects.filter(case_study=self, active=True).first()
        process_discovery = ProcessDiscovery.objects.filter(case_study=self, active=True).first()
        extract_training_dataset = ExtractTrainingDataset.objects.filter(case_study=self, active=True).first()
        decision_tree_training = DecisionTreeTraining.objects.filter(case_study=self, active=True).first()

        return any([monitoring, prefilters, ui_elements_detection, ui_elements_classification, postfilters,
                    feature_extraction_technique, process_discovery, extract_training_dataset, decision_tree_training])
    
    def __str__(self):
        return self.title + ' - id:' + str(self.id)


class Execution(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='executions')
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, related_name='executions')
    created_at = models.DateTimeField(auto_now_add=True)
    executed = models.IntegerField(default=0, editable=True)
    exp_foldername = models.CharField(max_length=255, null=True, blank=True)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True, blank=True)
    monitoring = models.ForeignKey(Monitoring, null=True, blank=True, on_delete=models.CASCADE)
    prefilters = models.ForeignKey(Prefilters, null=True, blank=True, on_delete=models.CASCADE)
    ui_elements_detection = models.ForeignKey(UIElementsDetection, null=True, blank=True, on_delete=models.CASCADE)
    ui_elements_classification = models.ForeignKey(UIElementsClassification, null=True, blank=True, on_delete=models.CASCADE)
    postfilters = models.ForeignKey(Postfilters, null=True, blank=True, on_delete=models.CASCADE)
    feature_extraction_technique = models.ForeignKey(FeatureExtractionTechnique, null=True, blank=True, on_delete=models.CASCADE)
    process_discovery = models.ForeignKey(ProcessDiscovery, null=True, blank=True, on_delete=models.CASCADE)
    extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, blank=True, on_delete=models.CASCADE)
    decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, blank=True, on_delete=models.CASCADE)
    
    # Check phases dependencies and restrictions
    # To execute feature extraction, UIElementsDetection and UIElementsClassification must be executed
    # To execute decision tree training, Decision Tree training must be executed
    def clean(self):
        # Check if at least one phase is executed
        if not (self.monitoring or self.prefilters or self.ui_elements_detection or self.ui_elements_classification or self.postfilters or self.feature_extraction_technique or self.process_discovery or self.extract_training_dataset or self.decision_tree_training):
            raise ValidationError('At least one phase must be executed.')
        
        # Check phases dependencies and restrictions
        if self.feature_extraction_technique and not (self.ui_elements_detection and self.ui_elements_classification):
            raise ValidationError('UI Elements Detection  and Classification  must be executed before Feature Extraction.')
        if self.decision_tree_training and not self.extract_training_dataset:
            raise ValidationError('Extract Training Dataset must be executed before Decision Tree Training.')

    def save(self, *args, **kwargs):
        # Retrieve active configurations and set them to the execution
        self.monitoring = Monitoring.objects.filter(case_study=self.case_study, active=True).first()
        self.prefilters = Prefilters.objects.filter(case_study=self.case_study, active=True).first()
        self.ui_elements_detection = UIElementsDetection.objects.filter(case_study=self.case_study, active=True).first()
        self.ui_elements_classification = UIElementsClassification.objects.filter(case_study=self.case_study, active=True).first()
        self.postfilters = Postfilters.objects.filter(case_study=self.case_study, active=True).first()
        self.feature_extraction_technique = FeatureExtractionTechnique.objects.filter(case_study=self.case_study, active=True).first()
        self.process_discovery = ProcessDiscovery.objects.filter(case_study=self.case_study, active=True).first()
        self.extract_training_dataset = ExtractTrainingDataset.objects.filter(case_study=self.case_study, active=True).first()
        self.decision_tree_training = DecisionTreeTraining.objects.filter(case_study=self.case_study, active=True).first()

        self.scenarios_to_study = self.case_study.scenarios_to_study

        self.clean()

        for stage in [self.monitoring, self.prefilters, self.ui_elements_detection,
                      self.ui_elements_classification, self.postfilters, self.feature_extraction_technique, 
                      self.process_discovery, self.extract_training_dataset, self.decision_tree_training]:
            if stage:
                stage.freeze = True
                stage.save()

        super().save(*args, **kwargs)

        if not self.exp_folder_complete_path or self.exp_folder_complete_path == '':
            self.create_folder_structure()
        
        super().save(*args, **kwargs)

    def check_preloaded_file(self):            
        for ph in DEFAULT_EXECUTION_ATTRIBUTES_PHASES:
            if hasattr(self, ph) and hasattr(getattr(self, ph), "preloaded") and getattr(self, ph).preloaded:
                preloaded_file_path = f"{PRIVATE_STORAGE_ROOT}{sep}{getattr(self, ph).preloaded_file.name}"
                unzip_file(preloaded_file_path, self.exp_folder_complete_path)
                print("Preloaded file unzipped!:", self.exp_folder_complete_path)

    
    def create_folder_structure(self):
        self.exp_foldername = f"exec_{self.id}" #exec_1
        self.exp_folder_complete_path = os.path.join(self.case_study.exp_folder_complete_path, 'executions', self.exp_foldername)
        #output: media/unzipped/zip_1321321/executions/exec_1
        
        #FOR ONLY ONE SCENARIO IN YOUR CASE STUDY
        #If the phases own the preloaded_file, the preloaded_file wil be unzipped from media/ and it will be stored
        #in the corresponding execution_id folder 
    

        if not os.path.exists(self.exp_folder_complete_path):
            os.makedirs(self.exp_folder_complete_path)

        # Create a symbolic link to the case study scenarios to study inside the execution folder
        for scenario in self.scenarios_to_study:

            # Os Simlink only works for files in windows
            if os.name == 'nt':
                subprocess.call(['cmd', '/c', 'mklink', '/D', os.path.join(self.exp_folder_complete_path, scenario), os.path.join('..\\..\\', scenario)])
            else:
                os.symlink(
                    os.path.join('../../', scenario),
                    os.path.join(self.exp_folder_complete_path, scenario)
                    )