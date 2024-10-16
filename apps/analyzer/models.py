import os
import subprocess
import zipfile
import time
import shutil
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.db.models import JSONField
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.exceptions import ValidationError
from private_storage.fields import PrivateFileField
from core.settings import PRIVATE_STORAGE_ROOT, DEFAULT_PHASES
from apps.processdiscovery.models import ProcessDiscovery
from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining
from apps.featureextraction.models import Prefilters, UIElementsDetection, UIElementsClassification, Postfilters, FeatureExtractionTechnique, Postprocessing
from apps.behaviourmonitoring.models import Monitoring
from apps.reporting.models import PDD
from django.utils.translation import gettext_lazy as _

def unzip_file(zip_file_path, dest_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder_path)
        
def unzip_file_here(zip_file_path, dest_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue

            # copy file (taken from zipfile's extract)
            source = zip_ref.open(member)
            target = open(os.path.join(dest_folder_path, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)


def default_special_colnames():
    return dict(
        
        {
        "Case": "trace_id",
        "Activity": "activity_id",
        "Screenshot": "Screenshot", 
        "Variant": "auto_variant",
        "Timestamp": "Timestamp",
        "NameApp": "NameApp",
        "EventType": "MorKeyb",
        "CoorX": "Coor_X",
        "CoorY": "Coor_Y",
        "Header": "header"
        }
    )

    
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
    exp_file = PrivateFileField("File", null=True, editable=True)
    exp_foldername = models.CharField(max_length=255, null=True, blank=True, editable=True)
    exp_folder_complete_path = models.CharField(max_length=255)
    scenarios_to_study = ArrayField(models.CharField(max_length=100), null=True, blank=True) # example: sc_0_size50_Balanced,sc_0_size50_Imbalanced,sc_0_size75_Balanced,sc_0_size75_Imbalanced,sc_0_size100_Balanced,sc_0_size100_Imbalanced,sc_0_size300_Balanced,sc_0_size300_Imbalanced,sc_0_size500_Balanced,sc_0_size500_Imbalanced
    special_colnames = JSONField(default=default_special_colnames)
    phases_to_execute = JSONField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='CaseStudyExecuter')

    @property
    def num_executions(self):
        return Execution.objects.filter(case_study=self).count()
    
    @property
    def available_phases(self):
        """
        Returns the phases that can be configured based on the current active configurations
        """
        available_phases = ['Monitoring']
        # If there exists a log.csv in the unzipped folder or there exists a monitoring configutation, phases can be configured
        exists_log_csvs_paths = [os.path.exists(os.path.join(self.exp_folder_complete_path, scenario, 'log.csv')) for scenario in self.scenarios_to_study]
        if all(exists_log_csvs_paths) or Monitoring.objects.filter(case_study=self, active=True).exists():
            if not Postfilters.objects.filter(case_study=self, active=True).exists() and Monitoring.objects.filter(case_study=self, active=True).exists():
                available_phases.append('Prefilters')
            available_phases.append('UIElementsDetection')
            if UIElementsDetection.objects.filter(case_study=self, active=True).exists():
                available_phases.append("FeatureExtractionTechnique")
                if not Prefilters.objects.filter(case_study=self, active=True).exists() and Monitoring.objects.filter(case_study=self, active=True).exists():
                    available_phases.append('Postfilters')
            available_phases.append("ProcessDiscovery")
            if FeatureExtractionTechnique.objects.filter(case_study=self, active=True).exists() and ProcessDiscovery.objects.filter(case_study=self, active=True).exists():
                available_phases.append('Postprocessing')
                available_phases.append('ExtractTrainingDataset')
            if ExtractTrainingDataset.objects.filter(case_study=self, active=True).exists():
                available_phases.append("DecisionTreeTraining")

        return available_phases
    
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
            folder_path = os.path.join(PRIVATE_STORAGE_ROOT, 'unzipped', folder_name)
            # Create the unzipped folder
            os.makedirs(folder_path)
            # Unzip the uploaded file to the unzipped folder
            unzip_file(self.exp_file.path, folder_path)
            # Save the unzipped folder path to the model instance
            self.exp_folder_complete_path = folder_path
            self.exp_foldername = get_exp_foldername(folder_path)
            super().save(*args, **kwargs)

    def delete(self):
        # Delete the zip and unzipped folder
        self.exp_file.delete()
        shutil.rmtree(self.exp_folder_complete_path)
        super().delete()    
    
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
        postprocessing = Postprocessing.objects.filter(case_study=self, active=True).first()
        process_discovery = ProcessDiscovery.objects.filter(case_study=self, active=True).first()
        extract_training_dataset = ExtractTrainingDataset.objects.filter(case_study=self, active=True).first()
        decision_tree_training = DecisionTreeTraining.objects.filter(case_study=self, active=True).first()

        return any([monitoring, prefilters, ui_elements_detection, ui_elements_classification, postfilters,
                    feature_extraction_technique, process_discovery, postprocessing, extract_training_dataset, decision_tree_training])
    
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
    feature_extraction_techniques = models.ManyToManyField(FeatureExtractionTechnique, related_name='executions')
    process_discovery = models.ForeignKey(ProcessDiscovery, null=True, blank=True, on_delete=models.CASCADE)
    postprocessings = models.ManyToManyField(Postprocessing, related_name='executions')

    extract_training_dataset = models.ForeignKey(ExtractTrainingDataset, null=True, blank=True, on_delete=models.CASCADE)
    decision_tree_training = models.ForeignKey(DecisionTreeTraining, null=True, blank=True, on_delete=models.CASCADE)
    
    errored = models.BooleanField(default=False)
  
    @property
    def feature_extraction_technique(self):
        """
        Returns true if there is any feature extraction technique activated
        This is done for the purpose of maintaining compatibility with the previous version of the application where every phase is checked to exist with getatrr()
        """
        return self.feature_extraction_techniques.exists()

    @property
    def postprocessing(self):
        """
        Returns true if there is any postprocessing technique activated
        This is done for the purpose of maintaining compatibility with the previous version of the application where every phase is checked to exist with getatrr()
        """
        return self.postprocessings.exists()

    def clean(self):
        # Check if at least one phase is executed
        if not (self.monitoring or self.prefilters or self.ui_elements_detection or self.ui_elements_classification or self.postfilters or self.feature_extraction_technique or self.process_discovery or self.extract_training_dataset or self.decision_tree_training):
            raise ValidationError('At least one phase must be executed.')

        # Check phases dependencies and restrictions
        if self.feature_extraction_techniques.exists() and not self.ui_elements_detection:
            raise ValidationError('UI Elements Detection must be executed before Feature Extraction.')
        if self.decision_tree_training and not self.extract_training_dataset:
            raise ValidationError('Extract Training Dataset must be executed before Decision Tree Training.')

        if self.scenarios_to_study is None or len(self.scenarios_to_study) == 0:
            raise ValidationError('At least one scenario must be indicated.')

    def save(self, *args, **kwargs):
        # Retrieve active configurations and set them to the execution
        self.monitoring = Monitoring.objects.filter(case_study=self.case_study, active=True).first()
        self.prefilters = Prefilters.objects.filter(case_study=self.case_study, active=True).first()
        self.ui_elements_detection = UIElementsDetection.objects.filter(case_study=self.case_study, active=True).first()
        self.ui_elements_classification = UIElementsClassification.objects.filter(case_study=self.case_study, active=True).first()
        self.postfilters = Postfilters.objects.filter(case_study=self.case_study, active=True).first()
        self.process_discovery = ProcessDiscovery.objects.filter(case_study=self.case_study, active=True).first()
        self.extract_training_dataset = ExtractTrainingDataset.objects.filter(case_study=self.case_study, active=True).first()
        self.decision_tree_training = DecisionTreeTraining.objects.filter(case_study=self.case_study, active=True).first()

        self.scenarios_to_study = self.case_study.scenarios_to_study

        super().save(*args, **kwargs)

        active_feature_extraction_techniques = FeatureExtractionTechnique.objects.filter(case_study=self.case_study, active=True)
        self.feature_extraction_techniques.set(active_feature_extraction_techniques)

        active_postprocessings = Postprocessing.objects.filter(case_study=self.case_study, active=True)
        self.postprocessings.set(active_postprocessings)
        
        self.clean()

        for stage in [self.monitoring, self.prefilters, self.ui_elements_detection,
                      self.ui_elements_classification, self.postfilters, 
                      self.process_discovery, self.extract_training_dataset, self.decision_tree_training]:
            if stage:
                stage.executed += 1
                stage.freeze = True
                stage.save()
                
        for stage in self.feature_extraction_techniques.all():
            stage.executed += 1
            stage.freeze = True
            stage.save()
        
        for stage in self.postprocessings.all():
            stage.executed += 1
            stage.freeze = True
            stage.save()

        if not self.exp_folder_complete_path or self.exp_folder_complete_path == '':
            self.create_folder_structure()

        super().save(*args, **kwargs)

    def check_preloaded_file(self):            
        for ph in DEFAULT_PHASES:
            if hasattr(self, ph) and hasattr(getattr(self, ph), "preloaded") and getattr(self, ph).preloaded and hasattr(getattr(self,ph),"active") and getattr(self,ph).active == True:
                preloaded_file_path = os.path.join(PRIVATE_STORAGE_ROOT, getattr(self, ph).preloaded_file.name)
                unzip_file(preloaded_file_path, self.exp_folder_complete_path)
                print("Preloaded file unzipped!:", self.exp_folder_complete_path)
                
        for fe in self.feature_extraction_techniques.all():
            if hasattr(fe, "preloaded") and fe.preloaded:
                preloaded_file_path = os.path.join(PRIVATE_STORAGE_ROOT, fe.preloaded_file.name)
                unzip_file(preloaded_file_path, self.exp_folder_complete_path)
                print("Preloaded file unzipped!:", self.exp_folder_complete_path)

        for pp in self.postprocessings.all():
            if hasattr(pp, "preloaded") and pp.preloaded:
                preloaded_file_path = os.path.join(PRIVATE_STORAGE_ROOT, pp.preloaded_file.name)
                unzip_file(preloaded_file_path, self.exp_folder_complete_path)
                print("Preloaded file unzipped!:", self.exp_folder_complete_path)

    def create_folder_structure(self):
        self.exp_foldername = f"exec_{self.id}"
        self.exp_folder_complete_path = os.path.join(self.case_study.exp_folder_complete_path, 'executions', self.exp_foldername)

        if not os.path.exists(self.exp_folder_complete_path):
            os.makedirs(self.exp_folder_complete_path)

        # Create a symbolic link to the case study scenarios to study inside the execution folder
        for scenario in self.scenarios_to_study:
            source = os.path.join('..', '..', scenario)
            destination = os.path.join(self.exp_folder_complete_path, scenario)
            if os.name == 'nt':
                command = f'cmd /c mklink /D "{destination}" "{source}"'
                subprocess.run(command, shell=True)
            else:
                os.symlink(source, destination)
    
    def delete(self):
        # Delete the execution folder
        shutil.rmtree(self.exp_folder_complete_path)
        super().delete()
