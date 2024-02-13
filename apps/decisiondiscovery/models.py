from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.contrib.auth.models import User
from django.db import models
from django.core.exceptions import ValidationError
from django.contrib.postgres.fields import ArrayField
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

# Create your models here.

def get_default_extract_training_columns_to_ignore():
    return 'Coor_X, Coor_Y, Case'.split(', ') # this returns a list

def get_default_decision_tree_columns_to_ignore():
    return 'Timestamp_start, Timestamp_end'.split(', ') # this returns a list
def default_dd_configuration():
    return {
                "feature_values": {
                    "1": {
                        "or_cond": {
                            "numeric__qua2_TextView_2_B": 11.0,
                            "numeric__qua2_ImageView_2_B": 3.0,
                            "numeric__qua2_ImageButton_2_B": 63.0,
                            "status_categorical__sta_enabled_1044.0-775.0_2_B": 1.0,
                            "sta_enabled_717.5-606.5_2_B": 1.0
                        },
                        "or_cond_2": {
                            "sta_checked_649.0-1110.5_4_D": 1.0,
                            "status_categorical__sta_checked_649.0-1110.5_4_D": 1.0,
                            "numeric__qua2_Checkbox_checked_4_D": 1.0,
                            "numeric__qua2_Checkbox_unchecked_4_D": 0.0
                        }
                    },
                    "2": {
                        "or_cond": {
                            "numeric__qua2_TextView_2_B": 11.0,
                            "numeric__qua2_ImageView_2_B": 3.0,
                            "numeric__qua2_ImageButton_2_B": 63.0,
                            "sta_enabled_717.5-606.5_2_B": 1.0,
                            "status_categorical__sta_enabled_717.5-606.5_2_B": 1.0
                        },
                        "or_cond_2": {
                            "sta_checked_649.0-1110.5_4_D": 0.0,
                            "status_categorical__sta_checked_649.0-1110.5_4_D": 0.0,
                            "numeric__qua2_Checkbox_checked_4_D": 0.0,
                            "numeric__qua2_Checkbox_unchecked_4_D": 1.0
                        }
                    },
                    "3": {
                        "or_cond": {
                            "numeric__qua2_TextView_2_B": 0.0,
                            "numeric__qua2_ImageView_2_B": 0.0,
                            "numeric__qua2_ImageButton_2_B": 5.0,
                            "status_categorical__sta_enabled_717.5-606.5_2_B": 0.0,
                            "sta_enabled_717.5-606.5_2_B": 0.0
                        },
                        "or_cond_2": {
                            "sta_checked_649.0-1110.5_4_D": 1.0,
                            "status_categorical__sta_checked_649.0-1110.5_4_D": 1.0,
                            "numeric__qua2_Checkbox_checked_4_D": 1.0,
                            "numeric__qua2_Checkbox_unchecked_4_D": 0.0
                        }
                    },
                    "4": {
                        "or_cond": {
                            "numeric__qua2_TextView_2_B": 0.0,
                            "numeric__qua2_ImageView_2_B": 0.0,
                            "numeric__qua2_ImageButton_2_B": 5.0,
                            "sta_enabled_717.5-606.5_2_B": 0.0,
                            "status_categorical__sta_enabled_717.5-606.5_2_B": 0.0
                        },
                        "or_cond_2": {
                            "sta_checked_649.0-1110.5_4_D": 0.0,
                            "status_categorical__sta_checked_649.0-1110.5_4_D": 0.0,
                            "numeric__qua2_Checkbox_checked_4_D": 0.0,
                            "numeric__qua2_Checkbox_unchecked_4_D": 1.0
                        }
                    }
                },
                "centroid_threshold": 10000
            }

# def get_default_algorithms():
#     return 'ID3, CART, CHAID, C4.5'.split(', ') # this returns a list

class ExtractTrainingDataset(models.Model):
    target_label = models.CharField(max_length=50, default='Variant')
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)

    title = models.CharField(max_length=255, blank=True)
    columns_to_drop = ArrayField(models.CharField(max_length=25), default=get_default_extract_training_columns_to_ignore)
    columns_to_drop_before_decision_point = ArrayField(models.CharField(max_length=25), default=get_default_extract_training_columns_to_ignore)
    decision_point_activity = models.CharField(max_length=255)
    configurations = models.JSONField(default=dict, blank=True, null=True)

    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("decisiondiscovery:extract_training_dataset_list", args=[str(self.case_study_id)])    
    
    def __str__(self):
        return 'col to drop: ' + str(self.columns_to_drop)
    
class DecisionTreeTraining(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    configuration = models.JSONField(default=default_dd_configuration)
    library = models.CharField(max_length=255, default='sklearn') # 'sklearn'
    one_hot_columns = ArrayField(models.CharField(max_length=25))
    columns_to_drop_before_decision_point = ArrayField(models.CharField(max_length=50), default=get_default_decision_tree_columns_to_ignore)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def clean(self):
        cleaned_data = super().clean()
        if not ExtractTrainingDataset.objects.exists(case_study__id=self.case_study.id):
            raise ValidationError(_("To be able to apply decision tree training, a extract training dataset has to exist"))
        return cleaned_data
    
    def get_absolute_url(self):
        return reverse("decisiondiscovery:decision_tree_training_list", args=[str(self.case_study_id)])
    
    def delete(self):
        
        # TODO: si existe relacion con experiment
        
        super.delete()
    
    def __str__(self):
        return 'library: ' + self.library + ' - algs:' + str(self.configuration)