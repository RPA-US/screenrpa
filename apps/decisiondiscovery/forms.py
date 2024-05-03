# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import ExtractTrainingDataset, DecisionTreeTraining
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class ExtractTrainingDatasetForm(forms.ModelForm):
    class Meta:
        model = ExtractTrainingDataset
        exclude = (
            "user",
            )
        fields = (
            "title",
            "decision_point_activity",
            "columns_to_drop_before_decision_point",
            "configurations",
            "preloaded_file",
            "variants_to_study",
            "preloaded"
        )
        labels = {
            "title": _("Title"),
            "decision_point_activity": _("Decision Point To Study"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "configurations": _("Additional Configurations (JSON)"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "target_label": _("Target label"),
            "variants_to_study": _("Variants to study"),
            "preloaded_file":"Preload Execution Results"
        }

        widgets = {
            "title": forms.TextInput(attrs={
                    "class": "form-control",
                    "placeholder": "Extract Training Technique"
                    }
            ),
             "preloaded_file": forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }   
            ),
            "preloaded": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "target_label": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": _("Variant")
                    }
            ),
            "decision_point_activity": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "C"
                    }
            ),
            "variants_to_study": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Let it empty to study all variants. If there is only one variant, just type its name. If there are more than one, separate them by comma. E.g. v1, v2, v3"}
            ),
            "columns_to_drop_before_decision_point": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "['Coor_X', 'Coor_Y', 'MorKeyb', 'TextInput', 'Click']"
                    }
            ),
            "configurations": forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': "{}",
                'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'
            })
        }

    def __init__(self, *args, **kwargs):
        super(ExtractTrainingDatasetForm, self).__init__(*args, **kwargs)
   
        
class DecisionTreeTrainingForm(forms.ModelForm):
    class Meta:
        model = DecisionTreeTraining
        exclude = (
            "user",
            )
        fields = (
            "library",
            "configuration",
            "one_hot_columns",
            "columns_to_drop_before_decision_point",
             "preloaded_file",
            "preloaded",
            "title"
        )
        labels = {
            "library": _("Library"),
            "configuration": _("Configuration"),
            "one_hot_columns": _("One hot columns"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "preloaded_file":"Preload Execution Results",
            "title": "Title "
        }

        widgets = {
             "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Decision Tree Technique"
                    }
            ),
            "preloaded_file": forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }   
            ),
            "preloaded": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "library": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "chefboost"
                    }
            ),
            "configuration": forms.Textarea(
                attrs={
                    "class": "form-control"
                }
            ),
            "one_hot_columns": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "'NameApp', 'User'"
                    }
            ),
            "columns_to_drop_before_decision_point": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "['Timestamp_start', 'Timestamp_end']"
                    }
            )
        }

    def __init__(self, *args, **kwargs):
        super(DecisionTreeTrainingForm, self).__init__(*args, **kwargs)
