# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
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
            "columns_to_drop_before_decision_point",
            "configurations",
            "preloaded_file",
            "preloaded"
        )
        labels = {
            "title": _("Title"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "configurations": _("Additional Configurations (JSON)"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
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
        self.read_only = kwargs.pop('read_only', False)
        super(ExtractTrainingDatasetForm, self).__init__(*args, **kwargs)
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True
        
        
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
            "title",
            "balance_weights"
        )
        labels = {
            "library": _("Library"),
            "configuration": _("Configuration"),
            "one_hot_columns": _("One hot columns"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "preloaded_file":"Preload Execution Results",
            "title": "Title ",
            "balance_weights": "Balance Weights"
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
            "balance_weights": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
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
        self.read_only = kwargs.pop('read_only', False)
        super(DecisionTreeTrainingForm, self).__init__(*args, **kwargs)

        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True