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
            "configurations"
        )
        labels = {
            "title": _("Title"),
            "decision_point_activity": _("Decision Point To Study"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "configurations": _("Additional Configurations (JSON)")
        }

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "My Dataset Extraction Technique"
                    }
            ),
            "decision_point_activity": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "C"
                    }
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
            "columns_to_drop_before_decision_point"
        )
        labels = {
            "library": _("Library"),
            "configuration": _("Configuration"),
            "one_hot_columns": _("One hot columns"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point")
        }

        widgets = {
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
