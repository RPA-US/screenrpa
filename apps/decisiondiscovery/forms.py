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
            "columns_to_drop",
            "columns_to_drop_before_decision_point",
            "target_label",
            
        )
        labels = {
            "columns_to_drop": _("Columns to drop"),
            "columns_to_drop_before_decision_point": _("Columns to drop before decision point"),
            "target_label": _("Target label")
        }

        widgets = {
            "target_label": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": _("Variant")
                    }
            ),
            "columns_to_drop": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "['Case','Activity','Timestamp','Screenshot','Variant']"
                    }
            ),
            "columns_to_drop_before_decision_point": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "['Coor_X', 'Coor_Y', 'MorKeyb', 'TextInput', 'Click']"
                    }
            )
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
