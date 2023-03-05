# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import ExtractTrainingDataset, DecisionTreeTraining
from django.core.exceptions import ValidationError

class ExtractTrainingDatasetForm(forms.ModelForm):
    class Meta:
        model = ExtractTrainingDataset
        exclude = (
            "user",
            )
        fields = (
            "columns_to_drop",
            "columns_to_drop_before_decision_point"
        )

        widgets = {
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
            "algorithms",
            "one_hot_columns",
            "columns_to_drop_before_decision_point"
        )

        widgets = {
            "library": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "chefboost"
                    }
            ),
            "algorithms": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "['ID3','CHAID']"
                    }
            ),
            "one_hot_columns": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "['NameApp']"
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
