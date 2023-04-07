# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import GazeAnalysis

class GazeAnalysisForm(forms .ModelForm):
    class Meta:
        model = GazeAnalysis
        exclude = (
            "user",
            )
        fields = (
            "type",
            "configurations"
        )

        widgets = {
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "tony"
                    }
            ),
            "configurations": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "{'eyetracking_log_filename': 'eyetracking_log.csv'}"
                    }
            )
        }

    def __init__(self, *args, **kwargs):
        super(GazeAnalysisForm, self).__init__(*args, **kwargs)