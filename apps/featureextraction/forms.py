# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from apps.featureextraction.models import UIElementsDetection, UIElementsClassification, FeatureExtractionTechnique
from django.core.exceptions import ValidationError

class UIElementsDetectionForm(forms .ModelForm):
    class Meta:
        model = UIElementsDetection
        exclude = (
            "user",
            )
        fields = (
            "type",
            "skip" 
        )

        widgets = {
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "uied"
                    }
            ),
            "skip": forms.CheckboxInput(
                attrs={"class": "primary-checkbox", "checked": "checked"}
            )
        }

    def __init__(self, *args, **kwargs):
        super(UIElementsDetectionForm, self).__init__(*args, **kwargs)

class UIElementsClassificationForm(forms .ModelForm):
    class Meta:
        model = UIElementsClassification
        exclude = (
            "user",
            )
        fields = (
            "model",
            "model_properties",
            "type",
            "skip"
        )

        widgets = {
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "uied"
                    }
            ),
            "skip": forms.CheckboxInput(
                attrs={"class": "primary-checkbox", "checked": "checked"}
            ),
            "model": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "resources/models/custom-v2.h5"
                    }
            ),
            "model_properties": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "resources/models/custom-v2-properties.json"
                    }
            )
        }

    def __init__(self, *args, **kwargs):
        super(UIElementsClassificationForm, self).__init__(*args, **kwargs)
   
class FeatureExtractionTechniqueForm(forms.ModelForm):
    class Meta:
        model = FeatureExtractionTechnique
        exclude = (
            "user",
            )
        fields = (
            "technique_name",
            "skip",
            "identifier"
        )

        widgets = {
            "technique_name": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "status"
                    }
            ),
            "skip": forms.CheckboxInput(
                attrs={"class": "primary-checkbox", "checked": "checked"}
            ),
            "identifier": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "sta_s"
                    }
            )
        }

    def __init__(self, *args, **kwargs):
        super(FeatureExtractionTechniqueForm, self).__init__(*args, **kwargs)
