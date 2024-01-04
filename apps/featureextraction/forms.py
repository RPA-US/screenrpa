# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from apps.featureextraction.models import UIElementsDetection, UIElementsClassification, Prefilters, Postfilters, FeatureExtractionTechnique
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

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
        labels = {
            "type": _("Type"),
            "skip": _("Skip")
        }

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

class PrefiltersForm(forms .ModelForm):
    class Meta:
        model = Prefilters
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "type",
            "skip",
            "configurations",
        )
        labels = {
            "type": _("Type"),
            "skip": _("Skip"),
            "configurations": _("Configurations")
        }

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
            "configurations": forms.Textarea(attrs={
                'class': 'form-control',
                'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'
            })
        }

    def __init__(self, *args, **kwargs):
        super(PrefiltersForm, self).__init__(*args, **kwargs)

class PostfiltersForm(forms .ModelForm):
    class Meta:
        model = Postfilters
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "type",
            "skip",
            "configurations",
        )
        labels = {
            "type": _("Type"),
            "skip": _("Skip"),
            "configurations": _("Configurations")
        }

        widgets = {
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "rpa-us"
                    }
            ),
            "skip": forms.CheckboxInput(
                attrs={"class": "primary-checkbox", "checked": "checked"}
            ),
            "configurations": forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': "{'gaze':{'UI_selector': 'all','predicate': '(compo['row_min'] <= fixation_point_x) and (fixation_point_x <= compo['row_max']) and (compo['column_min'] <= fixation_point_y) and (fixation_point_y <= compo['column_max'])','only_leaf': true},'filter2':{}}",
                'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'
            })
        }

    def __init__(self, *args, **kwargs):
        super(PostfiltersForm, self).__init__(*args, **kwargs)

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
        labels = {
            "type": _("Type"),
            "skip": _("Skip"),
            "model": _("Model"),
            "model_properties": _("Model properties")
        }

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
        labels = {
            "technique_name": _("Technique name"),
            "skip": _("Skip"),
            "identifier": _("Identifier")
        }

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