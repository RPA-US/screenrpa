# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from apps.featureextraction.models import UIElementsDetection, UIElementsClassification, Prefilters, Postfilters, FeatureExtractionTechnique, CNNModels
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class UIElementsDetectionForm(forms .ModelForm):
    class Meta:
        model = UIElementsDetection
        exclude = (
            "user",
            )
        fields = (
            "title",
            "type",
            "configurations",
            "ocr",
            "preloaded_file",
            "preloaded",
        )
        labels = {
            "type": _("Type"),
            "skip": _("Skip")
        }

        labels = {
            "title": "Title *",
            "type": "Technique *",
            "configurations": "Additional Configurations",
            "ocr": "Apply OCR *",
            "preloaded_file":"Preload Execution Results"
        }

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "UI Elements Detection"
                    }
            ),
            # type is a selectable
            "type": forms.Select(
                choices=[('rpa-us', 'Kevin Moran'),('screen2som', 'Screen2SOM'), ('uied', 'UIED'), ('sam', 'SAM'), ('fast-sam', 'Fast-SAM')],
                attrs={
                    "class": "form-control",
                    # If value is screen2som, disable CNN model selectable
                    "onchange": """
                        if (this.value == 'screen2som') {
                            document.getElementById('id_model').value = 'IGNORE';
                            document.getElementById('id_model').disabled = true;
                        } else {
                            document.getElementById('id_model').disabled = false;
                        }
                        """
                    }
            ),
            "configurations": forms.Textarea(attrs={
                'class': 'form-control',
                'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'
            }),
            "ocr": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "preloaded": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "preloaded_file": forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }   
            ),

        }
    
    def __init__(self, *args, **kwargs):
        super(UIElementsDetectionForm, self).__init__(*args, **kwargs)
        self.fields['configurations'].initial = dict()

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
            "preloaded_file",
            "preloaded",
            "title"
        )
        labels = {
            "type": _("Type"),
            "skip": _("Skip"),
            "configurations": _("Configurations"),
            "preloaded_file":"Preload Execution Results"
        }

        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "type": forms.Select(
                choices=[('screen2som', 'Screen2SOM'), ('rpa-us', 'Kevin Moran'), ('uied', 'UIED'), ('sam', 'SAM'), ('fast-sam', 'Fast-SAM')],
                attrs={
                    "class": "form-control"
                }
            ),
            "skip": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "configurations": forms.Textarea(attrs={
                'class': 'form-control',
                'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'
            }),
            "preloaded_file": forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }   
            ),
            "preloaded": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            )
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
            "preloaded_file",
            "preloaded",
            "title"
        )
        labels = {
            "type": _("Type"),
            "skip": _("Skip"),
            "configurations": _("Configurations"),
            "preloaded_file":"Preload Execution Results"
        }

        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "New-postfiltering"
                    }
            ),
            "skip": forms.CheckboxInput(
                attrs={"class": "primary-checkbox", "checked": "checked"}
            ),
            "preloaded_file": forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }   
            ),
            "preloaded": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
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
    model = forms.ModelChoiceField(
        queryset=CNNModels.objects.all().exclude(name="screen2som"),
        to_field_name="name",
        empty_label="---",
        required=False,
        widget=forms.Select(
            attrs={
                "class": "form-control",
            }
        )
    )
    class Meta:
        model = UIElementsClassification
        exclude = (
            "user",
            "preloaded",
            "preloaded_file",
            )
        fields = (
            "type",
        )
        labels = {
            "type": _("Type"),
            "skip": _("Skip"),
        }

        labels = {
            "type": "Technique *",
        }

        widgets = {
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "uied"
                    }
            ),
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
            "identifier",
            "type",
            "technique_name",
            "consider_relevant_compos",
            "relevant_compos_predicate",
            "preloaded_file",
            "preloaded",
            "title"
        )
        labels = {
            "identifier": _("Identifier"),
            "type": _("Feature extraction type"),
            "technique_name": _("Technique"),
            "consider_relevant_compos": _("Apply Filtering (Relevant Component Selection)"),
            "relevant_compos_predicate": _("Condition for a UI Component to be relevant"),
            "preloaded_file":"Preload Execution Results",
            "title": "Title "
        }

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Feature Extraction Technique"
                    }
            ),
            "identifier": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "sta_s"
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
            "type": forms.Select(
                choices=[
                    ('SINGLE', 'Single: Feature extraction just after UI Elem. Detection'), 
                    ('AGGREGATE', 'Aggregate: Feature extraction after flattening UI log into dataset'), 
                ],
                attrs={
                    "class": "form-control",
                    "required": "false"
                }),
            "technique_name": forms.Select(
                # choices=[('quantity', 'Quantity of UI Elem. per class'), 
                #          ('location', 'Location (UI Elem. Centroid)'), 
                #          ('plaintext', 'Location (UI Elem. Centroid) + Plain Text'), 
                #          ('status', 'UI Elem. Status'), 
                #          ('number_ui_element', 'Number of UI Elem.'),
                #          ('caption_ui_element', 'Caption of UI Elem.')],
                attrs={
                    "class": "form-control",
                    "required": "false"
                    }
            ),
            "consider_relevant_compos": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "relevant_compos_predicate": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": 'compo["relevant"]=="True" or compo["relevant"] == "Nested"',
                    "required": "false"
                    }
            ),
        }

    def __init__(self, *args, **kwargs):
        super(FeatureExtractionTechniqueForm, self).__init__(*args, **kwargs)