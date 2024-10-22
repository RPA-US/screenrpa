# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
"""

import json
from django import forms
from apps.featureextraction.models import UIElementsDetection, UIElementsClassification, Prefilters, Postfilters, FeatureExtractionTechnique, CNNModels, Postprocessing
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
                            document.getElementById('id_model').disabled = true;
                            document.getElementById('id_model').value = '';
                            document.getElementById('id_model').required = false;
                        } else {
                            document.getElementById('id_model').disabled = false;
                            document.getElementById('id_model').required = true;
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
        self.read_only = kwargs.pop('read_only', False)
        super(UIElementsDetectionForm, self).__init__(*args, **kwargs)
        self.fields['configurations'].initial = dict()
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True

class PrefiltersForm(forms .ModelForm):
    class Meta:
        model = Prefilters
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "title",
            # "scale_factor",
            "preloaded_file",
            "preloaded",
        )
        labels = {
            "preloaded_file":_("Preload Execution Results"),
            # "scale_factor": _("Dispersion Scale Factor")
        }

        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control"}),
            # "scale_factor": forms.NumberInput(attrs={'type': 'range', 'min': 1, 'max': 10, 'step': 1}),
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
        self.read_only = kwargs.pop('read_only', False)
        super(PrefiltersForm, self).__init__(*args, **kwargs)
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True
                
        # if self.data.get('preloaded') == 'on':
        #     self.fields['scale_factor'].required = False
        # else:
        #     self.fields['scale_factor'].required = True

class PostfiltersForm(forms .ModelForm):
    class Meta:
        model = Postfilters
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "title",
            # "scale_factor",
            # "intersection_area_thresh",
            "consider_nested_as_relevant",
            "preloaded_file",
            "preloaded",
        )
        labels = {
            "preloaded_file":"Preload Execution Results",
            # "scale_factor": _("Dispersion Scale Factor"),
            # "intersection_area_thresh": "Intersection Area Threshold",
            "consider_nested_as_relevant": _("Consider Nested Components as Relevant")
        }

        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control"}),
            # "scale_factor": forms.NumberInput(attrs={'type': 'range', 'min': 1, 'max': 10, 'step': 1}),
            # "intersection_area_thresh": forms.NumberInput(attrs={"class": "form-control"}),
            "consider_nested_as_relevant": forms.CheckboxInput(attrs={"class": "primary-checkbox"}),
            "preloaded_file": forms.FileInput(attrs={'accept': '.zip'}),
            "preloaded": forms.CheckboxInput(attrs={"class": "primary-checkbox"})
        }

    def __init__(self, *args, **kwargs):
        self.read_only = kwargs.pop('read_only', False)
        super(PostfiltersForm, self).__init__(*args, **kwargs)
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True
        
        # if self.data.get('preloaded') == 'on':
        #     self.fields['scale_factor'].required = False
        # else:
        #     self.fields['scale_factor'].required = True



class UIElementsClassificationForm(forms .ModelForm):
    # Add a 'IGNORE' default value with string ---
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
        self.read_only = kwargs.pop('read_only', False)
        super(UIElementsClassificationForm, self).__init__(*args, **kwargs)
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True
  
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
                    ('', '--- Select type ---'),  # Placeholder
                    ('SINGLE', 'Single: Feature extraction just after UI Elem. Detection'), 
                    ('AGGREGATE', 'Aggregate: Feature extraction after flattening UI log into dataset'), 
                ],
                attrs={
                    "class": "form-control",
                    "required": "false",

                }),
            "technique_name": forms.Select(
                # choices=[
                #     ('ui_elem_location-class','ui_elem_location-class'),
                #     ('ui_elem_location-class_plaintext', 'ui_elem_location-class_plaintext'),
                #     ('class-ui_elem_location', 'class-ui_elem_location'),
                #     ('class_plaintext-ui_elem_location', 'class_plaintext-ui_elem_location'),
                #     ('xpath-class', 'xpath-class'),
                #     ('xpath+ui_elem_class-existence', 'xpath+ui_elem_class-existence'),
                #     ('ui_compo-existence', 'ui_compo-existence'),
                #     ('xpath-class_filtered_by_attention', 'xpath-class_filtered_by_attention'),
                #     ('ui_compos_stats', 'ui_compos_stats'),
                #     ('status', 'status'),
                #     ('quantity', 'quantity')],         
                attrs={
                    "class": "form-control",
                    "required": "false",
                    "onchange": "changeTechniqueOptions()"
                    }
            ),
            "consider_relevant_compos": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "relevant_compos_predicate": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": 'compo["relevant"]=="True" or compo["relevant"] == "Nested"',
                    "required": "false",
                    "disabled": "true"
                    }
            ),
        }

    def __init__(self, *args, **kwargs):
        
        self.read_only = kwargs.pop('read_only', False)
        super(FeatureExtractionTechniqueForm, self).__init__(*args, **kwargs)
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True

class PostprocessingForm(forms.ModelForm):
    class Meta:
        model = Postprocessing
        exclude = (
            "user",
            )
        fields = (
            "technique_name",
            "preloaded_file",
            "preloaded",
            "title"
        )
        labels = {
            "technique_name": _("Technique"),
            "preloaded_file":"Preload Execution Results",
            "title": "Title "
        }

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Postprocessing"
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
            "technique_name": forms.Select(
                attrs={
                    "class": "form-control",
                    "required": "true",
                    "onchange": "changeTechniqueOptions()",
                    }
            ),
        }

    def __init__(self, *args, **kwargs):
        self.read_only = kwargs.pop('read_only', False)
        super(PostprocessingForm, self).__init__(*args, **kwargs)
        techniques_json = json.load(open("configuration/postprocessing_techniques.json"))
        self.fields['technique_name'].widget.choices = list(map(lambda x: list(x), techniques_json.items()))
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True