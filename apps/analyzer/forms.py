# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import CaseStudy
from django.core.exceptions import ValidationError

class CaseStudyForm(forms.ModelForm):
    class Meta:
        model = CaseStudy
        exclude = (
            "active",
            "created_at",
            "executed",
            "user"
            )
        fields = (
            "title",
            "description",
            "exp_foldername",
            "exp_folder_complete_path",
            "scenarios_to_study",
            "special_colnames",
            "text_classname",
            "decision_point_activity",
            "gui_class_success_regex",
            "ui_elements_classification_image_shape",
            "ui_elements_classification_classes",
            "target_label",
            "ui_elements_detection",
            "noise_filtering",
            "ui_elements_classification",
            "feature_extraction_technique",
            "extract_training_dataset",
            "decision_tree_training"
        )

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Nice experiment title!"
                    }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "Short experiment description..."
                    }
            ),
            "exp_foldername": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "exp_folder"
                    }
            ),
            "exp_folder_complete_path": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "/rim/resources/exp_folder"
                    }
            ),
            "scenarios_to_study": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "['scenario_1', 'scenario_2']"}
            ),
            "special_colnames": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "special_colnames"
                    }
            ),
            "text_classname": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "text_classname"
                    }
            ),
            "decision_point_activity": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "decision_point_activity"
                    }
            ),
            "gui_class_success_regex": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "gui_class_success_regex"
                    }
            ),
            "ui_elements_classification_image_shape": forms.TextInput(
                attrs={
                    "class": "custom-select",
                    "placeholder": "ui_elements_classification_image_shape"
                    }
            ),
            "ui_elements_classification_classes": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "ui_elements_classification_classes"
                    }
            ),
            "target_label": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Variant"
                    }
            ),
            "ui_elements_detection": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
            "noise_filtering": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
            "ui_elements_classification": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
            "feature_extraction_technique": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
            "extract_training_dataset": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
            "decision_tree_training": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            )
        }

    def clean_title(self):
        title = self.cleaned_data.get("title")
        qs = CaseStudy.objects.filter(title=title)
        if qs.exists():
            raise forms.ValidationError("Title is taken")
        return title


    def __init__(self, *args, **kwargs):
        super(CaseStudyForm, self).__init__(*args, **kwargs)