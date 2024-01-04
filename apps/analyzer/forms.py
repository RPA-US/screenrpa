# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import CaseStudy
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class CaseStudyForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['exp_file'].required = True
    
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
            "exp_file",
            # "exp_foldername",
            # "exp_folder_complete_path",
            "scenarios_to_study",
            "special_colnames",
            # "text_classname",
            # "decision_point_activity",
            "gui_class_success_regex",
            "target_label"
        )
        labels = {
            "title": _("Title"),
            "description": _("Description"),
            "exp_file": _("Experiment file"),
            # "exp_foldername": _("Experiment foldername"),
            # "exp_folder_complete_path": _("Experiment folder complete path"),
            "scenarios_to_study": _("Scenarios to study"),
            "special_colnames": _("Special colnames"),
            # "text_classname": _("Text classname"),
            # "decision_point_activity": _("Decision point activity"),
            "gui_class_success_regex": _("GUI class success regex"),
            # "ui_elements_classification_image_shape": _("UI elements classification image shape"),
            # "ui_elements_classification_classes": _("UI elements classification classes"),
            "target_label": _("Target label")
        }

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": _("Nice experiment title!")
                    }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": _("Short experiment description...")
                    }
            ),
            'exp_file': forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }
            ),
            # "exp_foldername": forms.TextInput(
            #     attrs={
            #         "class": "form-control",
            #         "placeholder": "exp_folder"
            #         }
            # ),
            # "exp_folder_complete_path": forms.TextInput(
            #     attrs={
            #         "class": "form-control",
            #         "placeholder": "/rim/resources/exp_folder"
            #         }
            # ),
            "scenarios_to_study": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "['scenario_1', 'scenario_2']"}
            ),
            "special_colnames": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "special_colnames"
                    }
            ),
            # "text_classname": forms.TextInput(
            #     attrs={
            #         "class": "form-control",
            #         "placeholder": "text_classname"
            #         }
            # ),
            # "decision_point_activity": forms.TextInput(
            #     attrs={
            #         "class": "form-control",
            #         "placeholder": "decision_point_activity"
            #         }
            # ),
            "gui_class_success_regex": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "gui_class_success_regex"
                    }
            ),
            # "ui_elements_classification_image_shape": forms.TextInput(
            #     attrs={
            #         "class": "custom-select",
            #         "placeholder": "ui_elements_classification_image_shape"
            #         }
            # ),
            # "ui_elements_classification_classes": forms.Textarea(
            #     attrs={
            #         "class": "form-control",
            #         "placeholder": "ui_elements_classification_classes"
            #         }
            # ),
            "target_label": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": _("Variant")
                    }
            ),
            "model": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
            # "monitoring": forms.Select(
            #     attrs={
            #         "class": "custom-select"
            #         }
            # ),
            # "ui_elements_classification": forms.Select(
            #     attrs={
            #         "class": "custom-select"
            #         }
            # ),
            # "feature_extraction_technique": forms.Select(
            #     attrs={
            #         "class": "custom-select"
            #         }
            # ),
            # "extract_training_dataset": forms.Select(
            #     attrs={
            #         "class": "custom-select"
            #         }
            # ),
            # "decision_tree_training": forms.Select(
            #     attrs={
            #         "class": "custom-select"
            #         }
            # )
        }

    def clean_title(self):
        title = self.cleaned_data.get("title")
        qs = CaseStudy.objects.filter(title=title)
        if qs.exists():
            raise forms.ValidationError(_("Title is taken"))
        return title


    def __init__(self, *args, **kwargs):
        super(CaseStudyForm, self).__init__(*args, **kwargs)

