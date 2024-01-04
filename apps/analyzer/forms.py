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
            "scenarios_to_study",
            "special_colnames",
            "gui_class_success_regex",
            "target_label"
        )
        labels = {
            "title": _("Title"),
            "description": _("Description"),
            "exp_file": _("Experiment file"),
            "scenarios_to_study": _("Scenarios to study"),
            "special_colnames": _("Special colnames"),
            "gui_class_success_regex": _("GUI class success regex"),
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
            "scenarios_to_study": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "['scenario_1', 'scenario_2']"}
            ),
            "special_colnames": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "special_colnames"
                    }
            ),
            "gui_class_success_regex": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "gui_class_success_regex"
                    }
            ),
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
        }

    def clean_title(self):
        title = self.cleaned_data.get("title")
        qs = CaseStudy.objects.filter(title=title)
        if qs.exists():
            raise forms.ValidationError(_("Title is taken"))
        return title


    def __init__(self, *args, **kwargs):
        super(CaseStudyForm, self).__init__(*args, **kwargs)
