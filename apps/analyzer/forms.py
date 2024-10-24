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
        )
        labels = {
            "title": _("Title"),
            "description": _("Description"),
            "exp_file": _("Log file (zip)"),
            "scenarios_to_study": _("Scenarios to study"),
            "special_colnames": _("Map Log Columns to process Info"),
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
            attrs={"class": "form-control", "required": "true", "placeholder": "If there is only one scenario, just type its name. If there are more than one, separate them by comma. E.g. scenario1, scenario2, scenario3"}
            ),
            "special_colnames": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "special_colnames"
                    }
            ),
            "model": forms.Select(
                attrs={
                    "class": "custom-select"
                    }
            ),
        }

    def clean_title(self):
        # Apply only if title is changed
        if self.instance.title == self.cleaned_data.get("title"):
            return self.cleaned_data.get("title")
        title = self.cleaned_data.get("title")
        qs = CaseStudy.objects.filter(title=title)
        if qs.exists():
            raise forms.ValidationError(_("Title is taken"))
        return title


    def __init__(self, *args, **kwargs):
        super(CaseStudyForm, self).__init__(*args, **kwargs)
