# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import ProcessDiscovery
from django.utils.translation import gettext_lazy as _

class ProcessDiscoveryForm(forms .ModelForm):
    class Meta:
        model = ProcessDiscovery
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "type",
            "configurations",
            "preloaded_file",
            "preloaded",
            "title"
        )
        labels = {
            "type": _("Type"),
            "configurations": _("Configurations"),
             "preloaded_file":"Preload Execution Results",
            "title": "Title "
        }

        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Process discovery Technique"
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
            "type": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "imotions"
                    }
            ),
            "configurations": forms.Textarea(
                attrs={
                    "class": "form-control",
                    'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'
            })
        }

    def __init__(self, *args, **kwargs):
        super(ProcessDiscoveryForm, self).__init__(*args, **kwargs)