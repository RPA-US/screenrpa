# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import Monitoring

class MonitoringForm(forms .ModelForm):
    class Meta:
        model = Monitoring
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "type",
            "configurations"
        )

        widgets = {
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
        super(MonitoringForm, self).__init__(*args, **kwargs)