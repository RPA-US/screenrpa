# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import Monitoring
from django.utils.translation import gettext_lazy as _

class MonitoringForm(forms .ModelForm):
    class Meta:
        model = Monitoring
        exclude = (
            "user",
            "created_at",
            )
        fields = (
            "title",
            "type",
            "ui_log_filename",
            "ui_log_separator",
            "gaze_log_filename",
            "gaze_log_adjustment",
            "native_slide_events",
        )
        labels = {
            "type": _("Type"),
            "ui_log_filename": _("UI Log Filename"),
            "ui_log_separator": _("UI Log Separator"),
            "gaze_log_filename": _("Gaze Log Filename"),
            "gaze_log_adjustment": _("Gaze Log Adjustment"),
            "native_slide_events": _("System Info. Log"),
        }

        widgets = {
            # Type is a choice field
            "type": forms.Select(
                choices=[('imotions', 'imotions'), ('webgazer', 'webgazer')],
                attrs={
                    "class": "form-control",
                    "required": "false"
                    }
            ),
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "ui_log_filename": forms.TextInput(attrs={"class": "form-control"}),
            "ui_log_separator": forms.TextInput(attrs={"class": "form-control"}),
            "gaze_log_filename": forms.TextInput(attrs={"class": "form-control"}),
            # Gaze log adj. is a float number
            "gaze_log_adjustment": forms.NumberInput(attrs={"class": "form-control"}),
            "native_slide_events": forms.TextInput(attrs={"class": "form-control"}),
        }

    def __init__(self, *args, **kwargs):
        super(MonitoringForm, self).__init__(*args, **kwargs)