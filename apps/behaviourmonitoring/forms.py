# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
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
            "gaze_log_filename",
            "gaze_log_adjustment",
            "native_slide_events",
            "preloaded_file",
            "preloaded",
            "screen_inches",
            "observer_camera_distance",
            "screen_width",
            "screen_height"
        )
        labels = {
            "type": _("Type"),
            "ui_log_filename": _("UI Log Filename"),
            "gaze_log_filename": _("Gaze Log Filename"),
            "gaze_log_adjustment": _("Gaze Log Adjustment"),
            "native_slide_events": _("System Info. Log"),
            "preloaded_file":"Preload Execution Results",
            "screen:_inches":_("Screen Inches"),
            "observer_camera_distance":_("Observer to Webcam distance (in cm)"),
            "screen_width":_("Screen Width (in pixels)"),
            "screen_height":_("Screen Height (in pixels)")
        }

        widgets = {
            # Type is a choice field
            "type": forms.Select(
                choices=[('tobii', 'Tobii Spark Pro Eye tracking software'), ('webgazer', 'Webgazer.js Eye tracking software'),('imotions', 'iMotions Infrared Eye tracker')],
                attrs={
                    "class": "form-control",
                    "required": "false"
                    }
            ),
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "ui_log_filename": forms.TextInput(attrs={"class": "form-control"}),
            "gaze_log_filename": forms.TextInput(attrs={"class": "form-control"}),
            # Gaze log adj. is a float number
            "gaze_log_adjustment": forms.NumberInput(attrs={"class": "form-control"}),
            "native_slide_events": forms.TextInput(attrs={"class": "form-control"}),
                        "preloaded": forms.CheckboxInput(
                attrs={"class": "primary-checkbox"}
            ),
            "preloaded_file": forms.FileInput(
                attrs={
                    'accept': '.zip'
                    }   
            ),
            "screen_inches": forms.NumberInput(attrs={"class": "form-control"}),
            "observer_camera_distance": forms.NumberInput(attrs={"class": "form-control"}),
            "screen_width": forms.NumberInput(attrs={"class": "form-control"}),
            "screen_height": forms.NumberInput(attrs={"class": "form-control"})
        }

    def __init__(self, *args, **kwargs):
        self.read_only = kwargs.pop('read_only', False)
        super(MonitoringForm, self).__init__(*args, **kwargs)
        if self.read_only:
            for field_name in self.fields:
                self.fields[field_name].disabled = True