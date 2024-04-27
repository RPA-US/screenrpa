# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import PDD
from django.utils.translation import gettext_lazy as _

class ReportingForm(forms.ModelForm):

    class Meta:
        model = PDD
        fields = (
            "objective",
            "purpose",
            "indice_contenido",
            "process_overview",
            "applications_used",
            "as_is_process_map",
            "detailed_as_is_process_actions",
            "input_data_description",
        )

        labels = {
            "objective": _("Objective"),
            "purpose": _("Purpose"),
            "indice_contenido": _("Content Index"),
            "process_overview": _("Process Overview"),
            "applications_used": _("Applications Used"),
            "as_is_process_map": _("AS IS Process Map"),
            "detailed_as_is_process_actions": _("Detailed AS IS Process Actions"),
            "input_data_description": _("Input Data Description"),
        }

        widgets = {
            "objective": forms.TextInput(attrs={"class": "form-control", "placeholder": _("Objective of the PDD")}),
            "purpose": forms.Textarea(attrs={"class": "form-control", "placeholder": _("Purpose of the PDD...")}),
            #"indice_contenido": forms.CheckboxSelectMultiple(choices=PDD.CONTENT_INDEX_CHOICES),
            "process_overview": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "applications_used": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "as_is_process_map": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "detailed_as_is_process_actions": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "input_data_description": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }

    def clean_objective(self):
        objective = self.cleaned_data.get("objective")
        if PDD.objects.filter(objective=objective).exists():
            raise forms.ValidationError(_("Objective already exists"))
        return objective

    def clean_purpose(self):
        purpose = self.cleaned_data.get("purpose")
        # Add your validation logic for the purpose field if necessary
        return purpose
    

    def __init__(self, *args, **kwargs):
        super(ReportingForm, self).__init__(*args, **kwargs)
