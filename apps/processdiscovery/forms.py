# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import ProcessDiscovery
from django.utils.translation import gettext_lazy as _

class ProcessDiscoveryForm(forms.ModelForm):

    class Meta:
        model = ProcessDiscovery
        exclude = ("user", "created_at")
        fields = (
            "type",
            "configurations",
            "preloaded_file",
            "preloaded",
            "title",
            "model_type",
            "text_weight",
            "image_weight",
            "clustering_type",
            "labeling",
            "use_pca",
            "n_components",
            "show_dendrogram",
        )
        labels = {
            "type": _("Type"),
            "configurations": _("Configurations"),
            "preloaded_file":_("Preload Execution Results"),
            "title": _("Title"),
            "model_type": _("Model Type"),
            "text_weight": _("Text Weight"),
            "image_weight": _("Image Weight"),
            "clustering_type": _("Clustering Type"),
            "labeling": _("Labeling"),
            "use_pca": _("Use PCA"),
            "n_components": _("N Components"),
            "show_dendrogram": _("Show Dendrogram"),
        }
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "Process discovery Technique"}),
            "preloaded_file": forms.FileInput(attrs={'accept': '.zip'}),
            "preloaded": forms.CheckboxInput(attrs={"class": "primary-checkbox"}),
            "type": forms.TextInput(attrs={"class": "form-control", "placeholder": "imotions"}),
            "configurations": forms.Textarea(attrs={"class": "form-control", 'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'}),
            "model_type": forms.Select(attrs={'class': 'form-control'}),
            "text_weight": forms.NumberInput(attrs={'class': 'form-control'}),
            "image_weight": forms.NumberInput(attrs={'class': 'form-control'}),
            "clustering_type": forms.Select(attrs={'class': 'form-control'}),
            "labeling": forms.Select(attrs={'class': 'form-control'}),
            "use_pca": forms.CheckboxInput(attrs={'class': 'custom-control-input'}),
            "n_components": forms.NumberInput(attrs={'class': 'form-control'}),
            "show_dendrogram": forms.CheckboxInput(attrs={'class': 'custom-control-input'}),
        }

    def __init__(self, *args, **kwargs):
        super(ProcessDiscoveryForm, self).__init__(*args, **kwargs)
