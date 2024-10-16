# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
"""

from django import forms
from .models import ProcessDiscovery
from django.utils.translation import gettext_lazy as _

class ProcessDiscoveryForm(forms.ModelForm):

    class Meta:
        model = ProcessDiscovery
        exclude = ("user", "created_at")
        fields = (
            "configurations",
            "preloaded_file",
            "preloaded",
            "title",
            "model_type",
            "text_weight",
            "image_weight",
            "clustering_type",
            "use_pca",
            "n_components",
            "show_dendrogram",
            "remove_loops",
            "text_column"
        )
        labels = {
            "configurations": _("Configurations"),
            "preloaded_file":_("Preload Execution Results"),
            "title": _("Title"),
            "model_type": _("Model Type"),
            "text_weight": _("Text Weight"),
            "image_weight": _("Image Weight"),
            "clustering_type": _("Clustering Type"),
            "use_pca": _("Use PCA"),
            "n_components": _("N Components"),
            "show_dendrogram": _("Show Dendrogram"),
            "remove_loops": _("Remove Loops"),
            "text_column": _("Text Column"),
            
        }
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "Process discovery Technique"}),
            "preloaded_file": forms.FileInput(attrs={'accept': '.zip'}),
            "preloaded": forms.CheckboxInput(attrs={"class": "primary-checkbox"}),
            "configurations": forms.Textarea(attrs={"class": "form-control", 'onchange': 'this.value = JSON.stringify(JSON.parse(this.value), null, 4);'}),
            "model_type": forms.Select(attrs={'class': 'form-control'}),
            "text_weight": forms.NumberInput(attrs={'class': 'form-control'}),
            "image_weight": forms.NumberInput(attrs={'class': 'form-control'}),
            "clustering_type": forms.Select(attrs={'class': 'form-control'}),
            "use_pca": forms.CheckboxInput(attrs={'class': 'custom-control-input'}),
            "n_components": forms.NumberInput(attrs={'class': 'form-control'}),
            "show_dendrogram": forms.CheckboxInput(attrs={'class': 'custom-control-input'}),
            "remove_loops": forms.CheckboxInput(attrs={'class': 'custom-control-input'}),
            "text_column": forms.TextInput(attrs={'class': 'form-control'}),
        }

    def __init__(self, *args, **kwargs):
        # Pop the 'read_only' and 'case_study' from kwargs before passing to the superclass constructor
        self.read_only = kwargs.pop('read_only', False)
        case_study_instance = kwargs.pop('case_study', None)
        
        # Call the superclass constructor with the remaining arguments
        super(ProcessDiscoveryForm, self).__init__(*args, **kwargs)
        
        # Handle the 'read_only' functionality
        if self.read_only:   
            for field_name in self.fields:
                self.fields[field_name].disabled = True
                
        
        # Handle the 'case_study' functionality
        if case_study_instance is not None:
            special_colnames = case_study_instance.special_colnames
            text_column_choices = [(value, key) for key, value in special_colnames.items()]
            self.fields['text_column'] = forms.ChoiceField(
                choices=text_column_choices,
                widget=forms.Select(attrs={'class': 'form-control'}),
                label=_("Text Column"))
