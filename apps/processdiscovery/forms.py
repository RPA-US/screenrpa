# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from .models import ProcessDiscovery
from django.utils.translation import gettext_lazy as _

class ProcessDiscoveryForm(forms.ModelForm):
    
    model_type = forms.ChoiceField(
        choices=[('vgg', 'VGG'), ('clip', 'Clip')],
        label=_("Model Type"),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    text_weight = forms.DecimalField(
        required=False,
        label=_("Text Weight"),
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        initial=0.5
    )
    image_weight = forms.DecimalField(
        required=False,
        label=_("Image Weight"),
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        initial=0.5
    )
    clustering_type = forms.ChoiceField(
        choices=[('hierarchical', 'Hierarchical')],
        label=_("Clustering Type"),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    labeling = forms.ChoiceField(
        choices=[('automatic', 'Automatic'), ('manual', 'Manual')],
        label=_("Labeling"),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    use_pca = forms.BooleanField(
        required=False,
        label=_("Use PCA"),
        widget=forms.CheckboxInput(attrs={'class': 'custom-control-input'})
    )
    n_components = forms.FloatField(
        label=_("N Components"),
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        initial=0.95
    )
    show_dendrogram = forms.BooleanField(
        required=False,
        label=_("Show Dendrogram"),
        widget=forms.CheckboxInput(attrs={'class': 'custom-control-input'})
    )

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
