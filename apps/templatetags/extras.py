import json

from django import template
from apps.featureextraction.models import FeatureExtractionTechnique, UIElementsDetection, UIElementsClassification, GazeAnalysis
from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining

register = template.Library()

@register.filter
def pretty_json(value):
    # classname = value.__class__.__name__
    # obj = classname.objects.get(value.id)
    if value:
        value = value.__dict__
        value.pop("_state")
        value.pop("id")
    return value