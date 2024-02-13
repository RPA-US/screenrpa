from django import template
import json
# from apps.featureextraction.models import FeatureExtractionTechnique, UIElementsDetection, UIElementsClassification
# from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining

register = template.Library()
json_attributes = ["id", "_state"]


@register.filter
def divide(value, arg):
    try:
        return round(int(value) / int(arg),2)
    except (ValueError, ZeroDivisionError):
        return None