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

@register.filter
def handle_none(value):
    def replace_none(obj):
        if isinstance(obj, dict):
            return {k: replace_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_none(elem) for elem in obj]
        elif obj is None:
            return ""
        else:
            return obj

    return json.dumps(replace_none(value))