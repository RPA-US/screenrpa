from django import template
import json
# from apps.featureextraction.models import FeatureExtractionTechnique, UIElementsDetection, UIElementsClassification
# from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining

register = template.Library()
json_attributes = ["id", "_state"]


# @register.filter
# def pretty_json(value):
#     # classname = value.__class__.__name__
#     # obj = classname.objects.get(value.id)
#     if value:
#         value = value.__dict__
#         for a in json_attributes:
#             if a in value:
#                 value.pop(a)
#     return value


@register.filter(name='pretty_json')
def pretty_json(value):
    try:
        value = value.__dict__
        if value is not None:
            if isinstance(value,dict):
                return value
            else:
                return json.loads(value)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", str(e))
        pass
    
    except Exception as e:
        print(e)

@register.filter
def divide(value, arg):
    try:
        return round(int(value) / int(arg),2)
    except (ValueError, ZeroDivisionError):
        return None