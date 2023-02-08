from apps.featureextraction.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from art import tprint
from core.settings import platform_name, classification_phase_name, feature_extraction_phase_name
from apps.analyzer.utils import detect_fe_function

def ui_elements_classification(*data):
    # Classification can be done with different algorithms
    data_list = list(data)
    classifier_type = data_list.pop()
    data = tuple(data_list)

    tprint(platform_name + " - " + classification_phase_name, "fancy60")
    print(data_list[4]+"\n")
    
    match classifier_type:
        case "rpa-us":
            output = legacy_ui_elements_classification(*data)
        case "uied":
            output = uied_ui_elements_classification(*data)
        case _:
            pass
    return output

def feature_extraction_technique(*data):
    tprint(platform_name + " - " + feature_extraction_phase_name, "fancy60")

    data_list = list(data)
    feature_extraction_technique_name = data_list.pop()
    skip = data_list.pop()
    data = tuple(data_list)
    output = None

    print("Feature extraction selected: " + feature_extraction_technique_name+"\n")
    
    if not skip:
        output = detect_fe_function(feature_extraction_technique_name)(*data)
    return output