from featureextraction.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from featureextraction.feature_extraction_techniques import quantity_ui_elements_fe_technique, location_ui_elements_and_plaintext_fe_technique, location_ui_elements_fe_technique
from art import tprint
from rim.settings import platform_name, classification_phase_name, feature_extraction_phase_name

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

def feature_extraction(*data):
    tprint(platform_name + " - " + feature_extraction_phase_name, "fancy60")

    data_list = list(data)
    feature_extraction_technique_name = data_list.pop()
    skip = data_list.pop()
    data = tuple(data_list)
    output = None

    print(feature_extraction_technique_name+"\n")
    
    if not skip:
        match feature_extraction_technique_name:
            case "quantity":
                output = quantity_ui_elements_fe_technique(*data)
            case "location":
                output = location_ui_elements_fe_technique(*data)
            case "plaintext":
                output = location_ui_elements_and_plaintext_fe_technique(*data)
            case _:
                pass
    return output