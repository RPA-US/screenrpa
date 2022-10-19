from featureextraction.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from featureextraction.feature_extraction_techniques import quantity_ui_elements_fe_technique, location_ui_elements_and_plaintext_fe_technique, location_ui_elements_fe_technique

def ui_elements_classification(*data):
    # Classification can be done with different algorithms
    data_list = list(data)
    classifier = data_list.pop()
    data = tuple(data_list)
    match classifier:
        case "legacy":
            output = legacy_ui_elements_classification(*data)
        case "uied":
            output = uied_ui_elements_classification(*data)
        case _:
            pass
    return output

def feature_extraction(*data):
    match data[len(data)-1]:
        case "quantity":
            output = quantity_ui_elements_fe_technique(*data)
        case "location":
            output = location_ui_elements_fe_technique(*data)
        case "plaintext":
            output = location_ui_elements_and_plaintext_fe_technique(*data)
        case _:
            pass
    return output