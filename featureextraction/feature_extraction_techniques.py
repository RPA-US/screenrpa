
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def quantity_ui_elements_fe_technique(
    ui_elements_classification_classes, 
    screenshot_colname, metadata_json_root, ui_log_path,
    enriched_log_output_path):
    """
    Since not all images have all classes, a dataset with different columns depending on the images will be generated.
    It will depend whether GUI components of every kind appears o only a subset of these. That is why we initiañize a 
    dataframe will all possible columns, and include row by row the results obtained from the predictions

    :param feature_extraction_technique_name: Feature extraction technique name
    :type feature_extraction_technique_name: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param skip: Rewrite log
    :type skip: bool

    """
    log = pd.read_csv(ui_log_path, sep=",")
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    quantity_ui_elements = {}

    for screenshot_filename in screenshot_filenames:
        print("holaa")
        # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
        with open(metadata_json_root + screenshot_filename + '.json', 'r') as f:
            data = json.load(f)

        for c in ui_elements_classification_classes:
            counter = 0
            for j in range(0, len(data["compos"])):
                if data["compos"][j]["class"] == c:
                    counter+=1
            quantity_ui_elements[c] = counter

        data["features"]["quantity"] = quantity_ui_elements
        with open(metadata_json_root + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(data, jsonFile)

    df = pd.DataFrame([], columns=ui_elements_classification_classes)
    for i in range(0, len(screenshot_filenames)):
        df.loc[i] = quantity_ui_elements.values()

    """
    Once the dataset corresponding to the ammount of elements of each class contained in each of the images is obtained,
    we merge it with the complete log, adding the extracted characteristics from the images
    """

    log_enriched = log.join(df).fillna(method='ffill')

    """
    Finally we obtain an entiched log, which is turned as proof of concept of our hypothesis based on the premise that if
    we not only capture keyboard or mouse events on the monitorization through a keylogger, but also screencaptures,
    we can extract much more useful information, being able to improve the process mining over said log.
    As a pending task, we need to validate this hypothesis through a results comparison against the non-enriched log.
    We expect to continue this project in later stages of the master
    """
    log_enriched.to_csv(enriched_log_output_path)
    print("\n\n=========== ENRICHED LOG GENERATED: path=" + enriched_log_output_path)



def location_ui_elements_fe_technique(
    ui_elements_classification_classes, 
    screenshot_colname, metadata_json_root, ui_log_path,
    enriched_log_output_path):
    print("TODO") # TODO: 

def location_ui_elements_and_plaintext_fe_technique(feature_extraction_technique_name, skip):
    print("TODO") # TODO: 


