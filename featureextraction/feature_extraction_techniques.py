
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def quantity_ui_elements_fe_technique(ui_elements_classification_classes, decision_point, case_colname, activity_colname, screenshot_colname,
                                      metadata_json_root, flattened_log, ui_log_path, enriched_log_output, text_classname):
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
    id = "quantity"
    with open(flattened_log, 'r') as f:
        ui_log_data = json.load(f)
    
    log = pd.read_csv(ui_log_path, sep=",")
    # df = pd.DataFrame([], columns=ui_elements_classification_classes)
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    quantity_ui_elements = {}
    before_DP = True
    aux_case = -1    

    for i, screenshot_filename in enumerate(screenshot_filenames):
        case = log.at[i, case_colname]
        activity = log.at[i, activity_colname]
        
        if case != aux_case:
            before_DP = True
        
        if before_DP:
            # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
            with open(metadata_json_root + screenshot_filename + '.json', 'r') as f:
                data = json.load(f)

            for c in ui_elements_classification_classes:
                counter = 0
                for j in range(0, len(data["compos"])):
                    if data["compos"][j]["class"] == c:
                        counter+=1
                quantity_ui_elements[c] = counter
                ui_log_data[str(case)][id+"_"+c+"_"+activity] = counter

            if "features" in data:
                data["features"][id] = quantity_ui_elements
            else:
                data["features"] = { id: quantity_ui_elements }
            
            # df.loc[i] = quantity_ui_elements.values()
    
                
            with open(metadata_json_root + screenshot_filename + '.json', "w") as jsonFile:
                json.dump(data, jsonFile, indent=4)
                
            if activity == decision_point:
                aux_case = case
                before_DP = False

    with open(flattened_log, 'w') as f:
        json.dump(ui_log_data, f, indent=4)
        
    """
    Once the dataset corresponding to the ammount of elements of each class contained in each of the images is obtained,
    we merge it with the complete log, adding the extracted characteristics from the images
    """

    # log_enriched = log.join(df).fillna(method='ffill')

    """
    Finally we obtain an entiched log, which is turned as proof of concept of our hypothesis based on the premise that if
    we not only capture keyboard or mouse events on the monitorization through a keylogger, but also screencaptures,
    we can extract much more useful information, being able to improve the process mining over said log.
    As a pending task, we need to validate this hypothesis through a results comparison against the non-enriched log.
    We expect to continue this project in later stages of the master
    """
    # log_enriched.to_csv(enriched_log_output)
    print("\n\n=========== ENRICHED LOG GENERATED: path=" + enriched_log_output)



def location_ui_elements_fe_technique(ui_elements_classification_classes, decision_point, 
    case_colname, activity_colname, screenshot_colname, metadata_json_root, flattened_log, ui_log_path, enriched_log_output, text_classname):

    log = pd.read_csv(ui_log_path, sep=",")
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    headers = dict()
    info_to_join: dict[str:list] = {}

    for elem in ui_elements_classification_classes:
        headers[elem] = 0

    for i, screenshot_filename in enumerate(screenshot_filenames):
        # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
        with open(metadata_json_root + screenshot_filename + '.json', 'r') as f:
            data = json.load(f)

        screenshot_compos_frec = headers.copy()

        for j in range(0, len(data["compos"])):
            compo_class = data["compos"][j]["class"]
            compo_x1 = data["compos"][j]["column_min"]
            compo_y1 = data["compos"][j]["row_min"]
            compo_x2 = data["compos"][j]["column_max"]
            compo_y2 = data["compos"][j]["row_max"]
            centroid_y = (compo_y2 - compo_y1 / 2) + compo_y1
            centroid_x = (compo_x2 - compo_x1 / 2) + compo_x1
            data["compos"][j]["centroid"] = [centroid_x, centroid_y]
            screenshot_compos_frec[compo_class] += 1
            
            column_name = compo_class+"_"+str(screenshot_compos_frec[compo_class])

            if column_name in info_to_join:
                if not len(info_to_join[column_name]) == i:
                    for k in range(len(info_to_join[column_name]),i):
                        info_to_join[column_name].append("")
                info_to_join[column_name].append(data["compos"][j]["centroid"])
            else:
                column_as_vector = []
                for k in range(0,i):
                    column_as_vector.append("")
                column_as_vector.append(data["compos"][j]["centroid"])
                info_to_join[column_name] = column_as_vector
                
        if "features" in data:
            data["features"]["location"] = info_to_join
        else:
            data["features"] = { "location": info_to_join }
        
        with open(metadata_json_root + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(data, jsonFile)

    for column_name in info_to_join:
        if not len(info_to_join[column_name]) == len(screenshot_filenames):
            for k in range(len(info_to_join[column_name]),len(screenshot_filenames)):
                info_to_join[column_name].append("")

    df = pd.DataFrame([])
    keys = list(info_to_join.keys())
    keys.sort()
    for column in keys:
        df.insert(loc=len(df.columns), column=column, value=info_to_join[column])

    log_enriched = log.join(df).fillna(method='ffill')
    log_enriched.to_csv(enriched_log_output)

    print("\n\n=========== ENRICHED LOG GENERATED: path=" + enriched_log_output)

def location_ui_elements_and_plaintext_fe_technique(ui_elements_classification_classes, decision_point, 
    case_colname, activity_colname, screenshot_colname, metadata_json_root, flattened_log, ui_log_path, enriched_log_output_path, text_classname):

    log = pd.read_csv(ui_log_path, sep=",")
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    headers = dict()
    info_to_join: dict[str:list] = {}

    for elem in ui_elements_classification_classes:
        headers[elem] = 0

    for i, screenshot_filename in enumerate(screenshot_filenames):
        # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
        with open(metadata_json_root + screenshot_filename + '.json', 'r') as f:
            data = json.load(f)

        screenshot_compos_frec = headers.copy()

        for j in range(0, len(data["compos"])):
            compo_class = data["compos"][j]["class"]
            compo_x1 = data["compos"][j]["column_min"]
            compo_y1 = data["compos"][j]["row_min"]
            compo_x2 = data["compos"][j]["column_max"]
            compo_y2 = data["compos"][j]["row_max"]
            centroid_y = (compo_y2 - compo_y1 / 2) + compo_y1
            centroid_x = (compo_x2 - compo_x1 / 2) + compo_x1
            data["compos"][j]["centroid"] = [centroid_x, centroid_y]
            screenshot_compos_frec[compo_class] += 1
            
            if compo_class == text_classname:
                column_name = compo_class+"_"+str(screenshot_compos_frec[compo_class])
            else:
                column_name = compo_class+"_"+str(screenshot_compos_frec[compo_class])

            if column_name in info_to_join:
                if not len(info_to_join[column_name]) == i:
                    for k in range(len(info_to_join[column_name]),i):
                        info_to_join[column_name].append("")
                info_to_join[column_name].append(data["compos"][j]["centroid"])
            else:
                column_as_vector = []
                for k in range(0,i):
                    column_as_vector.append("")
                column_as_vector.append(data["compos"][j]["centroid"])
                info_to_join[column_name] = column_as_vector
                
        with open(metadata_json_root + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(data, jsonFile)

    for column_name in info_to_join:
        if not len(info_to_join[column_name]) == len(screenshot_filenames):
            for k in range(len(info_to_join[column_name]),len(screenshot_filenames)):
                info_to_join[column_name].append("")

    df = pd.DataFrame([])
    keys = list(info_to_join.keys())
    keys.sort()
    for column in keys:
        df.insert(loc=len(df.columns), column=column, value=info_to_join[column])

    log_enriched = log.join(df).fillna(method='ffill')
    log_enriched.to_csv(enriched_log_output_path+"location_enriched_log.csv")

    print("\n\n=========== ENRICHED LOG GENERATED: path=" + enriched_log_output_path + "location_enriched_log.csv")


