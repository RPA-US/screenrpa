from core.settings import platform_name, info_filtering_phase_name, sep
from art import tprint
import pandas as pd
from tqdm import tqdm
import json
import logging

def gaze_filering(log_path, root_path, special_colnames):
    """
    This function remove from 'screenshot000X.JPG' the compos that do not receive attention by the user (there's no fixation point that matches compo area)
    """
    ui_log = pd.read_csv(log_path, sep=",") 
    # fixation.json -> previous, screenshot001, screenshot002, screenshot003, ... , subsequent
    with open(root_path + 'fixation.json', 'r') as f:
        fixation_json = json.load(f)
    
    for screenshot_filename in ui_log[special_colnames["Screenshot"]]:
        # screenshot.json -> compos: [
        #     "column_min": 0,
        #     "row_min": 5,
        #     "column_max": 5,
        #     "row_max": 24,
        # ]
        with open(root_path + "components_json" + sep + screenshot_filename + '.json', 'r') as f:
            screenshot_json = json.load(f)
        
        for compo in screenshot_json["compos"]:
            if screenshot_filename in fixation_json:
                for fixation_point in fixation_json[screenshot_filename]["fixation_points"]:
                    fixation_coordinates = fixation_point.split("#")
                    x_coord = float(fixation_coordinates[0])
                    y_coord = float(fixation_coordinates[1])
                    x_cond = (compo["row_min"] <= x_coord) and (x_coord <= compo["row_max"])
                    y_cond = (compo["column_min"] <= y_coord) and (y_coord <= compo["column_max"])
                    if x_cond and y_cond:
                        compo["relevant"] = True
                    else:
                        compo["relevant"] = False
            else:
                compo["relevant"] = False
                
        with open(root_path + "components_json" + sep + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(screenshot_json, jsonFile, indent=4)

def apply_filters(log_path, root_path, special_colnames, configurations, skip):
    if not skip:
        for key in tqdm(configurations, desc="Filters have been processed: "):
            # ui_selector = configurations["key"]["UI_selector"]
            # predicate = configurations["key"]["predicate"]
            # remove_nested = configurations["key"]["remove_nested"]
            s = "and applied!"
            match key:
                case "gaze":
                    gaze_filering(log_path, root_path, special_colnames)
                case _:
                    s = "but not executed. It's not one of the possible filters to apply!"
                    pass
            
            print("Filter '" + key + "' detected " + s)
            logging.info("apps/featureextraction/filters.py Filter '" + key + "' detected " + s)


def info_filtering(*data):
    data_list = list(data)
    filters_format_type = data_list.pop()
    data = tuple(data_list)
    
    tprint(platform_name + " - " + info_filtering_phase_name, "fancy60")
    
    match filters_format_type:
        case "rpa-us":
            output = apply_filters(*data)
        case _:
            raise Exception("You select a type of UI element classification that doesnt exists")
    return output