import json
import os
import cv2
import logging
import time
import math
from tqdm import tqdm
from art import tprint
from core.utils import read_ui_log_as_dataframe
from core.settings import platform_name, info_postfiltering_phase_name, sep

import math

def is_component_relevant_v1(compo, fixation_point, fixation_point_x, fixation_point_y):
    # Calcular el radio del círculo de fijación
    dispersion = fixation_point["dispersion"]
    
    # Calcular el centro del círculo de fijación
    circle_center_x = fixation_point_x
    circle_center_y = fixation_point_y
    
    # Calcular el área del círculo de fijación
    fixation_area = math.pi * dispersion**2
    
    # Obtener las coordenadas del bounding box del UI compo
    row_min, column_min = compo['row_min'], compo['column_min']
    row_max, column_max = compo['row_max'], compo['column_max']
    
    # Calcular el área del UI compo
    compo_area = (row_max - row_min) * (column_max - column_min)
    
    # Calcular las coordenadas del área de intersección
    intersection_x1 = max(column_min, circle_center_x - dispersion)
    intersection_x2 = min(column_max, circle_center_x + dispersion)
    intersection_y1 = max(row_min, circle_center_y - dispersion)
    intersection_y2 = min(row_max, circle_center_y + dispersion)
    
    # Calcular el área de intersección
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    
    # Calcular el área coincidente teniendo en cuenta la diferencia entre figuras geométricas
    if intersection_area == 0:
        return False
    
    rect_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    circle_area = math.pi * min(dispersion, (intersection_x2 - intersection_x1)/2)**2
    
    overlap_percentage = (rect_area - circle_area) / compo_area
    
    # Verificar si el porcentaje de área coincidente supera el 50%
    if overlap_percentage > 0.5:
        return True
    else:
        return False


def is_component_relevant_v2(compo, fixation_point, fixation_point_x, fixation_point_y):
    compo_area = (compo['row_max'] - compo['row_min']) * (compo['column_max'] - compo['column_min'])
    
    circle_x = fixation_point_x
    circle_y = fixation_point_y
    circle_radius = fixation_point["dispersion"]
    
    intersection_area = 0
    
    if circle_x < compo['row_min']:
        closest_x = compo['row_min']
    elif circle_x > compo['row_max']:
        closest_x = compo['row_max']
    else:
        closest_x = circle_x
    
    if circle_y < compo['column_min']:
        closest_y = compo['column_min']
    elif circle_y > compo['column_max']:
        closest_y = compo['column_max']
    else:
        closest_y = circle_y
    
    if (circle_x >= compo['row_min'] and circle_x <= compo['row_max'] and
            circle_y >= compo['column_min'] and circle_y <= compo['column_max']):
        intersection_area = circle_radius**2
    else:
        distance = ((circle_x - closest_x)**2 + (circle_y - closest_y)**2)**0.5
        if distance < circle_radius:
            intersection_area = (circle_radius**2 * math.acos(distance / circle_radius) -
                                 distance * (circle_radius**2 - distance**2)**0.5)
    
    return intersection_area / compo_area > 0.5

def gaze_filering(log_path, root_path, special_colnames, configurations, key):
    """
    This function remove from 'screenshot000X.JPG' the compos that do not receive attention by the user (there's no fixation point that matches compo area)
    """
    ui_log = read_ui_log_as_dataframe(log_path) 
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
            # compo matches UI selector
            if (configurations[key]["UI_selector"] == "all" or (compo["category"] in configurations[key]["UI_selector"])) and + \
                (screenshot_filename in fixation_json): # screenshot has fixation
                for fixation_point in fixation_json[screenshot_filename]["fixation_points"]:
                    fixation_coordinates = fixation_point.split("#")
                    fixation_obj = fixation_json[screenshot_filename]["fixation_points"][fixation_point]
                    fixation_point_x = float(fixation_coordinates[0])
                    fixation_point_y = float(fixation_coordinates[1])
                    # predicate: "(compo['row_min'] <= fixation_point_x) and (fixation_point_x <= compo['row_max']) and (compo['column_min'] <= fixation_point_y) and (fixation_point_y <= compo['column_max'])"
                    # predicate: is_component_relevant_v2(compo, fixation_obj, fixation_point_x, fixation_point_y)
                    
                    if eval(configurations[key]["predicate"]):
                        if configurations[key]["only_leaf"] and (len(compo["contain"]) > 0):
                            compo["relevant"] = "Nested"
                        else:
                            compo["relevant"] = "True"
                    else:
                        compo["relevant"] = "False"
            else:
                compo["relevant"] = "False"

        with open(root_path + "components_json" + sep + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(screenshot_json, jsonFile, indent=4)

def apply_filters(log_path, root_path, special_colnames, configurations):
    times = {}
    for key in tqdm(configurations, desc="Postfilters have been processed: "):
        # ui_selector = configurations["key"]["UI_selector"]
        # predicate = configurations["key"]["predicate"]
        # remove_nested = configurations["key"]["remove_nested"]
        s = "and applied!"
        start_t = time.time()
        match key:
            case "gaze":
                gaze_filering(log_path, root_path, special_colnames, configurations, "gaze")
            case _:
                s = "but not executed. It's not one of the possible filters to apply!"
                pass
        
        print("Filter '" + key + "' detected " + s)
        logging.info("apps/featureextraction/filters.py Filter '" + key + "' detected " + s)
        times[key] = {"duration": float(time.time()) - float(start_t)}

    return times

def info_postfiltering(*data):
    data_list = list(data)
    filters_format_type = data_list.pop()
    skip = data_list.pop()
    data = tuple(data_list)
    if not skip:  
        tprint(platform_name + " - " + info_postfiltering_phase_name, "fancy60")
        
        match filters_format_type:
            case "rpa-us":
                output = apply_filters(*data)
            case _:
                raise Exception("You select a type of filter that doesnt exists")
    else:
        logging.info("Phase " + info_postfiltering_phase_name + " skipped!")
        output = "Phase " + info_postfiltering_phase_name + " skipped!"
    return output

def draw_postfilter_relevant_ui_compos_borders(exp_path):
    root_path = exp_path + sep + "components_json" + sep
    arr = os.listdir(root_path)

    if not os.path.exists(exp_path + sep + "compo_json_borders"):
        os.mkdir(exp_path + sep + "compo_json_borders")
    
    for compo_json_filename in arr:
        with open(root_path + compo_json_filename, 'r') as f:
            print(compo_json_filename)
            compo_json = json.load(f)
            
        # Load image
        image = cv2.imread(exp_path + sep + compo_json_filename[:19])
        
        
        for compo in compo_json["compos"]:
            if ("relevant" in compo) and compo["relevant"] == "True":
                # Extract component properties
                column_min = int(compo['column_min'])
                row_min = int(compo['row_min'])
                column_max = int(compo['column_max'])
                row_max = int(compo['row_max'])

                # Define border color based on compo ID
                color_id = compo['id'] % 3  # Assuming 3 different colors
                if color_id == 0:
                    border_color = (255, 0, 0)  # Blue
                elif color_id == 1:
                    border_color = (0, 255, 0)  # Green
                else:
                    border_color = (0, 0, 255)  # Red

                # Draw border rectangle on the image
                cv2.rectangle(image, (row_min, column_min), (row_max, column_max), border_color, 2)

            
        # Save the image with component borders
        output_path = exp_path + sep + "compo_json_borders" + sep + compo_json_filename[:19]  # Replace with your desired output file path
        cv2.imwrite(output_path, image)
        image = None


        with open(root_path + compo_json_filename, "w") as outfile:
            json.dump(compo_json, outfile, indent=4)