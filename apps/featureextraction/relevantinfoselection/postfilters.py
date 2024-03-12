import json
import os
import cv2
import logging
import time
import math
import pandas as pd
from art import tprint
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, box
from core.utils import read_ui_log_as_dataframe
from core.settings import PLATFORM_NAME, INFO_POSTFILTERING_PHASE_NAME, sep
from apps.featureextraction.utils import draw_geometry_over_image
from django.utils.translation import gettext_lazy as _

def calculate_intersection_area_v1(rect_x, rect_y, rect_width, rect_height, circle_x, circle_y, circle_radius):
    # Calcular las coordenadas del rectángulo y el círculo
    rect_left = rect_x
    rect_right = rect_x + rect_width
    rect_top = rect_y
    rect_bottom = rect_y + rect_height
    circle_left = circle_x - circle_radius
    circle_right = circle_x + circle_radius
    circle_top = circle_y - circle_radius
    circle_bottom = circle_y + circle_radius
    
    # Calcular las coordenadas de la intersección
    x_inter = max(rect_left, circle_left)
    y_inter = max(rect_top, circle_top)
    width_inter = min(rect_right, circle_right) - x_inter
    height_inter = min(rect_bottom, circle_bottom) - y_inter
    
    # Verificar si hay intersección
    if width_inter < 0 or height_inter < 0:
        return 0
    
    # Calcular el área de la intersección
    if width_inter < height_inter:
        radius_inter = width_inter / 2
    else:
        radius_inter = height_inter / 2
    
    area_inter = math.pi * radius_inter**2
    
    return area_inter

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


def calculate_intersection_area_v2(compo, fixation_point_x, fixation_point_y, circle_radius):
    # compo_area = (compo['row_max'] - compo['row_min']) * (compo['column_max'] - compo['column_min'])
    x_min = compo["row_min"]
    x_max = compo["row_max"]
    y_min = compo["column_min"]
    y_max = compo["column_max"]
    intersection_area = 0
    
    if fixation_point_x < x_min:
        closest_x = x_min
    elif fixation_point_x > x_max:
        closest_x = x_max
    else:
        closest_x = fixation_point_x
    
    if fixation_point_y < y_min:
        closest_y = y_min
    elif fixation_point_y > y_max:
        closest_y = y_max
    else:
        closest_y = fixation_point_y
    
    if (fixation_point_x >= x_min and fixation_point_x <= x_max and
            fixation_point_y >= y_min and fixation_point_y <= y_max):
        intersection_area = circle_radius**2
    else:
        distance = ((fixation_point_x - closest_x)**2 + (fixation_point_y - closest_y)**2)**0.5
        if distance < circle_radius:
            intersection_area = (circle_radius**2 * math.acos(distance / circle_radius) -
                                 distance * (circle_radius**2 - distance**2)**0.5)
    
    return intersection_area

def is_component_relevant_v2(compo, fixation_point, fixation_point_x, fixation_point_y, scale_factor, previous_coincident_areas):
    compo_area = (compo['row_max'] - compo['row_min']) * (compo['column_max'] - compo['column_min'])
    
    circle_radius = fixation_point["dispersion"] * int(scale_factor)
    intersection_area = calculate_intersection_area_v2(compo, fixation_point_x, fixation_point_y, circle_radius)
    res = intersection_area / compo_area
    return res


def gaze_filtering(log_path, root_path, special_colnames, configurations, key):
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
            print("screenshots_json: ", screenshot_json)
        
        if configurations[key]["predicate"] == "is_component_relevant":
            polygon_circles = []
            polygon_rectangles = []
            for fixation_point in fixation_json[screenshot_filename]["fixation_points"]:
                fixation_coordinates = fixation_point.split("#")
                fixation_obj = fixation_json[screenshot_filename]["fixation_points"][fixation_point]
                fixation_point_x = float(fixation_coordinates[0])
                fixation_point_y = float(fixation_coordinates[1])     
                centre = Point(fixation_point_x, fixation_point_y)
                radio = fixation_obj["dispersion"] * float(configurations[key]["scale_factor"])
                if not pd.isna(radio):
                    polygon_circle = centre.buffer(radio)
                    polygon_circles.append(polygon_circle)

                fixation_mask = unary_union(polygon_circles)
          
            for compo in screenshot_json["compos"]:
                compo["relevant"] = "NaN"
                # compo matches UI selector
                if (configurations[key]["UI_selector"] == "all" or (compo["category"] in configurations[key]["UI_selector"])) and + \
                    (screenshot_filename in fixation_json): # screenshot has fixation
                        
                    x_min = compo['row_min']
                    y_min = compo['column_min']
                    x_max = compo['row_max']
                    y_max = compo['column_max']
                    
                    polygon_rect = box(x_min, y_min, x_max, y_max)
                    intersection = polygon_rect.intersection(fixation_mask)

                    compo["intersection_area"] = intersection.area
                    
                    if polygon_rect.area > 0 and (intersection.area / polygon_rect.area) > float(configurations[key]["intersection_area_thresh"]):
                        nested = bool(configurations[key]["only_leaf"])
                        if nested and (len(compo["contain"]) > 0 or compo["label"] == "UIGroup"):
                            compo["relevant"] = "Nested"
                            if configurations[key]["consider_nested_as_relevant"] == "True":
                                polygon_rectangles.append(polygon_rect)
                        else:
                            compo["relevant"] = "True"
                            polygon_rectangles.append(polygon_rect)
                            
                    else:
                        compo["relevant"] = "False"
                                    
                else:
                    compo["relevant"] = "False"
                    
            if configurations[key]["mode"] == "draw":
                if not os.path.exists(root_path + "postfilter_attention_maps" + sep):
                    os.makedirs(root_path + "postfilter_attention_maps" + sep)
                draw_geometry_over_image(root_path + screenshot_filename, polygon_circles, polygon_rectangles, root_path + "postfilter_attention_maps" + sep + screenshot_filename)
            
        else:
            for compo in screenshot_json["compos"]:
                compo["relevant"] = "NaN"
                # compo matches UI selector
                if (configurations[key]["UI_selector"] == "all" or (compo["category"] in configurations[key]["UI_selector"])) and + \
                    (screenshot_filename in fixation_json): # screenshot has fixation
                        
                    last_intersection_area = 0
                    previous_coincident_areas = []
                    polygons = []
                    
                    for fixation_point in fixation_json[screenshot_filename]["fixation_points"]:
                        fixation_coordinates = fixation_point.split("#")
                        fixation_obj = fixation_json[screenshot_filename]["fixation_points"][fixation_point]
                        fixation_point_x = float(fixation_coordinates[0])
                        fixation_point_y = float(fixation_coordinates[1])                    
                        # predicate: "(compo['row_min'] <= fixation_point_x) and (fixation_point_x <= compo['row_max']) and (compo['column_min'] <= fixation_point_y) and (fixation_point_y <= compo['column_max'])"
                        # predicate: is_component_relevant_option2(compo, fixation_obj, fixation_point_x, fixation_point_y)

                        if compo["relevant"] == "True" or compo["relevant"] == "Nested":
                            pass
                        else:
                            cond = eval(configurations[key]["predicate"])
                            
                            if cond:
                                nested = bool(configurations[key]["only_leaf"])
                                if nested and (len(compo["contain"]) > 0):
                                    compo["relevant"] = "Nested"
                                else:
                                    compo["relevant"] = "True"
                            else:
                                compo["relevant"] = "False"
                                
                    if configurations[key]["mode"] == "draw":
                        if not os.path.exists(root_path + "postfilter_attention_maps/"):
                            os.makedirs(root_path + "postfilter_attention_maps/")
                        draw_geometry_over_image(root_path + screenshot_filename, polygons, root_path + "postfilter_attention_maps/" + screenshot_filename)
                        
                else:
                    compo["relevant"] = "False"

        with open(root_path + "components_json" + sep + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(screenshot_json, jsonFile, indent=4)
    print("apps/featureextraction/postfilters.py Postfilter '" + key + "' finished!!")
    logging.info("apps/featureextraction/postfilters.py Postfilter '" + key + "' finished!!")

def apply_filters(log_path, root_path, execution):
    
    special_colnames = execution.case_study.special_colnames
    configurations = execution.postfilters.configurations
    
    times = {}
    for key in tqdm(configurations, desc="Postfilters have been processed: "):
        # ui_selector = configurations["key"]["UI_selector"]
        # predicate = configurations["key"]["predicate"]
        # remove_nested = configurations["key"]["remove_nested"]
        s = "and applied!"
        start_t = time.time()
        match key:
            case "gaze":
                gaze_filtering(log_path, root_path, special_colnames, configurations, "gaze")
            case _:
                s = "but not executed. It's not one of the possible filters to apply!"
                pass
        
        print("Filter '" + key + "' detected " + s)
        logging.info("apps/featureextraction/filters.py Filter '" + key + "' detected " + s)
        times[key] = {"duration": float(time.time()) - float(start_t)}

    return times

def postfilters(log_path, root_path, execution):
    filters_format_type = execution.postfilters.type
    skip = execution.postfilters.preloaded
    
    if not skip:  
        tprint(PLATFORM_NAME + " - " + INFO_POSTFILTERING_PHASE_NAME, "fancy60")
        
        match filters_format_type:
            case "rpa-us":
                output = apply_filters(log_path, root_path, execution)
            case _:
                raise Exception(_("You select a type of filter that doesnt exists"))
    else:
        logging.info("Phase " + INFO_POSTFILTERING_PHASE_NAME + " skipped!")
        output = "Phase " + INFO_POSTFILTERING_PHASE_NAME + " skipped!"
    return output

def draw_postfilter_relevant_ui_compos_borders(exp_path):
    root_path = exp_path + "_results" + sep + "components_json" + sep
    arr = os.listdir(root_path)

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
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
        output_path = exp_path + "_results" + sep + "compo_json_borders" + sep + compo_json_filename[:19]  # Replace with your desired output file path
        cv2.imwrite(output_path, image)
        image = None


        with open(root_path + compo_json_filename, "w") as outfile:
            json.dump(compo_json, outfile, indent=4)