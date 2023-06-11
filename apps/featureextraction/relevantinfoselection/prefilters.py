import os
import time
import json
import logging
import pandas as pd
from tqdm import tqdm
from art import tprint
from PIL import Image, ImageDraw
from core.settings import sep
from core.settings import platform_name, info_prefiltering_phase_name, sep
from core.utils import read_ui_log_as_dataframe

# TODO
def rectangle_prefilter():
    print("Not implemented yet :)")

def attention_screen_mapping(root_path, fixation_data, screenshot_filename, scale_factor):
    # Load the image and create a new black image of the same size
    image = Image.open(root_path + screenshot_filename)
    attention_mask = Image.new('L', image.size, 0)

    # Loop through each fixation point and draw a circle on the attention mask
    draw = ImageDraw.Draw(attention_mask)
    for fixation_point in fixation_data[screenshot_filename]['fixation_points']:
        x, y = map(float, fixation_point.split('#'))
        if 'dispersion' in fixation_data[screenshot_filename]['fixation_points'][fixation_point]:
            dispersion = float(fixation_data[screenshot_filename]['fixation_points'][fixation_point]['dispersion'])
            if not pd.isna(dispersion):
                radius = int(dispersion) * int(scale_factor)  # Scale the dispersion to a reasonable size for the circle
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            else:
                logging.info(fixation_point + " - (dispersion attr is nan) not mapped to attention area in screenshot " + screenshot_filename)
        else:
            logging.info(fixation_point + " - (no dispersion attr) not mapped to attention area in screenshot " + screenshot_filename)

    # Apply the attention mask to the original image
    attention_map = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), attention_mask)

    attention_path = root_path + 'prefilter_attention_maps'
    
    if not os.path.exists(attention_path):
        os.mkdir(attention_path)
    
    # Save the resulting attention map
    attention_map.save(attention_path + sep + screenshot_filename)

def attention_areas_prefilter(log_path, root_path, special_colnames, configurations, config_key):
    ui_log = read_ui_log_as_dataframe(log_path)
    # Load the fixation data
    with open(root_path + 'fixation.json', 'r') as f:
        fixation_data = json.load(f)
    for screenshot_filename in ui_log[special_colnames["Screenshot"]]:
        if screenshot_filename in fixation_data:
            attention_screen_mapping(root_path, fixation_data, screenshot_filename, configurations[config_key]["scale_factor"])
        else:
            logging.info(str(screenshot_filename) + " doesn't generate filtered screenshot. It doesn't have fixations related.")

def apply_prefilters(log_path, root_path, special_colnames, configurations):
    times = {}
    for key in tqdm(configurations, desc="Prefilters have been processed: "):
        # ui_selector = configurations[key]["UI_selector"]
        # predicate = configurations[key]["predicate"]
        # remove_nested = configurations[key]["remove_nested"]
        s = "and applied!"
        start_t = time.time()
        match key:
            case "gaze":
                attention_areas_prefilter(log_path, root_path, special_colnames, configurations, "gaze")
            case _:
                s = "but not executed. It's not one of the possible prefilters to apply!"
                pass
        
        print("Prefilter '" + key + "' detected " + s)
        logging.info("apps/featureextraction/filters.py Filter '" + key + "' detected " + s)
        times[key] = {"duration": float(time.time()) - float(start_t)}
    return times

def info_prefiltering(*data):
    data_list = list(data)
    filters_format_type = data_list.pop()
    skip = data_list.pop()
    data = tuple(data_list)
    if not skip:  
        tprint(platform_name + " - " + info_prefiltering_phase_name, "fancy60")
        
        match filters_format_type:
            case "rpa-us":
                output = apply_prefilters(*data)
            case _:
                raise Exception("You select a type of prefilter that doesnt exists")
    else:
        logging.info("Phase " + info_prefiltering_phase_name + " skipped!")
        output = "Phase " + info_prefiltering_phase_name + " skipped!"
    return output