import os
import json
import logging
from tqdm import tqdm
from art import tprint
from PIL import Image, ImageDraw
from core.settings import sep
from core.settings import platform_name, info_prefiltering_phase_name, sep
from core.utils import read_ui_log_as_dataframe

# TODO
def rectangle_prefilter():
    print("Not implemented yet :)")

def attention_screen_mapping(root_path, screenshot_filename):
    # Load the fixation data
    with open(root_path + 'fixation.json', 'r') as f:
        fixation_data = json.load(f)[screenshot_filename]

    # Load the image and create a new black image of the same size
    image = Image.open(root_path + screenshot_filename)
    attention_mask = Image.new('L', image.size, 0)

    # Loop through each fixation point and draw a circle on the attention mask
    draw = ImageDraw.Draw(attention_mask)
    for fixation_point in fixation_data['fixation_points']:
        x, y = map(float, fixation_point.split('#'))
        dispersion = float(fixation_data['fixation_points'][fixation_point]['dispersion'])
        size = int(dispersion * 10)  # Scale the dispersion to a reasonable size for the circle
        draw.ellipse((x - size, y - size, x + size, y + size), fill=255)

    # Apply the attention mask to the original image
    attention_map = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), attention_mask)

    attention_path = root_path + 'attention_map_screenshots'
    
    if not os.path.exists(attention_path):
        os.mkdir(attention_path)
    
    # Save the resulting attention map
    attention_map.save(attention_path + sep + screenshot_filename)

def attention_areas_prefilter(log_path, root_path, special_colnames, configurations, config_key):
    ui_log = read_ui_log_as_dataframe(log_path)
    for screenshot_filename in ui_log[special_colnames["Screenshot"]]:
        attention_screen_mapping(root_path, screenshot_filename)

def apply_prefilters(log_path, root_path, special_colnames, configurations):
    for key in tqdm(configurations, desc="Prefilters have been processed: "):
        # ui_selector = configurations["key"]["UI_selector"]
        # predicate = configurations["key"]["predicate"]
        # remove_nested = configurations["key"]["remove_nested"]
        s = "and applied!"
        match key:
            case "gaze":
                attention_areas_prefilter(log_path, root_path, special_colnames, configurations, "gaze")
            case _:
                s = "but not executed. It's not one of the possible prefilters to apply!"
                pass
        
        print("Prefilter '" + key + "' detected " + s)
        logging.info("apps/featureextraction/filters.py Filter '" + key + "' detected " + s)

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