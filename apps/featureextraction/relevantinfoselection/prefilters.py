import os
import time
import json
import logging
import pandas as pd
from tqdm import tqdm
from art import tprint
from PIL import Image, ImageDraw
from core.settings import sep
from core.settings import PLATFORM_NAME, INFO_PREFILTERING_PHASE_NAME, sep
from core.utils import read_ui_log_as_dataframe
from django.utils.translation import gettext_lazy as _

# TODO
def rectangle_prefilter():
    print("Not implemented yet :)")

def attention_screen_mapping(log_path,path_scenario, fixation_data, screenshot_filename):
    
    scenario_results = path_scenario + '_results'
    print(scenario_results)
    # Load the image and create a new black image of the same size
    image = Image.open(os.path.join(path_scenario,screenshot_filename))
    attention_mask = Image.new('L', image.size, 0)

    # Loop through each fixation point and draw a circle on the attention mask
    draw = ImageDraw.Draw(attention_mask)
    for fixation_point in fixation_data[screenshot_filename]['fixation_points']:
        x, y = map(float, fixation_point.split('#'))
        if 'dispersion' in fixation_data[screenshot_filename]['fixation_points'][fixation_point]:
            dispersion = float(fixation_data[screenshot_filename]['fixation_points'][fixation_point]['dispersion'])
            if not pd.isna(dispersion):
                # radius = dispersion * (1000*scale_factor)  # Scale the dispersion to a reasonable size for the circle
                radius = dispersion
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            else:
                logging.info(fixation_point + " - (dispersion attr is nan) not mapped to attention area in screenshot " + screenshot_filename)
        else:
            logging.info(fixation_point + " - (no dispersion attr) not mapped to attention area in screenshot " + screenshot_filename)

    # Apply the attention mask to the original image
    attention_map = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), attention_mask)

    # attention_path = os.path.join(scenario_results ,'prefilter_attention_maps')
    attention_path = os.path.join(scenario_results ,'prefiltered_img')
    print("atenttion_path: "+ str(attention_path))
    # attention_path = root_path + 'prefilter_attention_maps'
    
    if not os.path.exists(attention_path):
        os.makedirs(attention_path)
    
    # Save the resulting attention map
    attention_map.save(os.path.join(attention_path ,screenshot_filename))
    print(screenshot_filename+" prefiltered correctly and saved in: "+ str(attention_path))

def attention_areas_prefilter(log_path, path_scenario, special_colnames):
    ui_log = read_ui_log_as_dataframe(log_path)
    # Load the fixation data
    with open(os.path.join(path_scenario, 'fixation.json'), 'r') as f:
        fixation_data = json.load(f)
    for screenshot_filename in ui_log[special_colnames["Screenshot"]]:
        if screenshot_filename in fixation_data:
            attention_screen_mapping(log_path, path_scenario, fixation_data, screenshot_filename)
        else:
            logging.info(str(screenshot_filename) + " doesn't generate filtered screenshot. It doesn't have fixations related.")

def apply_prefilters(log_path, path_scenario, special_colnames):
    times = {}
    # for key in tqdm(configurations, desc="Prefilters have been processed: "):
    #     # ui_selector = configurations[key]["UI_selector"]
    #     # predicate = configurations[key]["predicate"]
    #     # remove_nested = configurations[key]["remove_nested"]
    #     s = "and applied!"
    #     start_t = time.time()
    #     match key:
    #         case "gaze":
    #             attention_areas_prefilter(log_path, path_scenario, special_colnames, configurations, "gaze")
    #         case _:
    #             s = "but not executed. It's not one of the possible prefilters to apply!"
    #             pass
        
    #     print("Prefilter '" + key + "' detected " + s)
    #     logging.info("apps/featureextraction/filters.py Filter '" + key + "' detected " + s)
    #     times[key] = {"duration": float(time.time()) - float(start_t)}
    print("Prefiltering phase started...")
    start_t = time.time()
    attention_areas_prefilter(log_path, path_scenario, special_colnames)
    print("Prefiltering phase finished satisfactory!")
    times["prefiltering"] = {"duration": float(time.time()) - float(start_t)}

    return times

def prefilters(log_path, path_scenario, execution):
    special_colnames = execution.case_study.special_colnames
    # scale_factor = execution.prefilters.scale_factor
    skip = execution.prefilters.preloaded
    
    if not skip:  
        tprint(PLATFORM_NAME + " - " + INFO_PREFILTERING_PHASE_NAME, "fancy60")
        # match filters_format_type:
        #     case "rpa-us":
        #         output = apply_prefilters(log_path, execution.exp_folder_complete_path, special_colnames, configurations)
        output = apply_prefilters(log_path, path_scenario, special_colnames)
        #     case _:
        #         raise Exception(_("You select a type of prefilter that doesnt exists"))
    else:
        logging.info("Phase " + INFO_PREFILTERING_PHASE_NAME + " skipped!")
        output = "Phase " + INFO_PREFILTERING_PHASE_NAME + " skipped!"
    return output