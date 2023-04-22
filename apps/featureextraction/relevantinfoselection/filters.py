from core.settings import platform_name, info_filtering_phase_name
from art import tprint
import pandas as pd
from tqdm import tqdm

def apply_filters(log_path, root_path, special_colnames, configurations, skip):
    if not skip:
        for key in tqdm(configurations, desc="Filters have been processed: "):
            print("Filter '" + key + "' being applied")
            # ui_selector = configurations["key"]["UI_selector"]
            # predicate = configurations["key"]["predicate"]
            # remove_nested = configurations["key"]["remove_nested"]
            
            ui_log = pd.read_csv(log_path, sep=",")
            

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