from core.settings import platform_name, info_filtering_phase_name
from art import tprint


def apply_selectors(configurations, skip):
    if not skip:
        for key in configurations:
            print("Selector with name '" + key + "' being applied ...")

def info_filtering(*data):
    data_list = list(data)
    selectors_format_type = data_list.pop()
    data = tuple(data_list)
    
    tprint(platform_name + " - " + info_filtering_phase_name, "fancy60")
    
    match selectors_format_type:
        case "rpa-us":
            output = apply_selectors(*data)
        case _:
            raise Exception("You select a type of UI element classification that doesnt exists")
    return output