# Components detection
from os.path import join as pjoin
# Classification
from sklearn.linear_model import enet_path
from keras.models import model_from_json
from django.core.exceptions import ValidationError
import os
import pandas as pd
from core.settings import gaze_analysis_threshold
import pickle
from apps.featureextraction.CNN.CompDetCNN import CompDetCNN
from tqdm import tqdm
import urllib.parse
import json


def gaze_analysis(log_path, root_path, special_colnames, gaze_analysis_type, gaze_analysis_configurations):
    # eyetracking_log , log, img_index, image_names, special_colnames, timestamp_start, timestamp_end, last_upper_limit, init_value_ui_log_timestamp):
    
    log = pd.read_csv(log_path, sep=",")
    
    sep = gaze_analysis_configurations["separator"]
    
    eyetracking_log_filename = gaze_analysis_configurations["eyetracking_log_filename"]
    uilog_starttime = float(gaze_analysis_configurations["uilog_startdatetime"])
    
    if eyetracking_log_filename and os.path.exists(root_path + eyetracking_log_filename):
        gazeanalysis_log = pd.read_csv(root_path + eyetracking_log_filename, sep=sep)
    else:
        raise Exception("Eyetracking log cannot be read")
    

    if gaze_analysis_type == "imotions":
        data, metadata = decode_imotions_gaze_analysis(gazeanalysis_log)
        absolute_starttime = decode_imotions_native_slideevents(root_path, gaze_analysis_configurations["native_slide_events"], sep)
        

        gaze_events = {}  # key: row number,
        #value: { tuple: [coorX, coorY], gui_component_coordinate: [[corners_of_crop]]}

        last_upper_limit = 0
        
        # GAZE ANALYSIS
        if eyetracking_log is not False:
            timestamp_start = log[special_colnames['Timestamp']
                                    ][img_index]-init_value_ui_log_timestamp
            if img_index < len(image_names)-1:
                timestamp_end = log[special_colnames['Timestamp']
                                    ][img_index+1]-init_value_ui_log_timestamp
                interval, last_upper_limit = gaze_events_associated_to_event_time_range(
                    eyetracking_log,
                    special_colnames,
                    timestamp_start,
                    timestamp_end,
                    None)
            else:
                print("Function detect_images_components: LAST SCREENSHOT")
                interval, last_upper_limit = gaze_events_associated_to_event_time_range(
                    eyetracking_log,
                    special_colnames,
                    timestamp_start,
                    "LAST",
                    last_upper_limit)

            # { row_number: [[gaze_coorX, gaze_coorY],[gaze_coorX, gaze_coorY],[gaze_coorX, gaze_coorY]]}
            gaze_events[img_index] = interval

    return gaze_events


def gaze_events_associated_to_event_time_range(eyetracking_log, colnames, timestamp_start, timestamp_end, last_upper_limit):
    # timestamp starts from 0
    eyetracking_log_timestamp = eyetracking_log[colnames['eyetracking_recording_timestamp']]
    if eyetracking_log_timestamp[0] != 0:
        raise ValidationError(
            "Recording timestamp in eyetracking log must starts from 0")

    lower_limit = 0
    upper_limit = 0

    if timestamp_end != "LAST":
        for index, time in enumerate(eyetracking_log_timestamp):
            if time > timestamp_start:
                lower_limit = eyetracking_log_timestamp[index-1]
                if time >= timestamp_end:
                    upper_limit = eyetracking_log_timestamp[index]
                    break
    else:
        upper_limit = len(eyetracking_log_timestamp)
        lower_limit = last_upper_limit

    return eyetracking_log.loc[lower_limit:upper_limit, [colnames['eyetracking_gaze_point_x'], colnames['eyetracking_gaze_point_y']]], upper_limit


def decode_imotions_gaze_analysis(gazeanalysis_log):
    # Find row index where "#DATA" is located
    data_index = gazeanalysis_log.index[gazeanalysis_log.iloc[:, 0] == '#DATA'][0]

    # Split dataframe into two separate dataframes
    metadata_df = gazeanalysis_log.iloc[:data_index, :]
    data_df = gazeanalysis_log.iloc[data_index:, :]
    
    return data_df, metadata_df

def decode_imotions_native_slideevents(native_slideevents_path, native_slideevents_filename, sep):
    # Leer el archivo completo
    with open(native_slideevents_path + native_slideevents_filename, 'r') as file:
        data = file.readlines()
    # Encontrar los índices donde están las etiquetas #METADATA y #DATA
    metadata_index = data.index('#METADATA\n')
    encodedStr = data[metadata_index-1:metadata_index][0]
    native_properties= urllib.parse.unquote(encodedStr)
    native_properties= json.loads(native_properties)
    with open(native_slideevents_path + "native_properties.json", 'w') as f:
            json.dump(native_properties, f, indent=4)
    # data_index = data.index('#DATA\n')
    # # Crear dos listas separadas para los metadatos y los datos
    # metadata = [line.strip().split(sep) for line in data[metadata_index+1:data_index]]
    # data_csv = [line.strip().split(sep) for line in data[data_index+1:]]
    # # Convertir la lista de datos en un dataframe de pandas
    # df = pd.DataFrame(data_csv[1:], columns=data_csv[0])
    return native_properties["SlideShowStartDateTime"]