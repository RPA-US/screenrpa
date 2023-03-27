# Components detection
from os.path import join as pjoin
# Classification
from sklearn.linear_model import enet_path
from keras.models import model_from_json
from django.core.exceptions import ValidationError
from core.settings import gaze_analysis_threshold
import pickle
from apps.featureextraction.CNN.CompDetCNN import CompDetCNN
from tqdm import tqdm


def gaze_fixation(log_path, root_path, special_colnames, gaze_fixation_type, gaze_fixation_configurations):
    # eyetracking_log , log, img_index, image_names, special_colnames, timestamp_start, timestamp_end, last_upper_limit, init_value_ui_log_timestamp):
    
    # eyetracking_log = False
    # if eyetracking_log_filename and os.path.exists(param_img_root + eyetracking_log_filename):
    #     eyetracking_log = pd.read_csv(
    #         param_img_root + eyetracking_log_filename, sep=";")
    # init_value_ui_log_timestamp = log[special_colnames['Timestamp']][0]

    gaze_events = {}  # key: row number,
    #value: { tuple: [coorX, coorY], gui_component_coordinate: [[corners_of_crop]]}

    last_upper_limit = 0
    
    gaze_events = {}
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