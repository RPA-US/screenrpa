# Components detection
from genericpath import exists
import json
import numpy as np
import keras_ocr
import cv2
import pandas as pd
from os.path import join as pjoin
import os
from PIL import Image
import featureextraction.utils as utils
# Classification
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import enet_path
import tensorflow as tf
from keras.models import model_from_json
from django.core.exceptions import ValidationError
from rim.settings import cropping_threshold, sep
from rim.settings import gaze_analysis_threshold
import pickle
from featureextraction.CNN.CompDetCNN import CompDetCNN
from tqdm import tqdm


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