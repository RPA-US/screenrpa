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
# import sys
# from numpy.lib.function_base import append
# from django.shortcuts import render
# from sklearn.utils.multiclass import unique_labels
# import matplotlib.image as mpimg
# import seaborn as sns
# import itertools
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from keras import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import SGD, Adam
# from keras.callbacks import ReduceLROnPlateau
# from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
# from PIL import Image
# from tensorflow.keras.applications import VGG19  # For Transfer Learning
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder

# Create your views here.

"""
Text boxes detection: KERAS_OCR

In order to detect the text boxes inside the screenshots, we define the get_keras_ocr_image.
This function has a list of images as input, and a list of lists with the coordinates of text boxes coordinates
"""

def get_ocr_image(pipeline, param_img_root, images_input):
    """
    Applies Keras-OCR over the input image or images to extract plain text and the coordinates corresponding
    to the present words

    :param pipeline: keras pipeline
    :type pipeline: keras pipeline
    :param param_img_root: Path where the imaages associated to each log row are stored
    :type param_img_root: str
    :param images_input: Path or list of paths of the image/s to process
    :type images_input: str or list
    :returns: List of lists corresponding the words identified in the input. Example: ('delete', array([[1161.25,  390.  ], [1216.25,  390.  ], [1216.25,  408.75], [1161.25,  408.75]], dtype=float32))
    :rtype: list
    """

    if not isinstance(images_input, list):
        # print("Solamente una imagen como entrada")
        images_input = [images_input]

    # Get a set of three example images
    images = [
        keras_ocr.tools.read(param_img_root + path) for path in images_input
    ]
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)
    # Plot the predictions
    # fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    # for ax, image, predictions in zip(axs, images, prediction_groups):
    #     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    return prediction_groups


def nesting_inspection(org, grey, compos, ffl_block):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:
            replace = False
            clip_grey = compo.compo_clipping(grey)
            n_compos = utils.nested_components_detection(
                clip_grey, org, grad_thresh=ffl_block, show=False)

            for comp in n_compos:
                comp.compo_relative_position(
                    compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def get_uied_gui_components_crops(input_imgs_path, image_names, img_index):
    '''
    Analyzes an image and extracts its UI components with an alternative algorithm

    :param param_img_root: Path to the image
    :type param_img_root: str
    :param image_names: Names of the images in the path
    :type image_names: list
    :param img_index: Index of the image we want to analyze in images_names
    :type img_index: int
    :return: List of image crops and Dict object with components detected
    :rtype: Tuple
    '''
    resize_by_height = 800
    input_img_path = pjoin(input_imgs_path, image_names[img_index])

    uied_params = {
        'min-grad': 3,
        'ffl-block': 5,
        'min-ele-area': 25,
        'merge-contained-ele': True,
        'max-word-inline-gap': 4,
        'max-line-gap': 4
    }

    # ##########################
    # COMPONENT DETECTION
    # ##########################

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = utils.read_img(input_img_path, resize_by_height)
    binary = utils.binarization(org, grad_min=int(uied_params['min-grad']))

    # *** Step 2 *** element detection
    utils.rm_line(binary, show=False, wait_key=0)
    uicompos = utils.component_detection(
        binary, min_obj_area=int(uied_params['min-ele-area']))

    # *** Step 3 *** results refinement
    uicompos = utils.compo_filter(uicompos, min_area=int(
        uied_params['min-ele-area']), img_shape=binary.shape)
    uicompos = utils.merge_intersected_compos(uicompos)
    utils.compo_block_recognition(binary, uicompos)
    if uied_params['merge-contained-ele']:
        uicompos = utils.rm_contained_compos_not_in_block(uicompos)
    utils.compos_update(uicompos, org.shape)
    utils.compos_containment(uicompos)

    # *** Step 4 ** nesting inspection: check if big compos have nesting element
    uicompos += nesting_inspection(org, grey,
                                   uicompos, ffl_block=uied_params['ffl-block'])
    utils.compos_update(uicompos, org.shape)

    # ##########################
    # RESULTS
    # ##########################

    # *** Step 5 *** save detection result
    utils.compos_update(uicompos, org.shape)

    clips = [compo.compo_clipping(org) for compo in uicompos]

    return clips, uicompos


def get_gui_components_crops(param_img_root, image_names, texto_detectado_ocr, path_to_save_bordered_images, add_words_columns, img_index):
    '''
    Analyzes an image and extracts its UI components

    :param param_img_root: Path to the image
    :type param_img_root: str
    :param image_names: Names of the images in the path
    :type image_names: list
    :param texto_detectado_ocr: Text detected by OCR in previous step
    :type texto_detectado_ocr: list
    :param path_to_save_bordered_images: Path to save the image along with the components detected
    :type path_to_save_bordered_images: str
    :param add_words_columns: Save words detected by OCR
    :type add_words_columns: bool
    :param img_index: Index of the image we want to analyze in images_names
    :type img_index: int
    :return: Crops and text inside components
    :rtype: Tuple
    '''
    words = {}
    words_columns_names = {}

    # if gaze_analysis:
    #     gaze_point_x = gaze_analysis['gaze_point_x'][img_index]
    #     gaze_point_y = gaze_analysis['gaze_point_y'][img_index]
    #     duration = gaze_analysis['duration'][img_index]
    # else:
    #     gaze_point_x = False

    image_path = param_img_root + image_names[img_index]
    # Read the image
    img = cv2.imread(image_path)
    img_copy = img.copy()
    # cv2_imshow(img_copy)

    # Store on global_y all the "y" coordinates and text boxes
    # Each row is a different text box, much more friendly than the format returned by keras_ocr 
    global_y = []
    global_x = []
    words[img_index] = {}
    res = None

    for j in range(0, len(texto_detectado_ocr[img_index])):
        coordenada_y = []
        coordenada_x = []

        for i in range(0, len(texto_detectado_ocr[img_index][j][1])):
            coordenada_y.append(texto_detectado_ocr[img_index][j][1][i][1])
            coordenada_x.append(texto_detectado_ocr[img_index][j][1][i][0])

        if add_words_columns:
            word = texto_detectado_ocr[img_index][j][0]
            centroid = (np.mean(coordenada_x), np.mean(coordenada_y))
            if word in words[img_index]:
                words[img_index][word] += [centroid]
            else:
                words[img_index][word] = [centroid]

            if word in words_columns_names:
                words_columns_names[word] += 1
            else:
                words_columns_names[word] = 1

        global_y.append(coordenada_y)
        global_x.append(coordenada_x)
        #print('Coord y, cuadro texto ' +str(j+1)+ str(global_y[j]))
        #print('Coord x, cuadro texto ' +str(j+1)+ str(global_x[j]))

    # print("Number of text boxes detected (iteration " + str(img_index) + "): " + str(len(texto_detectado_ocr[img_index])))

    # Interval calculation of the text boxes
    intervalo_y = []
    intervalo_x = []
    for j in range(0, len(global_y)):
        intervalo_y.append([int(max(global_y[j])), int(min(global_y[j]))])
        intervalo_x.append([int(max(global_x[j])), int(min(global_x[j]))])
    # print("Intervalo y", intervalo_y)
    # print("Intervalo x", intervalo_x)

    # Conversion to grey Scale
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(gris)

    # Gaussian blur
    gauss = cv2.GaussianBlur(gris, (5, 5), 0)
    # cv2_imshow(gauss)

    # Border detection with Canny
    canny = cv2.Canny(gauss, 50, 150)
    # cv2_imshow(canny)

    # Countour search in the image
    (contornos, _) = cv2.findContours(canny.copy(),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of GUI components detected: ", len(contornos), "\n")

    # draw the countours on the image
    cv2.drawContours(img_copy, contornos, -1, (0, 0, 255), 2)
    # cv2_imshow(img_copy)
    cv2.imwrite(path_to_save_bordered_images +
                image_names[img_index] + '_contornos.png', img_copy)

    # We carry out the crops for each detected countour
    recortes = []
    lista_para_no_recortar_dos_veces_mismo_gui = []

    text_or_not_text = []

    comp_json = {"img_shape": [img.shape], "compos": []}

    for j in range(0, len(contornos)):
        cont_horizontal = []
        cont_vertical = []
        # Obtain x and y max and min values of the countour
        for i in range(0, len(contornos[j])):
            cont_horizontal.append(contornos[j][i][0][0])
            cont_vertical.append(contornos[j][i][0][1])
        x = min(cont_horizontal)
        w = max(cont_horizontal)
        y = min(cont_vertical)
        h = max(cont_vertical)
        #print('Coord x, componente' + str(j+1) + '  ' + str(x) + ' : ' + str(w))
        #print('Coord y, componente' + str(j+1) + '  ' + str(y) + ' : ' + str(h))

        # Check that the countours are not overlapping with text boxes. If so, cut the text boxes
        condicion_recorte = True
        no_solapa = 1
        for k in range(0, len(intervalo_y)):
            solapa_y = 0
            solapa_x = 0
            y_min = min(intervalo_y[k])-cropping_threshold
            y_max = max(intervalo_y[k])+cropping_threshold
            x_min = min(intervalo_x[k])-cropping_threshold
            x_max = max(intervalo_x[k])+cropping_threshold
            # and max([y-y_min, y_max-y, h-y_min, y_max-h])<=surrounding_max_diff
            solapa_y = (y_min <= y <= y_max) or (y_min <= h <= y_max)
            # and max([x-x_min, x_max-x, w-x_min, x_max-w])<=surrounding_max_diff
            solapa_x = (x_min <= x <= x_max) or (x_min <= w <= x_max)
            if (solapa_y and solapa_x):
                if (lista_para_no_recortar_dos_veces_mismo_gui.count(k) == 0):
                    lista_para_no_recortar_dos_veces_mismo_gui.append(k)
                else:
                    # print("Text inside GUI component " + str(k) + " twice")
                    condicion_recorte = False
                x = min(intervalo_x[k])
                w = max(intervalo_x[k])
                y = min(intervalo_y[k])
                h = max(intervalo_y[k])
                no_solapa *= 0
                #crop_img = img[min(intervalo_y[k]) : max(intervalo_y[k]), min(intervalo_x[k]) : max(intervalo_x[k])]
                #print("Componente " + str(j+1) + " solapa con cuadro de texto")
        # if (solapa_y == 1 and solapa_x == 1):
        #crop_img = img[min(intervalo_y[k]) : max(intervalo_y[k]), min(intervalo_x[k]) : max(intervalo_x[k])]
        #print("Componente " + str(j+1) + " solapa con cuadro de texto")
        # recortes.append(crop_img)
        # else:
        # If the GUI component overlaps with the textbox, cut the later one
        # gaze_point_x and gaze_point_x >= x and gaze_point_x <= w and gaze_point_y >= y and gaze_point_y <= h and duration >= gaze_analysis_threshold
        coincidence_with_attention_point = True
        if (condicion_recorte and coincidence_with_attention_point):
            crop_img = img[y:h, x:w]
            text = []
            text = [word for word in words[img_index] if len([coord for coord in words[img_index][word] if x <= coord[0] <= w and y <= coord[1] <= h]) > 0]
            is_text = True if len(text)>0 else False
            comp_json["compos"].append({
                "id": int(j+1),
                "class": "Text" if is_text else "Compo",
                "Text": text[0] if is_text else None,
                "column_min": int(x),
                "row_min": int(y),
                "column_max": int(w),
                "row_max": int(h),
                "width": int(w - x),
                "height": int(h - y)
            })
            recortes.append(crop_img)
            text_or_not_text.append(abs(no_solapa-1))

    return (recortes, comp_json, text_or_not_text, words)


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


def detect_images_components(param_img_root, log, special_colnames, overwrite_npy, eyetracking_log_filename, image_names, text_detected_by_OCR, path_to_save_bordered_images, path_to_save_gui_components_npy, add_words_columns, algorithm):
    """
    With this function we process the screencaptures using the information resulting by aplying OCR
    and the image itself. We crop the GUI components and store them in a numpy array with all the 
    cropped components for each of the images in images_names


    :param param_img_root: Path where the imaages associated to each log row are stored
    :type param_img_root: str
    :image_names: Names of images in the log by alphabetical order
    :type image_names: list
    :texto_detectado_ocr: List of lists corresponding the words identified in the images on the log file
    :type texto_detectado_ocr: list
    :path_to_save_gui_components_npy: Path where the numpy arrays with the crops must be saved
    :type path_to_save_gui_components_npy: str
    :path_to_save_bordered_images: Path where the images along with their component borders must be stored
    :type path_to_save_bordered_images: str
    """
    no_modification = True

    eyetracking_log = False
    if eyetracking_log_filename and os.path.exists(param_img_root + eyetracking_log_filename):
        eyetracking_log = pd.read_csv(
            param_img_root + eyetracking_log_filename, sep=";")
    init_value_ui_log_timestamp = log[special_colnames['Timestamp']][0]

    gaze_events = {}  # key: row number,
    #value: { tuple: [coorX, coorY], gui_component_coordinate: [[corners_of_crop]]}

    last_upper_limit = 0

    # Iterate over the list of images
    for img_index in tqdm(range(0, len(image_names)), desc=f"Getting crops for {param_img_root}"):
        screenshot_texts_npy = path_to_save_gui_components_npy + \
            image_names[img_index] + "_texts.npy"
        screenshot_npy = path_to_save_gui_components_npy + \
            image_names[img_index] + ".npy"
        files_exists = os.path.exists(screenshot_npy)
        no_modification = no_modification and files_exists
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
                print("detect_images_components: LAST SCREENSHOT")
                interval, last_upper_limit = gaze_events_associated_to_event_time_range(
                    eyetracking_log,
                    special_colnames,
                    timestamp_start,
                    "LAST",
                    last_upper_limit)

            # { row_number: [[gaze_coorX, gaze_coorY],[gaze_coorX, gaze_coorY],[gaze_coorX, gaze_coorY]]}
            gaze_events[img_index] = interval

        if not files_exists or overwrite_npy:
            path_to_save_components_json = path_to_save_gui_components_npy.replace(
                "components_npy", "components_json")
            if not os.path.exists(path_to_save_components_json):
                os.makedirs(path_to_save_components_json)

            if algorithm == "legacy":
                recortes, comp_json, text_or_not_text, words = get_gui_components_crops(
                    param_img_root, image_names, text_detected_by_OCR, path_to_save_bordered_images, add_words_columns, img_index)
                
                with open(path_to_save_components_json + image_names[img_index] + '.json', "w") as outfile:
                    json.dump(comp_json, outfile)
                
                aux = np.array(recortes)
                np.save(screenshot_texts_npy, text_or_not_text)
                np.save(screenshot_npy, aux)
            elif algorithm == "uied":
                recortes, uicompos = get_uied_gui_components_crops(param_img_root, image_names, img_index)

                utils.save_corners_json(path_to_save_components_json + image_names[img_index] + '.json', uicompos)

                aux = np.array(recortes, dtype=object)
                np.save(screenshot_npy, aux)

            # if (add_words_columns and (not no_modification)) or (add_words_columns and (not os.path.exists(param_img_root+"text_colums.csv"))):
            #     storage_text_info_as_dataset(words, image_names, log, param_img_root)

# In this example, we choose zero-padding to resize the images


def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))


def uied_classify_image_components(model_path="resources/models/model.json", param_model_weights="resources/models/custom-v2.h5",
                                   model_properties="resources/models/custom-v2-classes.json", param_images_root="resources/screenshots/components_npy/",
                                   param_json_root="resources/screenshots/components_json/", param_log_path="resources/log.csv",
                                   enriched_log_output_path="resources/enriched_log_feature_extracted.csv", screenshot_colname="Screenshot",
                                   rewrite_log=False):
    """
    With this function we classify the copped component from each of the sreenshots to later add to the log the number of
    columns corresponding to the ammount to classes in the given model. These are the classes that a GUI component can fall into.
    The values indicated in these columns added indicate how many GUI components are present with regard to their class.
    Example. 2 button, 3 image_view, 1 text_view, etc.
    
    :param_model_weights: Classification model 
    :type param_model_weights: h5
    :param model_properties: Classes and shape of the model
    :type model_properties: json file
    :param param_images_root: Path where the cropped images are stored
    :type param_images_root: str
    :param param_json_root: Path where the json will all the components is stored
    :type param_json_root: str
    :param param_log_path: Path twhere the log we want to enrich is located
    :type param_log_path: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param screenshot_colname: Name of the column in the log indicating the images names
    :type screenshot_colname: str
    :param rewrite_log: Rewrite log
    :type rewrite_log: bool
    :returns: Enriched log
    :rtype: DataFrame
    """
    if not os.path.exists(enriched_log_output_path) or rewrite_log:
        # Load the model properties from the json
        f = json.load(open(model_properties,))
        classes = f["classes"]
        shape = tuple(f["shape"])

        # Load the ML classifier model for the crops
        # Default model is custom-v2, a model creating by using transfer learning from UIED's generalized model
        classifier = {}
        classifier['Elements'] = CompDetCNN(
            param_model_weights, classes, shape)
        print("\n\nLoaded ML model from disk\n")

        """
        Load the crops calculated in get_(uied)_gui_components_crops
        """
        log = pd.read_csv(param_log_path, sep=",")

        images_root = param_images_root  # "mockups_vector/"
        crop_imgs = {}
        # images_names = [ x + ".npy" for x in log.loc[:,"Screenshot"].values.tolist()] # os.listdir(images_root)
        images_names = log.loc[:, screenshot_colname].values.tolist()
        # print(images_names)
        for img_filename in images_names:
            crop_img_aux = np.load(images_root+img_filename +
                                   ".npy", allow_pickle=True)
            crop_imgs[img_filename] = {
                'content': crop_img_aux}

        """
        Once loaded, proceed to evaluate them with the corresponding model
        """
        print("\nLog dictionary length: " + str(len(crop_imgs)))

        for i in range(0, len(crop_imgs)):
            """
            This network gives as output the name of the detected class.
            Additionally, we moddify the json file with the components to add the corresponding classes
            """
            with open(param_json_root + images_names[i] + '.json', 'r') as f:
                data = json.load(f)

            clips = crop_imgs[images_names[i]]["content"].tolist()
            result = classifier['Elements'].predict(clips)

            crop_imgs[images_names[i]]["result"] = result
            crop_imgs[images_names[i]]["result_freq"] = pd.Series(
                result).value_counts()
            crop_imgs[images_names[i]]["result_freq_df"] = crop_imgs[images_names[i]
                                                                     ]["result_freq"].to_frame().T
            for j in range(0, len(result)):
                data["compos"][j]["class"] = result[j]
            with open(param_json_root + images_names[i] + '.json', "w") as jsonFile:
                json.dump(data, jsonFile)

        """
        Since not all images have all classes, a dataset with different columns depending on the images will be generated.
        It will depend whether GUI components of every kind appears o only a subset of these. That is why we initiañize a 
        dataframe will all possible columns, and include row by row the results obtained from the predictions
        """

        nombre_clases = classifier["Elements"].class_map
        df = pd.DataFrame([], columns=nombre_clases)

        for i in range(0, len(images_names)):
            row1 = [0 for i in range(0, len(nombre_clases))]
            # Acess the previously stored frequencies
            df1 = crop_imgs[images_names[i]]["result_freq_df"]
            if len(df1.columns.tolist()) > 0:
                for x in df1.columns.tolist():
                    uiui = nombre_clases.index(x)
                    row1[uiui] = df1[x][0]
                    df.loc[i] = row1

        """
        Once the dataset corresponding to the ammount of elements of each class contained in each of the images is obtained,
        we merge it with the complete log, adding the extracted characteristics from the images
        """

        log_enriched = log.join(df).fillna(method='ffill')

        """
        Finally we obtain an entiched log, which is turned as proof of concept of our hypothesis based on the premise that if
        we not only capture keyboard or mouse events on the monitorization through a keylogger, but also screencaptures,
        we can extract much more useful information, being able to improve the process mining over said log.
        As a pending task, we need to validate this hypothesis through a results comparison against the non-enriched log.
        We expect to continue this project in later stages of the master
        """
        log_enriched.to_csv(enriched_log_output_path)
        print("\n\n=========== ENRICHED LOG GENERATED: path=" +
              enriched_log_output_path)
    else:
        log_enriched = pd.read_csv(
            enriched_log_output_path, sep=",", index_col=0)
        print("\n\n=========== ENRICHED LOG ALREADY EXISTS: path=" +
              enriched_log_output_path)
    return log_enriched


def classify_image_components(param_json_file_name="resources/models/model.json", param_model_weights="resources/models/model.h5", 
                            model_properties="resources/models/custom-v2-classes.json", param_images_root="resources/screenshots/components_npy/",
                            param_json_root="resources/screenshots/components_json/", param_log_path="resources/log.csv", 
                            enriched_log_output_path="resources/enriched_log_feature_extracted.csv",  screenshot_colname="Screenshot", 
                            rewrite_log=False):
    """
    With this function we classify the copped component from each of the sreenshots to later add to the log the number of
    columns corresponding to the ammount to classes in the given model. These are the classes that a GUI component can fall into.
    The values indicated in these columns added indicate how many GUI components are present with regard to their class.
    Example. 2 button, 3 image_view, 1 text_view, etc.

    :param param_json_file_name:
    :type param_json_file_name: json file
    :param_model_weights: Weights of the edges of the classification neural network 
    :type param_model_weights: h5
    :param param_images_root: Path where the cropped images are stored
    :type param_images_root: str
    :param param_json_root: Path where the json will all the components is stored
    :type param_json_root: str
    :param param_log_path: Path twhere the log we want to enrich is located
    :type param_log_path: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param screenshot_colname: Name of the column in the log indicating the images names
    :type screenshot_colname: str
    :param rewrite_log: Rewrite log
    :type rewrite_log: bool
    :returns: Enriched log
    :rtype: DataFrame
    """
    if not os.path.exists(enriched_log_output_path) or rewrite_log:
        column_names = ['x0_Button', 'x0_CheckBox', 'x0_CheckedTextView', 'x0_EditText',
                        'x0_ImageButton', 'x0_ImageView', 'x0_NumberPicker', 'x0_RadioButton',
                        'x0_RatingBar', 'x0_SeekBar', 'x0_Spinner', 'x0_Switch', 'x0_TextView',
                        'x0_ToggleButton']

        # print("\n\n====== Column names =======================")
        # print(column_names)
        # print("===========================================\n\n")

        # load json and create model
        json_file = open(param_json_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(param_model_weights)
        # model = tf.keras.models.load_model('media/models/model_remaui.pb')
        print("\n\nLoaded ML model from disk\n")

        """
        Load the crops calculated in get_(uied)_gui_components_crops
        """
        log = pd.read_csv(param_log_path, sep=",")

        images_root = param_images_root  # "mockups_vector/"
        crop_imgs = {}
        # images_names = [ x + ".npy" for x in log.loc[:,"Screenshot"].values.tolist()] # os.listdir(images_root)
        images_names = log.loc[:, screenshot_colname].values.tolist()
        # print(images_names)
        for img_filename in images_names:
            crop_img_aux = np.load(images_root+img_filename +
                                   ".npy", allow_pickle=True)
            text_or_not_text = np.load(
                images_root+img_filename+"_texts.npy", allow_pickle=True)
            crop_imgs[img_filename] = {
                'content': crop_img_aux, 'text': text_or_not_text}

        """
        Once loaded, we reduce their size to adapt them to the neural network entry
        """
        print("\nLog dictionary length: " + str(len(crop_imgs)))

        crop_images = list(crop_imgs)
        # print("Padded: (150, 150, 3)")
        # print("Cropped: (50, 50, 3)\n\n")
        for i in range(0, len(crop_imgs)):
            aux = []
            for img in crop_imgs[crop_images[i]]["content"]:
                # for index, img in enumerate(crop_imgs[crop_images[i]]["content"]):
                # print("Original "+str(index)+": "+str(img.shape))
                if img.shape[1] > 150:
                    img = img[0:img.shape[0], 0:150]
                if img.shape[0] > 150:
                    img = img[0:150, 0:img.shape[1]]
                img_padded = pad(img, 150, 150)
                # print("Padded: "+str(img_padded.shape))
                img_resized = tf.image.resize(img_padded, [
                    50, 50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True, antialias=True)
                aux.append(img_resized)
                # print("Cropped: "+str(img_resized.shape))
            crop_imgs[crop_images[i]]["content_preprocessed"] = aux
            """
            This neural network returns as output a number indicating each of its classes. This number must me mapped to its 
            corresponding class name (str)
            """
            content_preprocessed_aux = crop_imgs[images_names[i]
                                                 ]["content_preprocessed"]

            # print("Content preprocessed length: " + str(len(crop_imgs[images_names[i]]["content_preprocessed"])))
            # result = loaded_model.predict_classes(np.array(content_preprocessed_aux)) # removed from tensorflow 2.6
            # for gui_component in content_preprocessed_aux:
            # print("Content preprocessed object type: " + str(type(gui_component)))
            # print("Content preprocessed component shape: " + str(gui_component.shape))
            predict_x = loaded_model.predict(
                np.array(content_preprocessed_aux))
            result = np.argmax(predict_x, axis=1)
            # print("\nPREDICTIONS:")
            # print(result)

            result_mapped = ["x0_TextView" if crop_imgs[crop_images[i]]["text"]
                             [index] else column_names[x] for index, x in enumerate(result)]

            crop_imgs[images_names[i]]["result"] = result_mapped
            crop_imgs[images_names[i]]["result_freq"] = pd.Series(
                result_mapped).value_counts()
            crop_imgs[images_names[i]]["result_freq_df"] = crop_imgs[images_names[i]
                                                                     ]["result_freq"].to_frame().T
            
            # Update the json file with components
            with open(param_json_root + images_names[i] + '.json', 'r') as f:
                data = json.load(f)
            for j in range(0, len(result_mapped)):
                data["compos"][j]["class"] = result_mapped[j]
            with open(param_json_root + images_names[i] + '.json', "w") as jsonFile:
                json.dump(data, jsonFile)

        """
        Since not all images have all classes, a dataset with different columns depending on the images will be generated.
        It will depend whether GUI components of every kind appears o only a subset of these. That is why we initiañize a 
        dataframe will all possible columns, and include row by row the results obtained from the predictions
        """

        nombre_clases = ['x0_RatingBar', 'x0_ToggleButton', 'x0_Spinner', 'x0_Switch', 'x0_CheckBox', 'x0_TextView', 'x0_EditText',
                         'x0_ImageButton', 'x0_NumberPicker', 'x0_CheckedTextView', 'x0_SeekBar', 'x0_ImageView', 'x0_RadioButton', 'x0_Button']
        df = pd.DataFrame([], columns=nombre_clases)

        for i in range(0, len(images_names)):
            row1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # Acess the previously stored frequencies
            df1 = crop_imgs[images_names[i]]["result_freq_df"]
            if len(df1.columns.tolist()) > 0:
                for x in df1.columns.tolist():
                    uiui = nombre_clases.index(x)
                    row1[uiui] = df1[x][0]
                    df.loc[i] = row1

        """
        Once the dataset corresponding to the ammount of elements of each class contained in each of the images is obtained,
        we merge it with the complete log, adding the extracted characteristics from the images
        """

        log_enriched = log.join(df).fillna(method='ffill')

        """
        Finally we obtain an entiched log, which is turned as proof of concept of our hypothesis based on the premise that if
        we not only capture keyboard or mouse events on the monitorization through a keylogger, but also screencaptures,
        we can extract much more useful information, being able to improve the process mining over said log.
        As a pending task, we need to validate this hypothesis through a results comparison against the non-enriched log.
        We expect to continue this project in later stages of the master
        """
        log_enriched.to_csv(enriched_log_output_path)
        print("\n\n=========== ENRICHED LOG GENERATED: path=" +
              enriched_log_output_path)
    else:
        log_enriched = pd.read_csv(
            enriched_log_output_path, sep=",", index_col=0)
        print("\n\n=========== ENRICHED LOG ALREADY EXISTS: path=" +
              enriched_log_output_path)
    return log_enriched


def storage_text_info_as_dataset(words, image_names, log, param_img_root):
    headers = []
    for w in words[1]:
        if words[1][w] == 1:
            headers.append(w)
        else:
            [headers.append(w+"_"+str(i)) for i in range(1, words[1][w]+1)]
    initial_row = ["NaN"]*len(headers)

    df = pd.DataFrame([], columns=headers)
    words = words[0]

    for j in range(0, len(image_names)):
        for w in words[j]:
            centroid_ls = list(words[j][w])
            if len(centroid_ls) == 1:
                if w in headers:
                    pos = headers.index(w)
                else:
                    pos = headers.index(w+"_1")
                    initial_row[pos] = centroid_ls[0]
            else:
                ac = 0
                for index, h in enumerate(headers):
                    if len(centroid_ls) > ac and (str(w) in h):
                        initial_row[index] = centroid_ls[ac]
                        ac += 1
        df.loc[j] = initial_row

    log_enriched = log.join(df).fillna(method='ffill')
    log_enriched.to_csv(param_img_root+"text_colums.csv")


"""
GUI component detection

The objective of this section is to be able to detect, define and crop each
of the graphic components (Symbols, images or text boxes) that form constitute
a screenshot

Libraries Import: We will use mainly keras_ocr and Opencv (Cv2).
=====================================
Border detection and cropping: OpenCV

We make use of OpenCV to carry out the following tasks:
- Image read.
- Calculation of intervals occupied by the text boxes obtained from keras_ocr
- Image processing:
    > GreyScale conversion
    > Gaussian blur
    > **Canny algorithm** for border detection
    > Countour detection
- Comparison betweeen countour and text boxes, giving more importance to text boxes 
    in case there is overlapping
- Final crop of each of the components

"""


def gui_components_detection(param_log_path, param_img_root, special_colnames, eyetracking_log_filename, add_words_columns=False, overwrite_npy=False, algorithm="legacy"):
    # Log read
    log = pd.read_csv(param_log_path, sep=",")
    # Extract the names of the screenshots associated to each of the rows in the log
    image_names = log.loc[:, special_colnames["Screenshot"]].values.tolist()
    pipeline = keras_ocr.pipeline.Pipeline()
    file_exists = os.path.exists(param_img_root + "images_ocr_info.txt")

    if file_exists:
        print("\n\nReading images OCR info from file...")
        with open(param_img_root + "images_ocr_info.txt", "rb") as fp:   # Unpickling
            text_corners = pickle.load(fp)
    else:
        text_corners = []
        for img in image_names:
            ocr_result = get_ocr_image(pipeline, param_img_root, img)
            text_corners.append(ocr_result[0])
        with open(param_img_root + "images_ocr_info.txt", "wb") as fp:  # Pickling
            pickle.dump(text_corners, fp)

    # print(len(text_corners))

    bordered = param_img_root+"contornos/"
    components_npy = param_img_root+"components_npy/"
    for p in [bordered, components_npy]:
        if not os.path.exists(p):
            os.mkdir(p)

    detect_images_components(param_img_root, log, special_colnames, overwrite_npy, eyetracking_log_filename,
                             image_names, text_corners, bordered, components_npy, add_words_columns, algorithm)


################################
############# Utils ############
################################

def check_npy_components_of_capture(image_name="1.png.npy", image_path="media/screenshots/components_npy/", interactive=False):
    if interactive:
        image_path = input("Enter path to images numpy arrays location: ")
        image_name = input("Enter numpy array file name: ")
    recortes = np.load(image_path+image_name, allow_pickle=True)
    for i in range(0, len(recortes)):
        print("Length: " + str(len(recortes)))
        if recortes[i].any():
            print("\nComponent: ", i+1)
            plt.imshow(recortes[i], interpolation='nearest')
            plt.show()
        else:
            print("Empty component")
    if interactive:
        image_path = input(
            "Do you want to check another image components? Indicate another npy file name: ")
        check_npy_components_of_capture(image_path, None, True)

def quantity_ui_elements_fe_technique(feature_extraction_technique_name, overwrite_info):
    print("TODO") # TODO: 

def location_ui_elements_fe_technique(feature_extraction_technique_name, overwrite_info):
    print("TODO") # TODO: 

def location_ui_elements_and_plaintext_fe_technique(feature_extraction_technique_name, overwrite_info):
    print("TODO") # TODO: 