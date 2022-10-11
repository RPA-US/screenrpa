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
Detección cuadros de texto: KERAS_OCR

Con el objetivo de detectar los cuadros de texto dentro las capturas de pantalla,
definimos la función get_keras_ocr_image.
Esta función tendrá como input una lista de imagénes, y como output, una lista en la que habrá una lista por cada imagen de entrada.
En cada una de estas listas, estarán las coordenadas de las esquinas que conforman cada uno de los cuadros de texto detectados.
"""


def get_ocr_image(pipeline, param_img_root, images_input):
    """
    Aplica Keras-OCR sobre la imagen o imágenes de entrada para extraer el texto plano y las coordenadas
    correspondientes a las palabras que estén presentes.

    :pipeline: keras pipeline
    :type pipeline: keras pipeline
    :param_img_root: ruta donde se almacenan las imágenes capturadas asociadas a cada fila del log
    :type param_img_root: str
    :param images_input: ruta (string) o lista con las rutas, de la imagen/es a tratar
    :type images_input: str or list
    :returns: lista de listas correspondientes a las palabras identificadas en la imagen de entrada. Ejemplo: lista de array de información sobre palabras descubiertas ('delete', array([[1161.25,  390.  ], [1216.25,  390.  ], [1216.25,  408.75], [1161.25,  408.75]], dtype=float32))
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


def get_uied_gui_components_crops(input_imgs_path, image_names, img_index, path_to_save_components_json):

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
    words = {}
    words_columns_names = {}

    # if gaze_analysis:
    #     gaze_point_x = gaze_analysis['gaze_point_x'][img_index]
    #     gaze_point_y = gaze_analysis['gaze_point_y'][img_index]
    #     duration = gaze_analysis['duration'][img_index]
    # else:
    #     gaze_point_x = False

    image_path = param_img_root + image_names[img_index]
    # Leemos la imagen
    img = cv2.imread(image_path)
    img_copy = img.copy()
    # cv2_imshow(img_copy)

    # Almacenamos en global_y todas las coordenadas "y" de los cuadros de texto
    # Cada fila es un cuadro de texto distinto, mucho más amigable que el formato que devuelve keras_ocr
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

    # Cálculo los intervalos de los cuadros de texto
    intervalo_y = []
    intervalo_x = []
    for j in range(0, len(global_y)):
        intervalo_y.append([int(max(global_y[j])), int(min(global_y[j]))])
        intervalo_x.append([int(max(global_x[j])), int(min(global_x[j]))])
    # print("Intervalo y", intervalo_y)
    # print("Intervalo x", intervalo_x)

    # Convertimos a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(gris)

    # Aplicar suavizado Gaussiano
    gauss = cv2.GaussianBlur(gris, (5, 5), 0)
    # cv2_imshow(gauss)

    # Detectamos los bordes con Canny
    canny = cv2.Canny(gauss, 50, 150)
    # cv2_imshow(canny)

    # Buscamos los contornos
    (contornos, _) = cv2.findContours(canny.copy(),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of GUI components detected: ", len(contornos), "\n")

    # y los dibujamos
    cv2.drawContours(img_copy, contornos, -1, (0, 0, 255), 2)
    # cv2_imshow(img_copy)
    cv2.imwrite(path_to_save_bordered_images +
                image_names[img_index] + '_contornos.png', img_copy)

    # Llevamos a cabo los recortes para cada contorno detectado
    recortes = []
    lista_para_no_recortar_dos_veces_mismo_gui = []

    text_or_not_text = []

    for j in range(0, len(contornos)):
        cont_horizontal = []
        cont_vertical = []
        # Obtenemos componentes máximas y mínimas (eje x,y) del contorno
        for i in range(0, len(contornos[j])-1):
            cont_horizontal.append(contornos[j][i][0][0])
            cont_vertical.append(contornos[j][i][0][1])
            x = min(cont_horizontal)
            w = max(cont_horizontal)
            y = min(cont_vertical)
            h = max(cont_vertical)
        #print('Coord x, componente' + str(j+1) + '  ' + str(x) + ' : ' + str(w))
        #print('Coord y, componente' + str(j+1) + '  ' + str(y) + ' : ' + str(h))

        # Comprobamos que los contornos no solapen con cuadros de texto y optamos por recortar los cuadros de texto si solapan.
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
        # Si el componente GUI solapa con el cuadro de texto, cortamos el cuadro de texto a partir de las coordenadas de sus esquinas
        # gaze_point_x and gaze_point_x >= x and gaze_point_x <= w and gaze_point_y >= y and gaze_point_y <= h and duration >= gaze_analysis_threshold
        coincidence_with_attention_point = True
        if (condicion_recorte and coincidence_with_attention_point):
            crop_img = img[y:h, x:w]
            recortes.append(crop_img)
            text_or_not_text.append(abs(no_solapa-1))

    return (recortes, text_or_not_text, words)


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
    Con esta función preprocesamos las imágenes de las capturas a partir de la información resultante de 
    aplicar OCR y de la propia imagen. Recortamos los componentes GUI y se almacena un numpy array con
    todos los componentes recortados para cada una de las imágenes indicadas en images_names

    :param_img_root: ruta donde se almacenan las imágenes capturadas asociadas a cada fila del log
    :type param_img_root: str
    :image_names: lista con el nombre de las imágenes que están presentes en el log (listadas por orden de aparición)
    :type image_names: list
    :texto_detectado_ocr: lista de listas, en la que cada una de ellas se corresponde con las palabras detectadas en cada imagen del log y sus coordenadas correpsondientes. Se corresponde con el formato de salida de la función get_ocr_image
    :type texto_detectado_ocr: list
    :path_to_save_gui_components_npy: ruta donde almacenar los numpy arrays con los componentes recortados de cada imagen
    :type path_to_save_gui_components_npy: str
    :path_to_save_bordered_images: ruta donde se almacenan las imágenes de cada componente con el borde resaltado
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

    # Recorremos la lista de imágenes
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
            if algorithm == "legacy":
                recortes, text_or_not_text, words = get_gui_components_crops(
                    param_img_root, image_names, text_detected_by_OCR, path_to_save_bordered_images, add_words_columns, img_index)
                aux = np.array(recortes)
                np.save(screenshot_texts_npy, text_or_not_text)
                np.save(screenshot_npy, aux)
            elif algorithm == "uied":
                path_to_save_components_json = path_to_save_gui_components_npy.replace(
                    "components_npy", "components_json")
                recortes, uicompos = get_uied_gui_components_crops(
                    param_img_root, image_names, img_index, path_to_save_components_json)
                if not os.path.exists(path_to_save_components_json):
                    os.makedirs(path_to_save_components_json)
                utils.save_corners_json(
                    path_to_save_components_json + image_names[img_index] + '.json', uicompos)
                aux = np.array(recortes, dtype=object)
                np.save(screenshot_npy, aux)

            # if (add_words_columns and (not no_modification)) or (add_words_columns and (not os.path.exists(param_img_root+"text_colums.csv"))):
            #     storage_text_info_as_dataset(words, image_names, log, param_img_root)

# Para el caso de este ejemplo elegimos la función de Zero-padding para redimensionar las imágenes


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
    Con esta función clasificamos los componentes recortados de cada una de las capturas para posteriormente añadir al log
    10 columnas. Estas corresponden a cada una de las clases en las que se puede clasificar un componente GUI. Los valores 
    indicados en estas columnas añadidas indican cuántos de componentes GUI están presentes en la captura según su tipo, es
    decir, 2 button, 3 image_view, 1 text_view, etc.

    :param_img_root: ruta donde se almacenan las imágenes capturadas asociadas a cada fila del log
    :type param_img_root: str
    :param_json_file_name: estructura del modelo de conocimiento de la red neuronal que clasifica los componentes GUI
    :type param_json_file_name: json
    :param_model_weights: pesos de las aristas de la red neuronal que clasifica los componentes GUI 
    :type param_model_weights: h5
    :param_log_path: ruta donde se encuentra el log que queremos enriquecer con cuántos componentes GUI de cada tipo hay en sus capturas
    :type param_log_path: str
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
        Cargamos los recortes de las imagenes calculados en get_gui_components_crops
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
        Una vez cargadas, procedemos a evaluarlas con el modelo correspondiente
        """
        print("\nLog dictionary length: " + str(len(crop_imgs)))

        for i in range(0, len(crop_imgs)):
            """
            Esta red devuelve como salida el nombre de la clase detectada.
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
        Como todas las imágenes no tendrán todas las clases, generarán como salida un dataset que no tendrán siempre las mismas
        columnas. Dependerá si en la imagen aparecen componentes GUI de todo tipo o solamente un subconjunto de ellos. Por ello, 
        inicializamos un dataframe con todas las columnas posibles, y vamos incluyendo una fila por cada resultado obtenido de
        cada imagen pasada por la red.
        """

        nombre_clases = classifier["Elements"].class_map
        df = pd.DataFrame([], columns=nombre_clases)

        for i in range(0, len(images_names)):
            row1 = [0 for i in range(0, len(nombre_clases))]
            # accedemos a las frecuencias almacenadas anteriormente
            df1 = crop_imgs[images_names[i]]["result_freq_df"]
            if len(df1.columns.tolist()) > 0:
                for x in df1.columns.tolist():
                    uiui = nombre_clases.index(x)
                    row1[uiui] = df1[x][0]
                    df.loc[i] = row1

        """
        Una vez obtenido el dataset correspondiente a la cantidad de elementos de cada clase contenidos en cada una de las
        imágenes. Unimos este con el log completo, añadiendo las características extraídas de las imágenes.
        """

        log_enriched = log.join(df).fillna(method='ffill')

        """
        Finalmente obtenemos un log enriquecido, que se torna como prueba de concepto de nuestra hipótesis basada en que, si
        no solamente capturamos eventos de teclado o de ratón en la monitorización a través de un keylogger, sino que obtenemos
        también capturas de pantalla. Podemos extraer mucha más información útil, pudiendo mejorar la minería de procesos sobre
        dicho log. Nos ha quedado pendiente validar esta hipótesis mediante la comparación de resultados entre aplicar técnicas
        de pricess mining aplicadas sobre el log inicial vs. el log enriquecido. Esperamos poder continuar con este proyecto
        en fases posteriores del máster.
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


def classify_image_components(param_json_file_name="resources/models/model.json", param_model_weights="resources/models/model.h5", model_properties="resources/models/custom-v2-classes.json", param_images_root="resources/screenshots/components_npy/", param_log_path="resources/log.csv", enriched_log_output_path="resources/enriched_log_feature_extracted.csv", screenshot_colname="Screenshot", rewrite_log=False):
    """
    Con esta función clasificamos los componentes recortados de cada una de las capturas para posteriormente añadir al log
    14 columnas. Estas corresponden a cada una de las clases en las que se puede clasificar un componente GUI. Los valores 
    indicados en estas columnas añadidas indican cuántos de componentes GUI están presentes en la captura según su tipo, es
    decir, 2 button, 3 image_view, 1 text_view, etc.

    :param_img_root: ruta donde se almacenan las imágenes capturadas asociadas a cada fila del log
    :type param_img_root: str
    :param_json_file_name: estructura del modelo de conocimiento de la red neuronal que clasifica los componentes GUI
    :type param_json_file_name: json
    :param_model_weights: pesos de las aristas de la red neuronal que clasifica los componentes GUI 
    :type param_model_weights: h5
    :param_log_path: ruta donde se encuentra el log que queremos enriquecer con cuántos componentes GUI de cada tipo hay en sus capturas
    :type param_log_path: str
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
        Tras comprobar el accuracy obtenido tras la evaluación, vamos a aplicar el modelo a un experimento basado en un log,
        con imágenes mockeadas a modo de screenchots del proceso, para extraerle las características a estas.

        Cogemos los vectores de numpy correspondientes a cada imagen, obtenidos a través de la aplicación del notebook
        de Detección y recorte de componentes GUI, que tendrán que ser almacenadas en la ruta "mockups_vector/".
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
        Una vez cargadas, reducimos su tamaño para adecuarlo a la entrada de la red neuronal convolucional producto de este
        notebook.
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
            Esta red devuelve como salida un número, indicando una de las clases. Este número tendrá que ser mapeado a su texto
            (nombre de la clase) correspondiente.
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

        """
        Como todas las imágenes no tendrán todas las clases, generarán como salida un dataset que no tendrán siempre las mismas
        columnas. Dependerá si en la imagen aparecen componentes GUI de todo tipo o solamente un subconjunto de ellos. Por ello, 
        inicializamos un dataframe con todas las columnas posibles, y vamos incluyendo una fila por cada resultado obtenido de
        cada imagen pasada por la red.
        """

        nombre_clases = ['x0_RatingBar', 'x0_ToggleButton', 'x0_Spinner', 'x0_Switch', 'x0_CheckBox', 'x0_TextView', 'x0_EditText',
                         'x0_ImageButton', 'x0_NumberPicker', 'x0_CheckedTextView', 'x0_SeekBar', 'x0_ImageView', 'x0_RadioButton', 'x0_Button']
        df = pd.DataFrame([], columns=nombre_clases)

        for i in range(0, len(images_names)):
            row1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # accedemos a las frecuencias almacenadas anteriormente
            df1 = crop_imgs[images_names[i]]["result_freq_df"]
            if len(df1.columns.tolist()) > 0:
                for x in df1.columns.tolist():
                    uiui = nombre_clases.index(x)
                    row1[uiui] = df1[x][0]
                    df.loc[i] = row1

        """
        Una vez obtenido el dataset correspondiente a la cantidad de elementos de cada clase contenidos en cada una de las
        imágenes. Unimos este con el log completo, añadiendo las características extraídas de las imágenes.
        """

        log_enriched = log.join(df).fillna(method='ffill')

        """
        Finalmente obtenemos un log enriquecido, que se torna como prueba de concepto de nuestra hipótesis basada en que, si
        no solamente capturamos eventos de teclado o de ratón en la monitorización a través de un keylogger, sino que obtenemos
        también capturas de pantalla. Podemos extraer mucha más información útil, pudiendo mejorar la minería de procesos sobre
        dicho log. Nos ha quedado pendiente validar esta hipótesis mediante la comparación de resultados entre aplicar técnicas
        de pricess mining aplicadas sobre el log inicial vs. el log enriquecido. Esperamos poder continuar con este proyecto
        en fases posteriores del máster.
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
Detección de componentes GUI

El objetivo de esta parte es poder detectar, delimitar y posteriormente recortar,
cada uno de los componentes gráficos (símbolos, imágenes o cuadros de texto)
que componen una captura de pantalla.

Importación de Librerías: Nos vamos a servir principalmente de dos librerías,
keras_ocr y Opencv (Cv2).
======================================
Detección de bordes y recortes: OpenCV

Hacemos uso de la libería OpenCV para llevar a cabo las siguientes tareas:
- Lectura de la imagen.
- Cálculo de intervalos ocupados por los cuadros de texto obtenidos a través de keras_ocr
- Tratamiento de la imagen:
  > Conversión a escala de grises.
  > Suavizado gaussiano a la imagen.
  > **Algoritmo de Canny** para la detección de bordes.
  > Obtención de contornos.
- Comparación entre contornos y cuadros de texto, otorgando más peso al cuadro de texto en
  el caso de que se solape espacialmente con algún contorno detectado.
- Recorte final de cada uno de los componentes.

"""


def gui_components_detection(param_log_path, param_img_root, special_colnames, eyetracking_log_filename, add_words_columns=False, overwrite_npy=False, algorithm="legacy"):
    # Leemos el log
    log = pd.read_csv(param_log_path, sep=",")
    # Extraemos los nombres de las capturas asociadas a cada fila del log
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