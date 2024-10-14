import os
import copy
from os.path import join as pjoin
import json
import time
import keras_ocr
import numpy as np
import cv2
import zipfile
import matplotlib.pyplot as plt
from . import ip_draw as draw
import cv2
from art import tprint
import logging
import pickle
from tqdm import tqdm
from core.settings import PRIVATE_STORAGE_ROOT, sep
from core.settings import CROPPING_THRESHOLD, PLATFORM_NAME, DETECTION_PHASE_NAME
from apps.analyzer.utils import format_mht_file, read_ui_log_as_dataframe
import apps.featureextraction.utils as utils
from apps.featureextraction.SOM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import apps.featureextraction.SOM.ip_draw as draw
from apps.featureextraction.SOM.Component import Component 
from apps.featureextraction.SOM.sam import get_sam_gui_components_crops 
from apps.featureextraction.SOM.screen2som.predict import predict as screen2som_predict
from apps.featureextraction.SOM.screen2som.hierarchy_constructor import labels_to_output
from .UiComponent import UiComponent #QUIT
from django.utils.translation import gettext_lazy as _

"""
Text boxes detection: KERAS_OCR

In order to detect the text boxes inside the screenshots, we define the get_keras_ocr_image.
This function has a list of images as input, and a list of lists with the coordinates of text boxes coordinates
"""
def unzip_file(zip_file_path, dest_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder_path)

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
        keras_ocr.tools.read(os.path.join(param_img_root, path)) for path in images_input
    ]
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)
    # Plot the predictions
    # fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    # for ax, image, predictions in zip(axs, images, prediction_groups):
    #     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    return prediction_groups


def nesting_inspection(org, grey, compos, uied_params, times):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        # compos containment 
        start_nested_components_containment = time.time()
        for j in range(i + 1, len(compos)):
            relation = compos[i].compo_relation(compos[j])
            if relation == -1:
                compos[j].contain+=[compos[i]]
                compos[j] = compos[j]
            if relation == 1:
                compos[i].contain+=[compos[j]]
                compos[i] = compos[i]
        times["nested_compos_containment"] = time.time() - start_nested_components_containment
        
        if compo.height > uied_params["min-ele-height"]:
            replace = False
            clip_grey = compo.compo_clipping(grey)
            start_nested_components_detection = time.time()
            n_compos = utils.nested_components_detection(clip_grey, org, grad_thresh=uied_params["ffl-block"], uied_params=uied_params, show=False)
            times["nested_components_detection"] = time.time() - start_nested_components_detection
            
            if n_compos:
                compos[i].contain+=n_compos
                compos[i] = compos[i]
            
            for comp in n_compos:
                comp.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)

            # for n_compo in n_compos:
            #     if n_compo.redundant:
            #         compos[i] = n_compo
            #         replace = True
            #         break
            if not replace:
                nesting_compos += n_compos
        
    return compos + nesting_compos, times


def get_uied_gui_components_crops(input_imgs_path, path_to_save_bordered_images, image_names, img_index, times, uied_params={
        "min-grad": 3,
        "ffl-block": 5,
        "min-ele-area": 10,
        "min-ele-height": 10,
        "merge-contained-ele": False,
        "merge-intersected-compos": True,
        "max-word-inline-gap": 4,
        "max-line-gap": 4,
        "nested-min-compo-height": 5, # 10
        "nested-shared-area-percentage": 0.9,
        "nested-shared-area-to-be-redundant": 0.7,
        "nested-ignore-non-rectangle-blocks": False,
        "compo-height-considered-small": 5, # discarded during detection
        "compo-width-considered-small": 5, # discarded during detection
        "step_h": 2, # it affects hugely the detection time (if =5 real detection times close to mockup ones)
        "step_w": 2
    }):
    '''
    Analyzes an image and extracts its UI components with an alternative algorithm type

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

    '''
    ele:min-grad: gradient threshold to produce binary map         
    ele:ffl-block: fill-flood threshold
    ele:min-ele-area: minimum area for selected elements 
    ele:merge-contained-ele: if True, merge elements contained in others
    text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
    text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

    Tips:
    1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
    2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
    3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
    4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution
    '''

    
    name = input_img_path.split('/')[-1][:-4] if '/' in input_img_path else input_img_path.split('\\')[-1][:-4]
    ip_root = pjoin(path_to_save_bordered_images, "ip")
    if not os.path.exists(ip_root):
        os.mkdir(ip_root)

    # ##########################
    # COMPONENT DETECTION
    # ##########################

    start_get_binary_map_time = time.time()
    # *** Step 1 *** pre-processing: read img -> get binary map
    img, org, grey, og_shape = utils.read_img(input_img_path, resize_by_height)
    binary = utils.binarization(org, grad_min=int(uied_params['min-grad']))
    times["get_binary_map_time"] = time.time() - start_get_binary_map_time

    # *** Step 2 *** element detection
    start_get_component_detection = time.time()
    utils.rm_line(binary, show=False, wait_key=0)
    uicompos = utils.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']), uied_params=uied_params)
    new_uicompos = []

    # Resize to original image size
    for compo in uicompos:
        compo.compo_update(compo.id, org.shape, og_shape)
        new_uicompos.append(compo)
    uicompos = new_uicompos
    times["get_component_detection"] = time.time() - start_get_component_detection
    

    # *** Step 3 *** results refinement
    # DESKTOP: doesnt detect navbars
    # uicompos = utils.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=binary.shape)
    
    if uied_params['merge-intersected-compos']:
        start_merge_intersected_compos = time.time()
        uicompos = utils.merge_intersected_compos(uicompos)
        times["merge_intersected_compos"] = time.time() - start_merge_intersected_compos
    
    start_compo_block_recognition = time.time()
    utils.compo_block_recognition(binary, uicompos)
    times["compo_block_recognition"] = time.time() - start_compo_block_recognition
    
    if uied_params['merge-contained-ele']:
        start_rm_contained_compos_not_in_block = time.time()
        uicompos = utils.rm_contained_compos_not_in_block(uicompos)
        times["rm_contained_compos_not_in_block"] = time.time() - start_rm_contained_compos_not_in_block
    

    # *** Step 4 ** nesting inspection: check if big compos have nesting element
    start_nesting_inspection = time.time()
    uicompos, times = nesting_inspection(img, grey, uicompos, uied_params, times)
    times["nesting_inspection"] = time.time() - start_nesting_inspection

    # *** Step 5 *** save detection result
    # start_save_detection_result = time.time()
    # utils.compos_update(uicompos, img.shape, og_shape)
    # times["save_detection_result"] = time.time() - start_save_detection_result

    draw.draw_bounding_box(img, uicompos, show=False, name='merged compo', 
                           write_path=pjoin(ip_root, name + '.jpg'), 
                           wait_key=0)

    # ##########################
    # RESULTS
    # ##########################

    start_compo_clipping = time.time()
    clips = [compo.compo_clipping(img) for compo in uicompos]
    times["compo_clipping"] = time.time() - start_compo_clipping

    return clips, uicompos, times


def get_gui_components_crops(param_img_root, image_names, texto_detectado_ocr, path_to_save_bordered_images, img_index, text_classname, applied_ocr=False):
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
    :param img_index: Index of the image we want to analyze in images_names
    :type img_index: int
    :return: Crops and text inside components
    :rtype: Tuple
    '''
    words = {}

    image_path = os.path.join(param_img_root, image_names[img_index])
    # Read the image
    img = cv2.imread(image_path)
    img_copy = img.copy()
    # cv2_imshow(img_copy)

    # Store on global_y all the "y" coordinates and text boxes
    # Each row is a different text box, much more friendly than the format returned by keras_ocr 
    global_y = []
    global_x = []
    words[img_index] = {}

    if applied_ocr:
        for j in range(0, len(texto_detectado_ocr[img_index])):
            coordenada_y = []
            coordenada_x = []

            for i in range(0, len(texto_detectado_ocr[img_index][j][1])):
                coordenada_y.append(texto_detectado_ocr[img_index][j][1][i][1])
                coordenada_x.append(texto_detectado_ocr[img_index][j][1][i][0])

            word = texto_detectado_ocr[img_index][j][0]
            centroid = (np.mean(coordenada_x), np.mean(coordenada_y))
            if word in words[img_index]:
                words[img_index][word] += [centroid]
            else:
                words[img_index][word] = [centroid]

            global_y.append(coordenada_y)
            global_x.append(coordenada_x)
            # print('Coord y, cuadro texto ' +str(j+1)+ str(global_y[j]))
            # print('Coord x, cuadro texto ' +str(j+1)+ str(global_x[j]))

    # print("Number of text boxes detected (iteration " + str(img_index) + "): " + str(len(texto_detectado_ocr[img_index])))

    # Interval calculation of the text boxes
    intervalo_y = []
    intervalo_x = []
    for j in range(0, len(global_y)):
        intervalo_y.append([int(max(global_y[j])), int(min(global_y[j]))])
        intervalo_x.append([int(max(global_x[j])), int(min(global_x[j]))])

    # Conversion to grey Scale
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    gauss = cv2.GaussianBlur(gris, (5, 5), 0)

    # Border detection with Canny
    canny = cv2.Canny(gauss, 50, 150)

    # Countour search in the image
    (contornos, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of GUI components detected: ", len(contornos), "\n")

    # draw the countours on the image
    cv2.drawContours(img_copy, contornos, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(path_to_save_bordered_images, os.path.basename(image_names[img_index]) + '_bordered.png'), img_copy)

    # We carry out the crops for each detected countour
    recortes = []
    lista_para_no_recortar_dos_veces_mismo_gui = []

    text_or_not_text = []

    comp_json = {"img_shape": img.shape, "compos": []}

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
            y_min = min(intervalo_y[k])-CROPPING_THRESHOLD
            y_max = max(intervalo_y[k])+CROPPING_THRESHOLD
            x_min = min(intervalo_x[k])-CROPPING_THRESHOLD
            x_max = max(intervalo_x[k])+CROPPING_THRESHOLD
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
        # gaze_point_x and gaze_point_x >= x and gaze_point_x <= w and gaze_point_y >= y and gaze_point_y <= h and duration >= monitoring_threshold
        coincidence_with_attention_point = True # TODO: gaze analysis phase

        if (condicion_recorte and coincidence_with_attention_point):
            crop_img = img[y:h, x:w]
            text = [word for word in words[img_index] if len([coord for coord in words[img_index][word] if x <= coord[0] <= w and y <= coord[1] <= h]) > 0]
            is_text = True if len(text)>0 else False
            comp_json["compos"].append({
                "id": int(j+1),
                "class": text_classname if is_text else "Compo",
                "text": text[0] if is_text else "",
                "points": [[int(x), int(y)], [int(w), int(y)], [int(w), int(h)], [int(x), int(h)]],
                "centroid": [int((w-x)/2), int((h-y)/2)],
                "xpath": [],
                "relevant": True,
                # "column_min": int(x),
                # "row_min": int(y),
                # "column_max": int(w),
                # "row_max": int(h),
                # "width": int(w - x),
                # "height": int(h - y)
            })
            recortes.append(crop_img)
            text_or_not_text.append(abs(no_solapa-1))

    comp_json = labels_to_output(copy.deepcopy(comp_json), text_classname)

    return (recortes, comp_json, text_or_not_text, words)

def detect_images_components(scenario_path,param_img_root, log, special_colnames, skip, image_names, text_detected_by_OCR, path_to_save_bordered_images, algorithm, text_classname, metadata, configurations, applied_ocr):
    """
    With this function we process the screencaptures using the information resulting by aplying OCR
    and the image itself. We crop the GUI components and store them in a numpy array with all the 
    cropped components for each of the images in images_names

    :param scenario_path: Path to the scenario 
    :type scenario_path: str  
    :param param_img_root: Path where the images associated to each log row are stored
    :type param_img_root: str
    :image_names: Names of images in the log by alphabetical order
    :type image_names: list
    :texto_detectado_ocr: List of lists corresponding the words identified in the images on the log file
    :type texto_detectado_ocr: list
    :path_to_save_bordered_images: Path where the images along with their component borders must be stored
    :type path_to_save_bordered_images: str
    """
    # Since the path ends with a /, the last element of the split will be an empty string
    execution_root = scenario_path + '_results'
    
    path_to_save_gui_components_npy = os.path.join(execution_root, "components_npy")
    path_to_save_components_json = os.path.join(execution_root, "components_json")
    path_to_save_mask_elements=os.path.join(execution_root, 'sam_mask_elements')
    path_to_save_time_of_pipepile=os.path.join(execution_root, 'time_pipeline')

    # Iterate over the list of images
    for img_index in tqdm(range(0, len(image_names)), desc=f"Getting crops for {param_img_root}"):
        screenshot_texts_npy = os.path.join(path_to_save_gui_components_npy, os.path.basename(image_names[img_index]) + "_texts.npy")
        screenshot_npy = os.path.join(path_to_save_gui_components_npy, os.path.basename(image_names[img_index]) + ".npy")
        exists_screenshot_npy = os.path.exists(screenshot_npy)

        screenshot_json = os.path.join(path_to_save_components_json, os.path.basename(image_names[img_index]) + ".json")
        exists_screenshot_json = os.path.exists(screenshot_json)
        
        overwrite = (not exists_screenshot_json) or (not exists_screenshot_npy) or (not skip)

        times = {}
        
        if overwrite:
            start_t = time.time()
            if algorithm == "rpa-us":
                recortes, comp_json, text_or_not_text, words = get_gui_components_crops(param_img_root, image_names, text_detected_by_OCR, path_to_save_bordered_images, img_index, text_classname, applied_ocr=applied_ocr)
                
                # save metadata json
                with open(os.path.join(path_to_save_components_json, os.path.basename(image_names[img_index]) + '.json'), "w") as outfile:
                    json.dump(comp_json, outfile)

                # save texts npy
                np.save(screenshot_texts_npy, text_or_not_text)

            elif algorithm == "uied":
                # this method edit the metadata json with the ui element class and text if corresponds
                if configurations:
                    recortes, uicompos, times = get_uied_gui_components_crops(param_img_root, path_to_save_bordered_images, image_names, img_index, times, configurations)
                else:
                    recortes, uicompos, times = get_uied_gui_components_crops(param_img_root, path_to_save_bordered_images, image_names, img_index, times)

                # store all bounding boxes from the ui elements that are in 'uicompos'
                utils.save_corners_json(os.path.join(path_to_save_components_json, os.path.basename(image_names[img_index]) + '.json'), uicompos, img_index, text_detected_by_OCR, text_classname, applied_ocr)

            elif algorithm == "sam" or algorithm == "fast-sam":
                path_to_save_mask_npy=path_to_save_mask_elements+ os.path.basename(image_names[img_index])
                recortes, uicompos, mask_json, compos_json, arrays_dict,dict_times = get_sam_gui_components_crops(param_img_root, image_names, path_to_save_bordered_images, img_index, "checkpoints/", sam_type=algorithm)

                if not os.path.exists(path_to_save_time_of_pipepile):
                    os.makedirs(path_to_save_time_of_pipepile)
                
                with open(os.path.join(path_to_save_time_of_pipepile, image_names[img_index]+'_sam_time.json'),'w') as outfile:
                    json.dump(dict_times,outfile)

                #TODO save mask(sam) json
                # CURRENTLY COMMENTED OUT. IT DOES NOT PROVIDE MORE INFO THAN COMPOS JSON
                # with open(os.path.join(path_to_save_components_json, image_names[img_index]+'_sam_mask.json'),'w') as outfile:
                #     # json.dump(mask_json,outfile)
                #     outfile.write(mask_json)

                # save metadata json
                with open(os.path.join(path_to_save_components_json, os.path.basename(image_names[img_index]) + '.json'), "w") as outfile:
                    json.dump(compos_json, outfile)
                    # outfile.write(compos_json)

                path=path_to_save_mask_npy
                for n in ['segmentation','crop_box']:
                    path_element = path+'_'+n+'.npy'
                    aux = np.array(arrays_dict[n], dtype=object)
                    np.save(path_element,aux)

                # save texts npy
                # np.save(screenshot_texts_npy, text_or_not_text)

            elif algorithm == "screen2som":
                recortes, compos_json = screen2som_predict(os.path.join(param_img_root, image_names[img_index]), img_index, path_to_save_bordered_images, text_detected_by_OCR)
                screenshot_filename = os.path.basename(image_names[img_index])
                
                with open(os.path.join(path_to_save_components_json, screenshot_filename + '.json'), "w") as outfile:
                    json.dump(compos_json, outfile)

            else:
                raise Exception("You select a type of UI element detection that doesnt exists")
            # if (add_words_columns and (not no_modification)) or (add_words_columns and (not os.path.exists(param_img_root+"text_colums.csv"))):
            #     storage_text_info_as_dataset(words, image_names, log, param_img_root)

            if image_names[img_index] in metadata['screenshots']:
                metadata['screenshots'][image_names[img_index]]['detect_images_components duration'] = float(time.time()) - float(start_t)
                metadata['screenshots'][image_names[img_index]]['detect_images_components #UICompos'] = len(recortes)
            else:
                metadata['screenshots'][image_names[img_index]] = {
                    "detect_images_components duration": float(time.time()) - float(start_t),
                    "detect_images_components #UICompos": len(recortes)
                }
                
            for key in times:
                metadata['screenshots'][image_names[img_index]]['detect_images_components ' + key] = times[key]
            
            # save ui elements npy
            aux = np.array(recortes, dtype=object)
            np.save(screenshot_npy, aux)
        
    return metadata


# def storage_text_info_as_dataset(words, image_names, log, param_img_root):
#     headers = []
#     for w in words[1]:
#         if words[1][w] == 1:
#             headers.append(w)
#         else:
#             [headers.append(w+"_"+str(i)) for i in range(1, words[1][w]+1)]
#     initial_row = ["NaN"]*len(headers)

#     df = pd.DataFrame([], columns=headers)
#     words = words[0]

#     for j in range(0, len(image_names)):
#         for w in words[j]:
#             centroid_ls = list(words[j][w])
#             if len(centroid_ls) == 1:
#                 if w in headers:
#                     pos = headers.index(w)
#                 else:
#                     pos = headers.index(w+"_1")
#                     initial_row[pos] = centroid_ls[0]
#             else:
#                 ac = 0
#                 for index, h in enumerate(headers):
#                     if len(centroid_ls) > ac and (str(w) in h):
#                         initial_row[index] = centroid_ls[ac]
#                         ac += 1
#         df.loc[j] = initial_row

#     log_enriched = log.join(df).fillna(method='ffill')
#     log_enriched.to_csv(param_img_root+"text_colums.csv")


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


def ui_elements_detection(param_log_path, scenario_path, execution):
    log_input_filaname = execution.ui_elements_detection.input_filename
    img_root = scenario_path if not execution.prefilters else os.path.join(scenario_path + "_results", "prefiltered_img")
    special_colnames = execution.case_study.special_colnames
    configurations = execution.ui_elements_detection.configurations
    algorithm = execution.ui_elements_detection.type
    apply_ocr = execution.ui_elements_detection.ocr
    skip = execution.ui_elements_detection.preloaded
    text_classname = "text"
    
    
    tprint(PLATFORM_NAME + " - " + DETECTION_PHASE_NAME, "fancy60")
    print(scenario_path+"\n")

    if os.path.exists(param_log_path):
        logging.info(_("apps/featureextraction/SOM/detection.py Log already exists, it's not needed to execute format conversor"))
        print(_("Log already exists, it's not needed to execute format conversor"))
    elif "format" in configurations:
        logging.info(_("apps/featureextraction/SOM/detection.py Format conversor executed! Type: %(format), Filename: %(filename)") % {"format": configurations["format"], "filename": configurations["formatted_log_name"]})
        if "formatted_log_name" in configurations:
            log_filename = configurations["formatted_log_name"]
        else:
            log_filename = "log"
        param_log_path = format_mht_file(os.path.join(scenario_path, log_input_filaname), configurations["format"], scenario_path, log_filename, configurations["org:resource"])

    # Log read
    log = read_ui_log_as_dataframe(param_log_path)
    # Extract the names of the screenshots associated to each of the rows in the log
    image_names = log.loc[:, special_colnames["Screenshot"]].values.tolist()
    text_corners = []
    file_exists = os.path.exists(os.path.join(img_root, "images_ocr_info.txt"))

    metadata = { 'screenshots': {} } 

    if file_exists:
        print(_("\n\nReading images OCR info from file..."))
        with open(os.path.join(img_root, "images_ocr_info.txt"), "rb") as fp:   # Unpickling
            text_corners = pickle.load(fp)
    elif apply_ocr:
        pipeline = keras_ocr.pipeline.Pipeline()
        for img in image_names:
            start_t = time.time()
            ocr_result = get_ocr_image(pipeline, img_root, img)
            text_corners.append(ocr_result[0])
            metadata['screenshots'][img] = {"get_ocr_image duration": float(time.time()) - float(start_t)}

        with open(os.path.join(img_root, "images_ocr_info.txt"), "wb") as fp:  # Pickling
            pickle.dump(text_corners, fp)

    # print(len(text_corners))

    execution_root = scenario_path + '_results'

    bordered = os.path.join(execution_root, "borders")
    components_npy = os.path.join(execution_root, "components_npy")
    components_json = os.path.join(execution_root, "components_json")
    for p in [bordered, components_npy, components_json]:
        if not os.path.exists(p):
            os.makedirs(p)

    start_t = time.time()
    metadata = detect_images_components(scenario_path,img_root, log, special_colnames, skip, image_names, text_corners, bordered, algorithm, text_classname, metadata, configurations, apply_ocr)
    metadata["duration"] = float(time.time()) - float(start_t)
    return metadata


############################################
############# Development utils ############
############################################

def check_npy_components_of_capture(image_name="1.png.npy", image_path="media/screenshots/components_npy/", interactive=False):
    if interactive:
        image_path = input(_("Enter path to images numpy arrays location: "))
        image_name = input(_("Enter numpy array file name: "))
    recortes = np.load(image_path+image_name, allow_pickle=True)
    for i in range(0, len(recortes)):
        print(_("Length: ") + str(len(recortes)))
        if recortes[i].any():
            print(_("\nComponent: "), i+1)
            plt.imshow(recortes[i], interpolation='nearest')
            plt.show()
        else:
            print(_("Empty component"))
    if interactive:
        image_path = input(
            _("Do you want to check another image components? Indicate another npy file name: "))
        check_npy_components_of_capture(image_path, None, True)