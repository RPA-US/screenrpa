# Components detection
import json
import numpy as np
import pandas as pd
import os
# Classification
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from tqdm import tqdm
from core.utils import read_ui_log_as_dataframe
from core.settings import sep
from apps.featureextraction.SOM.CNN.CompDetCNN import CompDetCNN

default_ui_elements_classification_classes = ['x0_Button', 'x0_CheckBox', 'x0_CheckedTextView', 'x0_EditText', 'x0_ImageButton', 'x0_ImageView', 'x0_NumberPicker', 'x0_RadioButton', 'x0_RatingBar', 'x0_SeekBar', 'x0_Spinner', 'x0_Switch', 'x0_TextView', 'x0_ToggleButton']
default_ui_elements_classification_image_shape = [64, 64, 3]

###################################################################################################
######################################       UTILS        #########################################
###################################################################################################
def pad(img, h, w):
    """
    Zero-padding function to resize the images
    """
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))
    
def check_metadata_json_exists(ui_log_path, screenshot_colname, metadata_json_root):
    """
    Auxiliar function of 'ui_elements_classification' to check if there is any missing metadata json file

    :param metadata_json_root: Path where the json will all the components is stored
    :type metadata_json_root: str
    :param screenshot_colname: Name of the column in the log indicating the images names
    :type screenshot_colname: str
    :param ui_log_path: Path twhere the log we want to enrich is located
    :type ui_log_path: str
    :returns screenshot_filenames: filenames of log screenshots
    :rtype: list
    :returns missing_json_file: a boolean that indicates if there is any missing metadata json
    :rtype: bool
    """
    log = read_ui_log_as_dataframe(ui_log_path)
    # screenshot_filenames = [ x + ".npy" for x in log.loc[:,"Screenshot"].values.tolist()]
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    missing_json_file = False

    for screenshot in screenshot_filenames:
        screenshot = os.path.basename(screenshot)
        if not os.path.exists(metadata_json_root + screenshot + '.json'):
            missing_json_file = True
            break

    return screenshot_filenames, missing_json_file

###################################################################################################
###################################################################################################

def uied_ui_elements_classification(ui_log_path, path_scenario, execution):
    """
    With this function we classify the copped component from each of the sreenshots to later add to the log the number of
    columns corresponding to the ammount to classes in the given model. These are the classes that a GUI component can fall into.
    The values indicated in these columns added indicate how many GUI components are present with regard to their class.
    Example. 2 button, 3 image_view, 1 text_view, etc.
    
    :param_model: Classification model  
    :type param_model: h5
    :param model_properties:
    :type model_properties: Classes and shape of the model
    :param ui_elements_crops_npy_root: Path where the cropped images are stored
    :type ui_elements_crops_npy_root: str
    :param metadata_json_root: Path where the json will all the components is stored
    :type metadata_json_root: str
    :param ui_log_path: Path twhere the log we want to enrich is located
    :type ui_log_path: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param screenshot_colname: Name of the column in the log indicating the images names
    :type screenshot_colname: str
    :param text_classname: Name of the the corresponding class for the texts
    :type text_classname: str
    :param skip: Rewrite classification data (json files)
    :type skip: bool
    :param ui_elements_classification_classes: Model classes
    :type ui_elements_classification_classes: list
    
    """
    execution_root = path_scenario + "_results"
    # path_results = "/".join(path_scenario.split("/")[:-1]) + "_results" + "/"
    ui_elements_crops_npy_root = os.path.join(execution_root, 'components_npy')
    metadata_json_root = os.path.join(execution_root, 'components_json')
    model = execution.ui_elements_classification.model.path # specific extractors
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    text_classname = execution.ui_elements_classification.model.text_classname
    ui_elements_classification_classes = execution.ui_elements_classification.model.classes
    ui_elements_classification_image_shape = execution.ui_elements_classification.model.image_shape
    skip = execution.ui_elements_classification.preloaded
    
    screenshot_filenames, missing_json_file = check_metadata_json_exists(ui_log_path, screenshot_colname, metadata_json_root)

    if missing_json_file or (not skip):
        # Load the ML classifier model for the crops
        # Default model is custom-v2, a model creating by using transfer learning from UIED's generalized model
        classifier = {}
        classifier['Elements'] = CompDetCNN(
            model, ui_elements_classification_classes, ui_elements_classification_image_shape)
        print("\n\nLoaded ML model from disk\n")

        for screenshot_filename in tqdm(screenshot_filenames, desc=f"Classifying images in {ui_elements_crops_npy_root}"):
            screenshot_filename = os.path.basename(screenshot_filename)
            # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
            with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), 'r') as f:
                data = json.load(f)

            clips =  np.load(os.path.join(ui_elements_crops_npy_root, screenshot_filename + ".npy"), allow_pickle=True).tolist()
            result = classifier['Elements'].predict(clips)

            for j in range(0, len(result)):
                data["compos"][j]["class"] = result[j]
            with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), "w") as jsonFile:
                json.dump(data, jsonFile)


def legacy_ui_elements_classification(ui_log_path, path_scenario, execution):
    """
    With this function we classify the copped component from each of the sreenshots to later add to the log the number of
    columns corresponding to the ammount to classes in the given model. These are the classes that a GUI component can fall into.
    The values indicated in these columns added indicate how many GUI components are present with regard to their class.
    Example. 2 button, 3 image_view, 1 text_view, etc.

    :param model: Weights of the edges of the classification neural network 
    :type model: str
    :param ui_elements_crops_npy_root: Path where the cropped images are stored
    :type ui_elements_crops_npy_root: str
    :param metadata_json_root: Path where the json will all the components is stored
    :type metadata_json_root: str
    :param ui_log_path: Path twhere the log we want to enrich is located
    :type ui_log_path: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param screenshot_colname: Name of the column in the log indicating the images names
    :type screenshot_colname: str
    :param text_classname: Name of the the corresponding class for the texts
    :type text_classname: str
    :param skip: Rewrite classification data (json files)
    :type skip: bool
    :returns: Enriched log
    :rtype: DataFrame
    """
    
    path_results = path_scenario + "_results"
    ui_elements_crops_npy_root = os.path.join(path_results, 'components_npy')
    metadata_json_root = os.path.join(path_results, 'components_json')
    model = execution.ui_elements_classification.model.path # specific extractors
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    text_classname = execution.ui_elements_classification.model.text_classname
    ui_elements_classification_classes = execution.ui_elements_classification.model.classes
    ui_elements_classification_image_shape = execution.ui_elements_classification.model.image_shape
    skip = execution.ui_elements_classification.preloaded

    screenshot_filenames, missing_json_file = check_metadata_json_exists(ui_log_path, screenshot_colname, metadata_json_root)

    if missing_json_file or (not skip):
        # load json and create model
        loaded_model = model_from_json(json.dumps(execution.ui_elements_classification.model.model_properties))

        # load weights into new model
        loaded_model.load_weights(model)
        crops_info = {}
        
        for img_filename in screenshot_filenames:
            crops = np.load(os.path.join(ui_elements_crops_npy_root, img_filename + ".npy"), allow_pickle=True)
            text_crops = np.load(os.path.join(ui_elements_crops_npy_root, img_filename+"_texts.npy"), allow_pickle=True)
            crops_info[img_filename] = {'content': crops, 'text': text_crops}    

        # we reduce their size to adapt them to the neural network entry
        for i in range(0, len(crops_info)):
            preprocessed_crops_filename = os.path.join(ui_elements_crops_npy_root, "preprocessed_" + screenshot_filenames[i] + ".npy")
            preprocessed_crops_exists = os.path.exists(preprocessed_crops_filename)
            
            if not preprocessed_crops_exists:
                x = ui_elements_classification_image_shape[0]
                y = ui_elements_classification_image_shape[1]
                preprocessed_crops = []
                for img in crops_info[screenshot_filenames[i]]["content"]:
                    if img.shape[1] > y:
                        img = img[0:img.shape[0], 0:y]
                    if img.shape[0] > x:
                        img = img[0:x, 0:img.shape[1]]
                    crop_padded = pad(img, x, y)
                    crop_resized = tf.image.resize(crop_padded, [50, 50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True, antialias=True)
                    preprocessed_crops.append(crop_resized)
                    # print("Cropped: "+str(img_resized.shape))
                crops_info[screenshot_filenames[i]]["content_preprocessed"] = preprocessed_crops

                # We store all preprocessed crops
                preprocessed_crops_npy = np.array(crops_info[screenshot_filenames[i]]["content_preprocessed"])
                np.save(preprocessed_crops_filename, preprocessed_crops_npy)
            else:
                preprocessed_crops_npy = np.load(preprocessed_crops_filename, allow_pickle=True)

            predict_x = loaded_model.predict(preprocessed_crops_npy)

            # This neural network returns as output a integer indicating each of its classes. This number must me mapped to its corresponding class name (str)
            result = np.argmax(predict_x, axis=1)
            result_mapped = [text_classname if crops_info[screenshot_filenames[i]]["text"][index] else ui_elements_classification_classes[x] for index, x in enumerate(result)]

            crops_info[screenshot_filenames[i]]["result"] = result_mapped
            # crop_imgs[screenshot_filenames[i]]["result_freq"] = pd.Series(result_mapped).value_counts()
            # crop_imgs[screenshot_filenames[i]]["result_freq_df"] = crop_imgs[screenshot_filenames[i]]["result_freq"].to_frame().T
            
            # Update the json file with components
            with open(os.path.join(metadata_json_root, screenshot_filenames[i] + '.json'), 'r') as f:
                data = json.load(f)
            for j in range(0, len(result_mapped)):
                data["compos"][j]["class"] = result_mapped[j]
            with open(os.path.join(metadata_json_root, screenshot_filenames[i] + '.json'), "w") as jsonFile:
                json.dump(data, jsonFile)
