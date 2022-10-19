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
from featureextraction.CNN.CompDetCNN import CompDetCNN

default_ui_elements_classification_classes = ['x0_Button', 'x0_CheckBox', 'x0_CheckedTextView', 'x0_EditText', 'x0_ImageButton', 'x0_ImageView', 'x0_NumberPicker', 'x0_RadioButton', 'x0_RatingBar', 'x0_SeekBar', 'x0_Spinner', 'x0_Switch', 'x0_TextView', 'x0_ToggleButton']


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
    log = pd.read_csv(ui_log_path, sep=",")
    # screenshot_filenames = [ x + ".npy" for x in log.loc[:,"Screenshot"].values.tolist()]
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    missing_json_file = False

    for screenshot in screenshot_filenames:
        if not os.path.exists(metadata_json_root + screenshot + '.json'):
            missing_json_file = True
            break

    return screenshot_filenames, missing_json_file

###################################################################################################
###################################################################################################

def uied_ui_elements_classification(model_weights="resources/models/custom-v2.h5", model_properties="resources/models/custom-v2-classes.json", ui_elements_crops_npy_root="resources/screenshots/components_npy/",
                            metadata_json_root="resources/screenshots/components_json/", ui_log_path="resources/log.csv", screenshot_colname="Screenshot", 
                            rewrite_info=False, ui_elements_classification_classes=default_ui_elements_classification_classes):
    """
    With this function we classify the copped component from each of the sreenshots to later add to the log the number of
    columns corresponding to the ammount to classes in the given model. These are the classes that a GUI component can fall into.
    The values indicated in these columns added indicate how many GUI components are present with regard to their class.
    Example. 2 button, 3 image_view, 1 text_view, etc.
    
    :param_model_weights: Classification model  
    :type param_model_weights: h5
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
    :param rewrite_info: Rewrite classification data (json files)
    :type rewrite_info: bool
    
    """

    screenshot_filenames, missing_json_file = check_metadata_json_exists(ui_log_path, screenshot_colname, metadata_json_root)

    if missing_json_file or rewrite_info:
        # Load the model properties from the json
        f = json.load(open(model_properties,))
        classes = f["classes"]
        shape = tuple(f["shape"])

        # Load the ML classifier model for the crops
        # Default model is custom-v2, a model creating by using transfer learning from UIED's generalized model
        classifier = {}
        classifier['Elements'] = CompDetCNN(
            model_weights, classes, shape)
        print("\n\nLoaded ML model from disk\n")

        """
        Load the crops calculated in get_(uied)_gui_components_crops
        """
        log = pd.read_csv(ui_log_path, sep=",")

        crop_imgs = {}
        # images_names = [ x + ".npy" for x in log.loc[:,"Screenshot"].values.tolist()] # os.listdir(ui_elements_crops_npy_root)
        images_names = log.loc[:, screenshot_colname].values.tolist()
        # print(images_names)
        for img_filename in images_names:
            crop_img_aux = np.load(ui_elements_crops_npy_root+img_filename +
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
            with open(metadata_json_root + images_names[i] + '.json', 'r') as f:
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
            with open(metadata_json_root + images_names[i] + '.json', "w") as jsonFile:
                json.dump(data, jsonFile)

    #     """
    #     Since not all images have all classes, a dataset with different columns depending on the images will be generated.
    #     It will depend whether GUI components of every kind appears o only a subset of these. That is why we initiañize a 
    #     dataframe will all possible columns, and include row by row the results obtained from the predictions
    #     """

    #     nombre_clases = classifier["Elements"].class_map
    #     df = pd.DataFrame([], columns=nombre_clases)

    #     for i in range(0, len(images_names)):
    #         row1 = [0 for i in range(0, len(nombre_clases))]
    #         # Acess the previously stored frequencies
    #         df1 = crop_imgs[images_names[i]]["result_freq_df"]
    #         if len(df1.columns.tolist()) > 0:
    #             for x in df1.columns.tolist():
    #                 uiui = nombre_clases.index(x)
    #                 row1[uiui] = df1[x][0]
    #                 df.loc[i] = row1

    #     """
    #     Once the dataset corresponding to the ammount of elements of each class contained in each of the images is obtained,
    #     we merge it with the complete log, adding the extracted characteristics from the images
    #     """

    #     log_enriched = log.join(df).fillna(method='ffill')

    #     """
    #     Finally we obtain an entiched log, which is turned as proof of concept of our hypothesis based on the premise that if
    #     we not only capture keyboard or mouse events on the monitorization through a keylogger, but also screencaptures,
    #     we can extract much more useful information, being able to improve the process mining over said log.
    #     As a pending task, we need to validate this hypothesis through a results comparison against the non-enriched log.
    #     We expect to continue this project in later stages of the master
    #     """
    #     log_enriched.to_csv(enriched_log_output_path)
    #     print("\n\n=========== ENRICHED LOG GENERATED: path=" +
    #           enriched_log_output_path)
    # else:
    #     log_enriched = pd.read_csv(
    #         enriched_log_output_path, sep=",", index_col=0)
    #     print("\n\n=========== ENRICHED LOG ALREADY EXISTS: path=" +
    #           enriched_log_output_path)
    # return log_enriched


def legacy_ui_elements_classification(model_weights="resources/models/model.h5", model_properties="resources/models/model.json", ui_elements_crops_npy_root="resources/screenshots/components_npy/",
                            metadata_json_root="resources/screenshots/components_json/", ui_log_path="resources/log.csv", screenshot_colname="Screenshot", 
                            rewrite_info=False, ui_elements_classification_classes=default_ui_elements_classification_classes):
    """
    With this function we classify the copped component from each of the sreenshots to later add to the log the number of
    columns corresponding to the ammount to classes in the given model. These are the classes that a GUI component can fall into.
    The values indicated in these columns added indicate how many GUI components are present with regard to their class.
    Example. 2 button, 3 image_view, 1 text_view, etc.

    :param_model_weights: Weights of the edges of the classification neural network 
    :type param_model_weights: h5
    :param model_properties:
    :type model_properties: json file
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
    :param rewrite_info: Rewrite classification data (json files)
    :type rewrite_info: bool
    :returns: Enriched log
    :rtype: DataFrame
    """

    screenshot_filenames, missing_json_file = check_metadata_json_exists(ui_log_path, screenshot_colname, metadata_json_root)

    if missing_json_file or rewrite_info:
        # load json and create model
        json_file = open(model_properties, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(model_weights)

        crops_info = {}
        
        for img_filename in screenshot_filenames:
            crops = np.load(ui_elements_crops_npy_root + img_filename + ".npy", allow_pickle=True)
            text_crops = np.load(ui_elements_crops_npy_root + img_filename+"_texts.npy", allow_pickle=True)
            crops_info[img_filename] = {'content': crops, 'text': text_crops}    

        # we reduce their size to adapt them to the neural network entry
        for i in range(0, len(crops_info)):
            preprocessed_crops_filename = ui_elements_crops_npy_root + "preprocessed_" + screenshot_filenames[i] + ".npy"
            preprocessed_crops_exists = os.path.exists(preprocessed_crops_filename)
            
            if not preprocessed_crops_exists:
                preprocessed_crops = []
                for img in crops_info[screenshot_filenames[i]]["content"]:
                    if img.shape[1] > 150:
                        img = img[0:img.shape[0], 0:150]
                    if img.shape[0] > 150:
                        img = img[0:150, 0:img.shape[1]]
                    crop_padded = pad(img, 150, 150)
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
            result_mapped = ["x0_TextView" if crops_info[screenshot_filenames[i]]["text"][index] else ui_elements_classification_classes[x] for index, x in enumerate(result)]

            crops_info[screenshot_filenames[i]]["result"] = result_mapped
            # crop_imgs[screenshot_filenames[i]]["result_freq"] = pd.Series(result_mapped).value_counts()
            # crop_imgs[screenshot_filenames[i]]["result_freq_df"] = crop_imgs[screenshot_filenames[i]]["result_freq"].to_frame().T
            
            # Update the json file with components
            with open(metadata_json_root + screenshot_filenames[i] + '.json', 'r') as f:
                data = json.load(f)
            for j in range(0, len(result_mapped)):
                data["compos"][j]["class"] = result_mapped[j]
            with open(metadata_json_root + screenshot_filenames[i] + '.json', "w") as jsonFile:
                json.dump(data, jsonFile)
