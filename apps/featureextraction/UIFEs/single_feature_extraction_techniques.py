import os
import json
import pandas as pd
from core.utils import read_ui_log_as_dataframe
from core.settings import STATUS_VALUES_ID, ENRICHED_LOG_SUFFIX, sep
from core.utils import read_ui_log_as_dataframe


def ui_compos_stats(ui_log_path, path_scenario, execution, fe):
    """
    Add to each compo_json a key named 'features' with the number of UI Components, UI Groups, UI Elements
    """
    ui_elements_classification_classes = execution.ui_elements_classification.model.classes
    # decision_point = fe.decision_point_activity
    case_colname = execution.case_study.special_colnames["Case"]
    activity_colname = execution.case_study.special_colnames["Activity"]
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    metadata_json_root = os.path.join(path_scenario, 'components_json')
    flattened_log = os.path.join(path_scenario, 'flattened_dataset.json')
    enriched_log_output = path_scenario + fe.technique_name+'_enriched_log.csv'
    text_classname = execution.case_study.ui_elements_classification.text_classname
    consider_relevant_compos = fe.consider_relevant_compos
    relevant_compos_predicate = fe.relevant_compos_predicate
    id = fe.identifier
    
    log = read_ui_log_as_dataframe(ui_log_path)

    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    headers = dict()
    info_to_join: dict[str:list] = {}

    for elem in ui_elements_classification_classes:
        headers[elem] = 0

    num_screenshots = len(screenshot_filenames)

    df = pd.DataFrame(columns=["screenshot", "#UICompos", "#UIElements", "#UIGroups"])
    
    for i, screenshot_filename in enumerate(screenshot_filenames):
        num_UI_compos = 0
        num_UI_elements = 0
        num_UI_groups = 0
        # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
        with open(metadata_json_root + screenshot_filename + '.json', 'r') as f:
            data = json.load(f)

        screenshot_compos_frec = headers.copy()
        
        if consider_relevant_compos:
            compos_list = [ compo for compo in data["compos"] if eval(relevant_compos_predicate)]
        else:
            compos_list = data["compos"]

        for j in range(0, len(compos_list)):
            # Centroid is Already calculated in the prediction
            # compo_x1 = compos_list[j]["column_min"]
            # compo_y1 = compos_list[j]["row_min"]
            # compo_x2 = compos_list[j]["column_max"]
            # compo_y2 = compos_list[j]["row_max"]
            # centroid_y = (compo_y2 - compo_y1 / 2) + compo_y1
            # centroid_x = (compo_x2 - compo_x1 / 2) + compo_x1
            # compos_list[j]["centroid"] = [centroid_x, centroid_y]
            screenshot_compos_frec[compos_list[j]["class"]] += 1
            
            column_name = compos_list[j]["class"]+"_"+str(screenshot_compos_frec[compos_list[j]["class"]])

            if column_name in info_to_join:
                if not len(info_to_join[column_name]) == i:
                    for k in range(len(info_to_join[column_name]),i):
                        info_to_join[column_name].append("")
                info_to_join[column_name].append(compos_list[j]["centroid"])
            else:
                column_as_vector = []
                for k in range(0,i):
                    column_as_vector.append("")
                column_as_vector.append(compos_list[j]["centroid"])
                info_to_join[column_name] = column_as_vector
            
            if len(compos_list[j]["contain"]) > 0 or compos_list[j]["contain"] == "UIGroup":
                num_UI_groups+=1
            else:
                num_UI_elements+=1
            
            num_UI_compos += 1
                
        if not "features" in data:
            data["features"] = {}
            
        data["features"]["location"] = info_to_join
        data["features"]["#UICompos"] = num_UI_compos
        data["features"]["#UIElements"] = num_UI_elements
        data["features"]["#UIGroups"] = num_UI_groups
        new_row = [screenshot_filename, num_UI_compos, num_UI_elements, num_UI_groups]
        df.loc[i] = new_row
        with open(metadata_json_root + screenshot_filename + '.json', "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
            
    df.to_csv(metadata_json_root + ENRICHED_LOG_SUFFIX + ".csv", index=False)
            
        

    print("\n\n=========== ENRICHED COMPO_JSON: path=" + metadata_json_root + 'XXXXXX.json')
    
    return num_UI_compos, num_screenshots, None, None

# ========================================================================================================
# Centroid as values / class as column name
# ========================================================================================================

def aux_iterate_compos(ui_log_path, path_scenario, execution, fe, centroid_columnname_type):
    """
    Column name: compoclass+int
    Column value: centroid 
    """
    execution_root = path_scenario + '_results'
    metadata_json_root = os.path.join(execution_root, 'components_json')
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    consider_relevant_compos = fe.consider_relevant_compos
    relevant_compos_predicate = fe.relevant_compos_predicate
    ui_elements_classification_classes = execution.ui_elements_classification.model.classes
    #decision_point = fe.decision_point_activity
    id = fe.identifier
    case_colname = execution.case_study.special_colnames["Case"]
    activity_colname = execution.case_study.special_colnames["Activity"]
    flattened_log = os.path.join(execution_root, 'flattened_dataset.json')
    enriched_log_output = os.path.join(execution_root, fe.technique_name + '_enriched_log.csv')
    text_classname = execution.ui_elements_classification.model.text_classname
    
    log = read_ui_log_as_dataframe(ui_log_path)

    enriched_log = log.copy()
    
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    headers = dict()
    info_to_join: dict[str:list] = {}

    for elem in ui_elements_classification_classes:
        headers[elem] = 0

    new_columns = pd.DataFrame(index=log.index)
    num_screenshots = len(screenshot_filenames)
    num_UI_elements = 0
    max_num_UI_elements = 0
    min_num_UI_elements = 99999999999999999

    for i, screenshot_filename in enumerate(screenshot_filenames):
        screenshot_filename = os.path.basename(screenshot_filename)
        
        # Check if the file exists, if exists, then we can continue
        if os.path.exists(os.path.join(metadata_json_root, screenshot_filename + '.json')):
            with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), 'r') as f:
                data = json.load(f)

            screenshot_compos_frec = headers.copy()
            
            if consider_relevant_compos:
                compos_list = [ compo for compo in data["compos"] if eval(relevant_compos_predicate)]
            else:
                compos_list = data["compos"]

            for j in range(0, len(compos_list)):
                compo_class = compos_list[j]["class"]
                try:
                    screenshot_compos_frec[compo_class] += 1
                except:
                    raise Exception("UI Elements Detection model classes not compatible with Preloaded FE files ones: please select the correct model")

# ========================================================================================================
# ========================================================================================================
                if centroid_columnname_type == "class_as_colname":
                    column_name = f"{id}_{compo_class}_{str(screenshot_compos_frec[compo_class])}"

                    if column_name in info_to_join:
                        if not len(info_to_join[column_name]) == i:
                            for k in range(len(info_to_join[column_name]),i):
                                info_to_join[column_name].append("")
                        info_to_join[column_name].append(compos_list[j]["centroid"])
                
                        enriched_log.at[i, column_name] = compos_list[j]["centroid"]  # Añade el centroide a la fila y columna correspondiente
                    else:
                        column_as_vector = []
                        for k in range(0,i):
                            column_as_vector.append("")
                        column_as_vector.append(compos_list[j]["centroid"])
                        info_to_join[column_name] = column_as_vector
                        
                        enriched_log[column_name] = [''] * num_screenshots  # Inicializa la nueva columna con valores vacíos
                        enriched_log.at[i, column_name] = compos_list[j]["centroid"]  # Añade el centroide a la fila y columna correspondiente
# ========================================================================================================
# ========================================================================================================
                elif centroid_columnname_type == "classplaintext_as_colname":
                    if compo_class == text_classname:
                        aux = compos_list[j]["text"]
                        column_name = f"{id}_{aux}_{str(screenshot_compos_frec[aux])}"
                    else:
                        aux = compo_class
                        column_name = f"{id}_{compo_class}_{str(screenshot_compos_frec[compo_class])}"

                    screenshot_compos_frec[aux] += 1
                    
                    if column_name in info_to_join:
                        if not len(info_to_join[column_name]) == i:
                            for k in range(len(info_to_join[column_name]),i):
                                info_to_join[column_name].append("")
                        info_to_join[column_name].append(compos_list[j]["centroid"])
                        
                        enriched_log.at[i, column_name] = compos_list[j]["centroid"]  # Añade el centroide a la fila y columna correspondiente
                    else:
                        column_as_vector = []
                        for k in range(0,i):
                            column_as_vector.append("")
                        column_as_vector.append(compos_list[j]["centroid"])
                        info_to_join[column_name] = column_as_vector
                        
                        enriched_log[column_name] = [''] * num_screenshots  # Inicializa la nueva columna con valores vacíos
                        enriched_log.at[i, column_name] = compos_list[j]["centroid"]  # Añade el centroide a la fila y columna correspondiente
                
# ========================================================================================================
# ========================================================================================================
                elif centroid_columnname_type == "centroid_class":
                    centroid = compos_list[j]["centroid"]
                    activity = log.at[i, activity_colname]
                    column_name = f"{id}_{centroid[0]}-{centroid[1]}_{activity}"
                    
                    if column_name in info_to_join:
                        if not len(info_to_join[column_name]) == i:
                            for k in range(len(info_to_join[column_name]),i):
                                info_to_join[column_name].append("")
                        info_to_join[column_name].append(compos_list[j]["class"])
                        
                        enriched_log.at[i, column_name] = compos_list[j]["class"]  # Añade el centroide a la fila y columna correspondiente
                    else:
                        column_as_vector = []
                        for k in range(0,i):
                            column_as_vector.append("")
                        column_as_vector.append(compos_list[j]["class"])
                        info_to_join[column_name] = column_as_vector
                        
                        if column_name not in new_columns:
                            enriched_log[column_name] = [''] * num_screenshots  # Inicializa la nueva columna con valores vacíos
                        enriched_log.at[i, column_name] = compos_list[j]["class"]  # Añade el centroide a la fila y columna correspondiente
# ========================================================================================================
# ========================================================================================================
                elif centroid_columnname_type == "centroid_classplaintext":
                    raise Exception("Not implemented yet")
# ========================================================================================================
# ========================================================================================================
                elif centroid_columnname_type == "xpath_class":
                    xpath = compos_list[j]["xpath"]
                    activity = log.at[i, activity_colname]
                    column_name = f"{id}_{xpath}_{activity}"
                    
                    if column_name in info_to_join:
                        if not len(info_to_join[column_name]) == i:
                            for k in range(len(info_to_join[column_name]),i):
                                info_to_join[column_name].append("")
                        info_to_join[column_name].append(compos_list[j]["class"])
                        
                        enriched_log.at[i, column_name] = compos_list[j]["class"]  # Añade el centroide a la fila y columna correspondiente
                    else:
                        column_as_vector = []
                        for k in range(0,i):
                            column_as_vector.append("")
                        column_as_vector.append(compos_list[j]["class"])
                        info_to_join[column_name] = column_as_vector
                        
                        if column_name not in new_columns:
                            enriched_log[column_name] = [''] * num_screenshots  # Inicializa la nueva columna con valores vacíos
                        enriched_log.at[i, column_name] = compos_list[j]["class"]  # Añade el centroide a la fila y columna correspondiente

                else:
                    raise Exception("UIFE: centroid_columnname_type not recognized")
                # num_UI_elements += 1
            if "features" in data:
                data["features"]["location"] = info_to_join
            else:
                data["features"] = { "location": info_to_join }
            
            with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), "w") as jsonFile:
                json.dump(data, jsonFile)
                
            # print("\n\n=========== ENRICHED LOG GENERATED: path=" + enriched_log_output)
            enriched_log.to_csv(os.path.join(execution_root, "log" + ENRICHED_LOG_SUFFIX + ".csv"), index=False)
                
        else:
            print("File not found: " + os.path.join(metadata_json_root, screenshot_filename + '.json'))

    
    
    return num_UI_elements, num_screenshots, max_num_UI_elements, min_num_UI_elements


def class_centroid_ui_element(ui_log_path, path_scenario, execution, fe):
    """
    Column name: compoclass+int
    Column value: centroid 
    """
    return aux_iterate_compos(ui_log_path, path_scenario, execution, fe, "class_as_colname")

def class_or_plaintext_centroid_ui_element(ui_log_path, path_scenario, execution, fe):
    """
    Column name: compoclass+int or (if it is text) plaintext+int
    Column value: centroid 
    """
    return aux_iterate_compos(ui_log_path, path_scenario, execution, fe, "classplaintext_as_colname")

def centroid_ui_element_class(ui_log_path, path_scenario, execution, fe):
    """
    Column name: centroid 
    Column value: compoclass+int
    """
    return aux_iterate_compos(ui_log_path, path_scenario, execution, fe, "centroid_class")

def centroid_ui_element_class_or_plaintext(ui_log_path, path_scenario, execution, fe):
    """
    Column name: centroid 
    Column value: compoclass+int or (if it is text) plaintext+int
    """
    return aux_iterate_compos(ui_log_path, path_scenario, execution, fe, "centroid_classplaintext")

# ========================================================================================================
# Class as value / xpath to reach ui element as column name
# ========================================================================================================
def xpath_class(ui_log_path, path_scenario, execution, fe):
    """
    Column name: compoclass+int or (if it is text) plaintext+int
    Column value: centroid 
    """
    return aux_iterate_compos(ui_log_path, path_scenario, execution, fe, "xpath_class")

# ========================================================================================================
# Boolean if exists as value / xpath to reach ui element as column name
# ========================================================================================================
def xpath_ui_elem_class_existence(ui_log_path, path_scenario, execution, fe):
    raise Exception("Not implemented yet")
    
# ========================================================================================================
# Boolean if exists as value / xpath to reach ui compo as column name
# ========================================================================================================
def rec_aux_process_ui_element(element, parent_ids, enriched_log, index):
    # Construir la cadena de identificadores de este elemento
    element_id_chain = '_'.join(parent_ids + [str(element['id'])])
    # Añadir el elemento al DataFrame
    enriched_log.at[index, element_id_chain] = 1  # Establecer valor a 1 para indicar presencia del elemento
    # Recorrer recursivamente los hijos, si los hay
    for child in element.get('children', []):
        rec_aux_process_ui_element(child, parent_ids + [str(element['id'])], enriched_log, index)


def ui_compo_existence(ui_log_path, path_scenario, execution, _):
    execution_root = path_scenario + '_results'
    metadata_json_root = os.path.join(execution_root, 'components_json')
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    
    # Leer el log original
    log = read_ui_log_as_dataframe(ui_log_path)
    enriched_log = log.copy()  # Crear una copia para enriquecerla

    # Procesar cada captura de pantalla asociada en el log
    for index, row in enriched_log.iterrows():
        screenshot_filename = os.path.basename(row[screenshot_colname])
        json_path = os.path.join(metadata_json_root, screenshot_filename + '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'som' in data:  # Asumiendo que 'som' es la clave para la estructura de la UI
                    rec_aux_process_ui_element(data['som'], [], enriched_log, index)
                else:
                    raise Exception("UI Hierarchy (Screen-Object Model) not found in " + json_path)

    # Guardar el log enriquecido en un nuevo archivo
    enriched_log.to_csv(metadata_json_root + ENRICHED_LOG_SUFFIX + ".csv", index=False)