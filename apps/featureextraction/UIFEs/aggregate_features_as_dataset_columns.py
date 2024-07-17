import json
import os
import re
import numpy as np
from core.settings import STATUS_VALUES_ID, PROCESS_DISCOVERY_LOG_FILENAME
from core.utils import read_ui_log_as_dataframe
from tqdm import tqdm
from shapely.geometry import Polygon, Point

def find_st_id(st_value):
    res = None
    f = open(STATUS_VALUES_ID)
    statuses = json.load(f)
    for k in statuses.keys():
        if st_value in statuses[k]:
            res = k
            break
    return res

def occurrence_ui_element_class(ui_log_path, path_scenario, execution, fe):
    """
    Since not all images have all classes, a dataset with different columns depending on the images will be generated.
    It will depend whether GUI components of every kind appears o only a subset of these. That is why we initiañize a 
    dataframe will all possible columns, and include row by row the results obtained from the predictions

    Column name: compoclass (13 columns -> 13 different classes)
    Column value: how many occurrence of this type of UI elements appears in the screenshot

    :param feature_extraction_technique_name: Feature extraction technique name
    :type feature_extraction_technique_name: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param skip: Rewrite log
    :type skip: bool

    """
    ui_elements_classification_classes = execution.ui_elements_classification.model.classes
    # decision_point = execution.feature_extraction_technique.decision_point_activity
    case_colname = execution.case_study.special_colnames["Case"]
    activity_colname = execution.case_study.special_colnames["Activity"]
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    metadata_json_root = os.path.join(path_scenario, 'components_json')
    flattened_log = os.path.join(path_scenario, 'flattened_dataset.json')
    enriched_log_output = path_scenario + fe.technique_name+'_enriched_log.csv',
    text_classname = execution.case_study.ui_elements_classification.text_classname,
    consider_relevant_compos = fe.consider_relevant_compos,
    relevant_compos_predicate = fe.relevant_compos_predicate,
    id = fe.identifier
    
    with open(flattened_log, 'r') as f:
        ui_log_data = json.load(f)
    
    log = read_ui_log_as_dataframe(ui_log_path)
    
    # df = pd.DataFrame([], columns=ui_elements_classification_classes)
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    quantity_ui_elements = {}
    before_DP = True
    aux_case = -1    

    num_screenshots = len(screenshot_filenames)
    num_UI_elements = 0
    max_num_UI_elements = 0
    min_num_UI_elements = 99999999999999999
    

    for i, screenshot_filename in enumerate(screenshot_filenames):
        case = log.at[i, case_colname]
        activity = log.at[i, activity_colname]
        
        if case != aux_case:
            before_DP = True
        
        if before_DP:
            # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
            with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), 'r') as f:
                data = json.load(f)
                
            if consider_relevant_compos:
                compos_list = [ compo for compo in data["compos"] if eval(relevant_compos_predicate)]
            else:
                compos_list = data["compos"]

            num_UI_elements += len(compos_list)
            if len(compos_list) > max_num_UI_elements:
                max_num_UI_elements = len(compos_list)
            if len(compos_list) < min_num_UI_elements:
                min_num_UI_elements = len(compos_list)
            
            for c in ui_elements_classification_classes:
                counter = 0
                if "_" in c:
                    st = c.split("_")
                    for j in range(0, len(compos_list)):
                        ui_elem = compos_list[j]
                        st_key = find_st_id(st[1])
                        if (ui_elem["class"] == st[0]) and (st_key in ui_elem) and (ui_elem[st_key] == st[1]):
                            counter+=1
                    quantity_ui_elements[c] = counter
                    ui_log_data[str(case)][id+"_"+c+"_"+activity] = counter
                else:
                    for j in range(0, len(compos_list)):
                        if compos_list[j]["class"] == c:
                            counter+=1
                    quantity_ui_elements[c] = counter
                    ui_log_data[str(case)][id+"_"+c+"_"+activity] = counter

            if "features" in data:
                data["features"][id] = quantity_ui_elements
            else:
                data["features"] = { id: quantity_ui_elements }
            
            # df.loc[i] = quantity_ui_elements.values()
    
                
            with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), "w") as jsonFile:
                json.dump(data, jsonFile, indent=4)
                
            # if activity == decision_point:
            #     aux_case = case
            #     before_DP = False

    with open(flattened_log, 'w') as f:
        json.dump(ui_log_data, f, indent=4)

    return num_UI_elements, num_screenshots, max_num_UI_elements, min_num_UI_elements

def state_ui_element_centroid(ui_log_path, path_scenario, execution, fe):
    """
    
    Column name: "sta_"+ substate + centroid_x + centroid_y + activity. Example: sta_enabled_229.5-1145.0_1_A
    Column value: state. Example: enabled
    
    ------------
    
    Only Leaf UI Elements or Simple UI Elements can have an associated state. Composed UI Elements (forms, dialogs, sheets...) have another information associated
    like caption, or key-value pairs...
    
    States communicate the status of UI elements to the user. Each state should be visually similar and not drastically alter a component, but must have clear affordances that distinguish it from other states and the surrounding layout.
    States must have clear affordances that distinguish them from one other.

    Types of states

    Enabled: An enabled state communicates an interactive component or element.
    Disabled: A disabled state communicates a noninteractive component or element.
    Hover: A hover state communicates when a user has placed a cursor above an interactive element.
    Focused: A focused state communicates when a user has highlighted an element, using an input method such as a keyboard or voice.
    Selected: A selected state communicates a user choice.
    Activated: An activated state communicates a highlighted destination, whether initiated by the user or by default.
    Pressed: A pressed state communicates a user tap.
    Dragged: A dragged state communicates when a user presses and moves an element.
    
    Ref. https://m2.material.io/design/interaction/states.html#usage
    
    Restrictions:
    (1) FABs (2) bottom sheets and (3) app bars cannot inherit a disabled state.
    Disabled components cannot be (1) hovered, (2) focused, (3) dragged or (4) pressed.
    (1) Sheets, (2) app bars or (3) dialogs cannot inherit a hover state
    Components that can't inherit a focus state include: (1) whole sheets, (2) whole app bars or (3) whole dialogs.
    (1) Buttons, (2) text fields, (3) app bars, and (4) dialogs can't inherit a selected state.
    (1) Buttons and (2) dialogs cannot inherit an activated state.
    Components such as (1) sheets, (2) app bars, or (3) dialogs cannot inherit a pressed state
    Components such as (1) buttons, (2) app bars, (3) dialogs, or (4) text fields cannot inherit a dragged state
    """
    execution_root = path_scenario + '_results'
    # decision_point = fe.feature_extraction_technique.decision_point_activity
    case_colname = execution.case_study.special_colnames["Case"]
    activity_colname = execution.case_study.special_colnames["Activity"]
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    metadata_json_root = os.path.join(execution_root, 'components_json')
    flattened_log = os.path.join(execution_root, 'flattened_dataset.json')
    consider_relevant_compos = fe.consider_relevant_compos
    relevant_compos_predicate = fe.relevant_compos_predicate
    id = fe.identifier
    
    # log = read_ui_log_as_dataframe(ui_log_path)
    log = read_ui_log_as_dataframe(os.path.join(path_scenario + "_results", PROCESS_DISCOVERY_LOG_FILENAME))

    # df = pd.DataFrame([], columns=ui_elements_classification_classes)
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    # Guardar en la variable flattened_logs los nombres de los archivos que en la ruta de execution_root cumplan el patron de que empiezan por "flattened_dataset" y terminan por ".json"
    flattened_logs = [os.path.join(execution_root, f) for f in os.listdir(execution_root) if f.startswith('flattened_dataset') and f.endswith('.json')]

    for flattened_log in flattened_logs:
        with open(flattened_log, 'r') as f:
            ui_log_data = json.load(f)
        
        before_DP = True
        aux_case = -1    

        num_screenshots = 0
        num_UI_elements = 0
        max_num_UI_elements = 0
        min_num_UI_elements = 99999999999999999
        
        for i, screenshot_filename in enumerate(screenshot_filenames):
            case = log.at[i, case_colname]
            activity = log.at[i, activity_colname]
            
            if case != aux_case:
                before_DP = True
            
            if before_DP:
                # This network gives as output the name of the detected class. Additionally, we moddify the json file with the components to add the corresponding classes
                with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), 'r') as f:
                    data = json.load(f)

                if consider_relevant_compos:
                    compos_list = [ compo for compo in data["compos"] if eval(relevant_compos_predicate)]
                else:
                    compos_list = data["compos"]
                
                num_UI_elements += len(compos_list)
                if len(compos_list) > max_num_UI_elements:
                    max_num_UI_elements = len(compos_list)
                if len(compos_list) < min_num_UI_elements:
                    min_num_UI_elements = len(compos_list)

                for j in range(0, len(compos_list)):
                    # Centroid is Already calculated in the prediction
                    # compo_x1 = compos_list[j]["column_min"]
                    # compo_y1 = compos_list[j]["row_min"]
                    # compo_x2 = compos_list[j]["column_max"]
                    # compo_y2 = compos_list[j]["row_max"]
                    # centroid_y = (compo_y2 - compo_y1 / 2) + compo_y1
                    # centroid_x = (compo_x2 - compo_x1 / 2) + compo_x1
                    # compos_list[j]["centroid"] = [centroid_x, centroid_y]

                    # Status columns 
                    status_columns = [i for i in dict(compos_list[j]).keys() if "st_" in i]
                    centroid = compos_list[j]["centroid"]
                    for status_col in status_columns:
                        status = compos_list[j][status_col]
                        sub_id = str(status_col).split("_")[1]
                        ui_log_data[str(case)][id+"_"+sub_id+"_"+str(centroid[0])+"-"+str(centroid[1])+"_"+str(activity)] = status
                num_screenshots += 1
                    
                with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), "w") as jsonFile:
                    json.dump(data, jsonFile, indent=4)
                    
                # if decision_point in activity:
                #     aux_case = case
                #     before_DP = False

        with open(flattened_log, 'w') as f:
            json.dump(ui_log_data, f, indent=4)
        
    # print("\n\n=========== ENRICHED LOG GENERATED: path=" + enriched_log_output)
    return num_UI_elements, num_screenshots, max_num_UI_elements, min_num_UI_elements

def combine_ui_element_centroid(ui_log_path, path_scenario, execution, fe):
    """
    Combine the information of the UI elements and the centroids of the UI elements in the same dataset
    for the same activities
    """
    # Iterate over the images again to find for each centroid the smallest object containing it
    execution_root = path_scenario + '_results'
    metadata_json_root = os.path.join(execution_root, 'components_json')
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    consider_relevant_compos = fe.consider_relevant_compos
    relevant_compos_predicate = fe.relevant_compos_predicate
    id = fe.identifier
    
    log = read_ui_log_as_dataframe(os.path.join(path_scenario + "_results", "pipeline_log.csv"))
    activities = list(set(log.loc[:, execution.case_study.special_colnames["Activity"]].values.tolist()))

    for activity in activities:
        rows = log[log[execution.case_study.special_colnames["Activity"]] == activity]
        for index, row in tqdm(rows.iterrows(), desc="Updating centroids with classes for each screenshot"):
            screenshot_filename = os.path.basename(row[screenshot_colname])

            # Check if the file exists, if exists, then we can continue
            if os.path.exists(os.path.join(metadata_json_root, screenshot_filename + '.json')):
                with open(os.path.join(metadata_json_root, screenshot_filename + '.json'), 'r') as f:
                    data = json.load(f)
                
                # Both components and centroids as numpy arrays to make it more performant
                if consider_relevant_compos:
                    compos_nparray = np.array([ compo for compo in data["compos"] if eval(relevant_compos_predicate)])
                else:
                    compos_nparray = np.array(list(filter(lambda x: x["relevant"] == True, data["compos"])))

                # identifier_-centroidY
                centroid_regex = re.compile(rf"{id}_(\d*\.?\d+)-(\d*\.?\d+)")
                # Get all the columns that match the regex and do not contain only nan values
                centroid_columns = [col for col in rows.columns if centroid_regex.match(col) and not rows[col].isnull().all()]
                centroids = np.array([np.array([centroid_regex.match(centroid).groups()[0], centroid_regex.match(centroid).groups()[1]]) for centroid in centroid_columns])

                # Pre-compute Polygon objects to avoid creating them in each iteration
                compos_polygons = [(Polygon(compo["points"]), compo) for compo in compos_nparray]

                # Match each centroid with the smallest object containing it using Polygon from shapely
                for centroid in centroids:
                    centroid_point = Point(centroid.astype(float))
                    containing_compos = [(compo, poly.area) for poly, compo in compos_polygons if poly.contains(centroid_point)] 
                    if len(containing_compos) == 0:
                        continue
                    compo = min(containing_compos, key=lambda x: x[1])[0]

                    # Insert the class of the smallest object containing the centroid
                    log.at[index, f"{id}_{centroid[0]}-{centroid[1]}"] = compo["class"]
    
    # Copy trace_id column because it gets deleted sometimes
    trace = log["trace_id"]
    # Remove columns with the same values
    log = log.T.drop_duplicates().T
    log["trace_id"] = trace
    log.to_csv(os.path.join(execution_root, "pipeline_log.csv"), index=False)
    return 0,0,0,0

