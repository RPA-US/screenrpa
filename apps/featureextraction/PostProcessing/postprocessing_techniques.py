import os
import json
import re
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import pandas as pd
from apps.featureextraction.utils import read_ui_log_as_dataframe

def combine_ui_element_centroid_aux(ui_log_path, path_scenario, execution, pp, use_text):
    """
    Combine the information of the UI elements and the centroids of the UI elements in the same dataset
    for the same activities
    """
    # Iterate over the images again to find for each centroid the smallest object containing it
    execution_root = path_scenario + '_results'
    metadata_json_root = os.path.join(execution_root, 'components_json')
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    text_classname = execution.ui_elements_classification.model.text_classname

    if not os.path.exists(os.path.join(path_scenario + "_results", "pipeline_log.csv")):
        fe_log = read_ui_log_as_dataframe(os.path.join(path_scenario + "_results", "log_enriched.csv"))
        pd_log = read_ui_log_as_dataframe(os.path.join(path_scenario + "_results", "pd_log.csv"))
        cols_to_drop = pd_log.columns.tolist()
        cols_to_drop.remove(execution.case_study.special_colnames["Screenshot"])
        fe_log = fe_log.drop(columns=cols_to_drop, errors="ignore")
        log = pd.merge(pd_log, fe_log, how='inner', on=execution.case_study.special_colnames["Screenshot"])
        del fe_log
        del pd_log
    else:
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
                compos_nparray = np.array(data["compos"])

                # identifier_-centroidY
                centroid_regex = re.compile(rf".*_(\d*\.?\d+)-(\d*\.?\d+)")
                # Get all the columns that match the regex and do not contain only nan values
                centroid_columns = [col for col in rows.columns if centroid_regex.match(col) and not rows[col].isnull().all()]

                # Pre-compute Polygon objects to avoid creating them in each iteration
                compos_polygons = [(Polygon(compo["points"]), compo) for compo in compos_nparray]

                # Match each centroid with the smallest object containing it using Polygon from shapely
                for col in centroid_columns:
                    centroid = np.array([centroid_regex.match(col).groups()[0], centroid_regex.match(col).groups()[1]])
                    centroid_point = Point(centroid.astype(float))
                    containing_compos = [(compo, poly.area) for poly, compo in compos_polygons if poly.contains(centroid_point)] 
                    if len(containing_compos) == 0:
                        continue
                    compo = min(containing_compos, key=lambda x: x[1])[0]

                    # Insert the class of the smallest object containing the centroid
                    if use_text and compo["class"] == text_classname:
                        log.at[index, col] = compo["text"]
                    else:
                        log.at[index, col] = compo["class"]
    
    # Copy trace_id column because it gets deleted sometimes
    trace = log[execution.case_study.special_colnames["Case"]]
    variant = log[execution.case_study.special_colnames["Variant"]]
    # Remove columns with the same values
    log = log.T.drop_duplicates().T
    log[execution.case_study.special_colnames["Case"]] = trace
    log[execution.case_study.special_colnames["Variant"]] = variant
    log.to_csv(os.path.join(execution_root, "pipeline_log.csv"), index=False)
    return 0,0,0,0

def combine_ui_element_centroid(ui_log_path, path_scenario, execution, pp):
    return combine_ui_element_centroid_aux(ui_log_path, path_scenario, execution, pp, False)

def combine_ui_element_centroid_and_text(ui_log_path, path_scenario, execution, pp):
    return combine_ui_element_centroid_aux(ui_log_path, path_scenario, execution, pp, True)