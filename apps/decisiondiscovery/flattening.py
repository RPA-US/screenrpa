import re
import pandas as pd
import json
import os
import numpy as np
from numpyencoder import NumpyEncoder

from apps.decisiondiscovery.utils import find_prev_act
from apps.processdiscovery.utils import extract_prev_act_labels, Process



def flat_dataset_row(log, columns, path_dataset_saved, special_colnames, 
                          actions_columns, process_discovery):
    """
    This function convert the log into a dataset, that is, to flat all the existing events over the same case,
    resulting on a single row per case. For this flattening only those events relative to the activities previous
    to the one indicated in decision_point_activity are taken in account. The names of the activities colums are concatinated
    with their activity id, for example, timestamp_A, timestamp_B, etc.

    """
    case_column_name = special_colnames["Case"]
    activity_column_name = special_colnames["Activity"]
    timestamp_column_name = special_colnames["Timestamp"]
    variant_colname = special_colnames["Variant"]
    cases = log.loc[:, case_column_name].values.tolist()
    #activities_before_dps = process_discovery.activities_before_dps

    try:
        json_traceability = json.load(open(os.path.join(path_dataset_saved, "traceability.json")))
        process_tracebility = Process.from_json(json_traceability)
    except:
        raise Exception("Tracebility.json not found durring dataset flattening")
    
    decision_points = process_tracebility.get_non_empty_dp_flattened()
    # activities_before_dps= extract_prev_act_labels(os.path.join(path_dataset_saved,"bpmn.dot"))
    activities_before_dps = list(map(lambda dp: (dp.prevAct, dp.id), decision_points))

    if not activities_before_dps or len(activities_before_dps) == 0:
        raise ValueError("The activities_before_dps list is empty. Please, provide a valid list of activities before the decision point or check the process model discovered.")
    
    for i, (act, dp) in enumerate(activities_before_dps):
        last_case = cases[0]
        before_DP = True
        log_dict = {}
        current_post_dps = list(map(lambda dp: dp.id, decision_points[i:]))

        # If the previous activity is the start event, there are no features to base ourselves on
        if act == "start":
            before_DP = False

        for index, c  in enumerate(cases):
            activity = log.at[index, activity_column_name]

            # Set the timestamp for the last event associated to the case
            if c != last_case:
                if cases[index-1] in log_dict.keys():
                    log_dict[cases[index-1]]["Timestamp_end"] = log.at[index-1, timestamp_column_name]
                before_DP = True

            if before_DP:
                if not (c in log_dict):
                    last_case = c
                    log_dict[c] = {
                            "Timestamp_start": log.at[index, timestamp_column_name],
                            variant_colname: log.at[index, variant_colname]
                        }
                for feat in columns:
                    if re.match(r'id[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]+', feat) \
                    and feat not in current_post_dps and type(log.at[index, feat])==str:
                        log_dict[c][feat] = log.at[index, feat]
                    elif feat not in actions_columns \
                    and not re.match(r'id[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]+', feat):
                        log_dict[c][feat+"_"+str(activity)] = log.at[index, feat]

            if str(activity) == act:
                branch = log.at[index, dp]
                before_DP = False
                if str(branch)!="nan":
                    log_dict[c]["dp_branch"] = branch
                else:
                    del log_dict[c]
                    continue
            # Extraer el valor único para cada columna que sigue el patrón especificado y añadirlo al diccionario
            # for col in log.columns:
            #     if "id" in col and re.match(r'id[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]+', col):
            #         if col == dp:
            #             branch = log.at[index, col]
            #             #unique_value = log[col,c].unique()[0]  # Suponiendo que hay un único valor
            #             if str(activity) == act:
            #                 log_dict[c]["dp_branch"] = branch
        if cases[len(cases)-1] in log_dict.keys():
            log_dict[cases[len(cases)-1]]["Timestamp_end"] = log.at[len(cases)-1, timestamp_column_name]
        
        # Serializing json
        json_object = json.dumps(log_dict, cls=NumpyEncoder, indent=4)

        # Writing to one_row_per_case.json
        if os.path.exists(os.path.join(path_dataset_saved, f"flattened_dataset_{act}.csv")):
            i = 1
            while True:
                if not os.path.exists(os.path.join(path_dataset_saved, f"flattened_dataset_{act}-{i}.csv")):
                    aux_path = os.path.join(path_dataset_saved, f"flattened_dataset_{act}-{i}")
                    with open(aux_path+".json", "w") as outfile:
                        outfile.write(json_object)
                    flattened_dataset = pd.read_json(aux_path+".json", orient ='index')
                    flattened_dataset.to_csv(aux_path + ".csv", index=False)
                    break
                i += 1
        else:
            aux_path = os.path.join(path_dataset_saved, f"flattened_dataset_{act}")
            with open(aux_path+".json", "w") as outfile:
                outfile.write(json_object)
            flattened_dataset = pd.read_json(aux_path+".json", orient ='index')
            flattened_dataset.to_csv(aux_path + ".csv", index=False)

    return log_dict