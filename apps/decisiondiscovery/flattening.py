import pandas as pd
import json
import os
from numpyencoder import NumpyEncoder

def flat_dataset_row(log, columns, path_dataset_saved, case_column_name, activity_column_name, timestamp_column_name, 
                          actions_columns, process_discovery):
    """
    This function convert the log into a dataset, that is, to flat all the existing events over the same case,
    resulting on a single row per case. For this flattening only those events relative to the activities previous
    to the one indicated in decision_point_activity are taken in account. The names of the activities colums are concatinated
    with their activity id, for example, timestamp_A, timestamp_B, etc.

    """
    cases = log.loc[:, case_column_name].values.tolist()
    
    activities_before_dps = process_discovery.activities_before_dps
    if not activities_before_dps or len(activities_before_dps) == 0:
        raise ValueError("The activities_before_dps list is empty. Please, provide a valid list of activities before the decision point or check the process model discovered.")
    
    for act in activities_before_dps:
        last_case = cases[0]
        before_DP = True
        log_dict = {}
        
        for index, c  in enumerate(cases, start=0):
            activity = log.at[index, activity_column_name]

            # Set the timestamp for the last event associated to the case
            if c != last_case:
                log_dict[cases[index-1]]["Timestamp_end"] = log.at[index-1, timestamp_column_name]
                before_DP = True

            if before_DP:
                if not (c in log_dict):
                    last_case = c
                    log_dict[c] = {
                            "Timestamp_start": log.at[index, timestamp_column_name],
                            "Variant": log.at[index, "Variant"]
                        }
                if str(act) == str(activity):
                    for h in columns:
                        log_dict[c][h+"_"+str(activity)] = log.at[index, h]
                else:
                    for h in columns:
                        if h not in actions_columns:
                            log_dict[c][h+"_"+str(activity)] = log.at[index, h]
                    before_DP = False
    
        log_dict[cases[len(cases)-1]]["Timestamp_end"] = log.at[len(cases)-1, timestamp_column_name]
        

        # Serializing json
        json_object = json.dumps(log_dict, cls=NumpyEncoder, indent=4)

        # Writing to one_row_per_case.json
        aux_path = os.path.join(path_dataset_saved, f"flattened_dataset_{act}")
        with open(aux_path+".json", "w") as outfile:
            outfile.write(json_object)
        flattened_dataset = pd.read_json(aux_path+".json", orient ='index')
        flattened_dataset.to_csv(aux_path + ".csv")

    return log_dict