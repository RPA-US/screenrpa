import pandas as pd
import json
import os
from numpyencoder import NumpyEncoder

def flat_dataset_row(log, columns, target_label, path_dataset_saved, case_column_name, activity_column_name, timestamp_column_name, 
                          decision_point_activity, actions_columns, process_discovery):
    """
    This function convert the log into a dataset, that is, to flat all the existing events over the same case,
    resulting on a single row per case. For this flattening only those events relative to the activities previous
    to the one indicated in decision_point_activity are taken in account. The names of the activities colums are concatinated
    with their activity id, for example, timestamp_A, timestamp_B, etc.
    :param log_dict: dict which keys correspond to the identification of each case and which values are the activies asociated to said case, along with their information
    :type log_dict: dict
    :param columns: Names of the dataset colums wanted to be stored for each activity
    :type columns: list
    :param target_label: name of the column where classification target label is stored
    :type target_label: str
    :param path_dataset_saved: path where files that results from the flattening are stored
    :type path_dataset_saved: str
    :param case_column_name: name of the column where case is stored
    :type case_column_name: str
    :param activity_column_name: name of the column where activity is stored
    :type activity_column_name: str
    :param timestamp_column_name: name of the column where timestamp is stored
    :type timestamp_column_name: str
    :param decision_point_activity: id of the activity inmediatly previous to the decision point which "why" wants to be discovered
    :type decision_point_activity: str
    :param actions_columns: list that contains column names that wont be added to the event information just before the decision point
    :type actions_columns: list
    :returns: Dataset
    :rtype: DataFrame
    """
    cases = log.loc[:, case_column_name].values.tolist()
    
    activities_before_dps = process_discovery.activities_before_dps
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
                            target_label: log.at[index, target_label]
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