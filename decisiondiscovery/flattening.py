from asyncore import write
from typing import List
import pandas as pd
import os
import json
import time
import shutil
import graphviz
import matplotlib.image as plt_img
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from chefboost import Chefboost as chef
from art import tprint
from rim.settings import sep, decision_foldername, platform_name, flattening_phase_name, decision_model_discovery_phase_name
from .decision_trees import CART_sklearn_decision_tree, chefboost_decision_tree
# import json
# import sys
# from django.shortcuts import render
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)


def flat_dataset_row(log, columns, target_label, path_dataset_saved, case_column_name, activity_column_name, timestamp_column_name, 
                          decision_point_activity, actions_columns):
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
    columns.drop(actions_columns)

    last_case = None
    log_dict = {"headers": columns, "cases": {}}
    for index, c in enumerate(cases, start=0):
            activity = log.at[index, activity_column_name]

            # Set the timestamp for the last event associated to the case
            if c != last_case:
                log_dict["cases"][cases[index-1]]["Timestamp_end"] = log_dict["cases"][cases[index-1]][decision_point_activity].get(timestamp_column_name)

            if not (c in log_dict["cases"]):
                last_case = c
                log_dict["cases"][c] = {
                        "Timestamp_start": log.at[index, timestamp_column_name],
                        target_label: log.at[index, target_label]
                    }
            if activity != decision_point_activity:
                for h in columns:
                    log_dict["cases"][c][h+"_"+activity] = log.at[index, h]
            else:
                for h in columns:
                    if h not in actions_columns:
                        log_dict["cases"][c][h+"_"+activity] = log.at[index, h]
                

    # Serializing json
    json_object = json.dumps(log_dict, indent = 4)

    # Writing to one_row_per_case.json
    with open(path_dataset_saved + "flattened_dataset.json", "w") as outfile:
        outfile.write(json_object)
    
    df = pd.read_json(json.dumps(log_dict["cases"]), orient ='case')    
    df.to_csv(path_dataset_saved + "flattened_dataset.csv")

    return log_dict