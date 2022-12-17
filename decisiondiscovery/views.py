from asyncore import write
from typing import List
import pandas as pd
import os
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


# Create your views here.
def flat_dataset_row(data, columns, param_timestamp_column_name, param_variant_column_name, columns_to_drop, param_decision_point_activity, actions_columns):
    """
    With this function we convert the log into a dataset, such that we flatten all the existing registers over the same case,
    resulting on a single row per case. Fot this flattening we only take in account those registers relative to the activities previous
    to the one indicated in para_decision_point_activity, including this last one. The names of the activities colums are concatinated
    with their id, for example, timestamp_A, timestamp_B, etc.

    :param data: Map which keys correspond to the identification of each case and which values are the activies asociated to said case, along with their information
    :type data: map
    :param columns: Names of the dataset colums wanted to be stored for each activity
    :type columns: list
    :param param_timestamp_column_name: nombre de la columna donde se almacena el timestamp
    :type param_timestamp_column_name: str
    :param param_variant_column_name: nombre de la columna donde se almacena la variante, que será la etiqueta de nuestro problema
    :type param_variant_column_name: str
    :param columns_to_drop: Names of the colums to remove from the dataset
    :type columns_to_drop: list
    :param param_decision_point_activity: Identificatior of the activity inmediatly previous to the decision point which "why" wants to be discovered
    :type param_decision_point_activity: str
    :param actions_columns: 
    :type actions_columns: list
    :returns: Dataset
    :rtype: DataFrame
    """
    add_case = False  # Set True to add case column
    df_content = []
    new_list = [col_name for col_name in columns if col_name not in actions_columns]
    for case in data["cases"]:
        # print(case)
        timestamp_start = data["cases"][case]["A"].get(
            key=param_timestamp_column_name)
        timestamp_end = data["cases"][case][param_decision_point_activity].get(
            param_timestamp_column_name)
        variant = data["cases"][case]["A"].get(key=param_variant_column_name)
        if add_case:
            row = [variant, case, timestamp_start,
                   timestamp_end]  # ADDING CASE COLUMN
            headers = ["Variant", "Case", "Timestamp_start", "Timestamp_end"]
        else:
            # WITHOUT CASE COLUMN
            row = [variant, timestamp_start, timestamp_end]
            headers = ["Variant", "Timestamp_start", "Timestamp_end"]
        for act in data["cases"][case]:
            # case
            # variant_id
            if act != param_decision_point_activity:
                # Add sufix with the corresponding activity letter to all columns
                row.extend(data["cases"][case][act].drop(
                    columns_to_drop).values.tolist())
                for c in columns:
                    headers.append(c+"_"+act)
            else:
                row.extend(data["cases"][case][act].drop(
                    columns_to_drop).drop(actions_columns).values.tolist())
                for c in new_list:
                    headers.append(c+"_"+act)
                break
        # Introduce the row with the info of all activities of the case in the dataset
        df_content.append(row)
        # print(row)
        # print(headers)
    df = pd.DataFrame(df_content, columns=headers)
    return df


def extract_training_dataset(
        param_decision_point_activity, param_log_path="media/enriched_log_feature_extracted.csv", param_path_dataset_saved="media/", actions_columns=["Coor_X", "Coor_Y", "MorKeyb", "TextInput", "Click"], param_variant_column_name="Variant", param_case_column_name="Case", param_screenshot_column_name="Screenshot", param_timestamp_column_name="Timestamp", param_activity_column_name="Activity"):
    """
    Iterate for every log row:
        For each case:
            Store in a map all the atributes of the activities until reaching the decision point
            Assuming the decision point is on activity D, the map would have the following structure:
        {
            "headers": ["timestamp", "MOUSE", "clipboard"...],
            "case1":
                {"A": ["value1","value2","value3",...]}
                {"B": ["value1","value2","value3",...]}
                {"C": ["value1","value2","value3",...]},
            "case2":
                {"A": ["value1","value2","value3",...]}
                {"B": ["value1","value2","value3",...]}
                {"C": ["value1","value2","value3",...]},...
        }

    Once the map is generated, for each case, we concatinate the header with the activity to name the columns and assign them the values
    For each case a new row in the dataframe is generated
    """
    tprint(platform_name + " - " + flattening_phase_name, "fancy60")
    print(param_log_path+"\n")

    log = pd.read_csv(param_log_path, sep=",", index_col=0)

    cases = log.loc[:, param_case_column_name].values.tolist()
    actual_case = 0

    log_dict = {"headers": list(log.columns), "cases": {}}
    for index, c in enumerate(cases, start=0):
        if c == actual_case:
            activity = log.at[index, param_activity_column_name]
            if c in log_dict["cases"]:
                log_dict["cases"][c][activity] = log.loc[index, :]
            else:
                log_dict["cases"][c] = {activity: log.loc[index, :]}
        else:
            activity = log.at[index, param_activity_column_name]
            log_dict["cases"][c] = {activity: log.loc[index, :]}
            actual_case = c

    # import p#print
    # pprint.pprint(log_dict)

    # Serializing json
    # json_object = json.dumps(log_dict, indent = 4)

    # Writing to sample.json
    # with open(param_path_dataset_saved + "preprocessed_log.json", "w") as outfile:
    #     outfile.write(json_object)

    columns_to_drop = [param_case_column_name, param_activity_column_name,
                       param_timestamp_column_name, param_screenshot_column_name, param_variant_column_name]
    columns = list(log.columns)
    for c in columns_to_drop:
        columns.remove(c)

    # Stablish common columns and the rest of the columns are concatinated with "_" + activity
    data_flattened = flat_dataset_row(log_dict, columns, param_timestamp_column_name,
                                      param_variant_column_name, columns_to_drop, param_decision_point_activity, actions_columns)
    # print(data_flattened)
    data_flattened.to_csv(param_path_dataset_saved +
                          "preprocessed_dataset.csv")

def decision_tree_training(param_preprocessed_log_path="media/preprocessed_dataset.csv",
                           param_path="media/", 
                           implementation="sklearn",
                           algorithms=['ID3', 'CART', 'CHAID', 'C4.5'],
                           columns_to_ignore=["Timestamp_start", "Timestamp_end"],
                           target_label='Variant',
                           one_hot_columns=['NameApp']):
    
    tprint(platform_name + " - " + decision_model_discovery_phase_name, "fancy60")
    print(param_preprocessed_log_path+"\n")
    
    flattened_dataset = pd.read_csv(param_preprocessed_log_path, index_col=0, sep=',')
    param_path += decision_foldername + sep
    if not os.path.exists(param_path):
        os.mkdir(param_path)
    one_hot_cols = []
    for c in flattened_dataset.columns:
        for item in one_hot_columns:
            if item in c:
                one_hot_cols.append(item)
        
        # if "TextInput" in c:
        #     columns_to_ignore.append(c)  # TODO: get type of field using NLP: convert to categorical variable (conversation, name, email, number, date, etc)
            
    flattened_dataset = pd.get_dummies(flattened_dataset, columns=one_hot_cols)
    flattened_dataset = flattened_dataset.drop(columns_to_ignore, axis=1)
    flattened_dataset = flattened_dataset.fillna(0.)
    # Splitting dataset
    # X_train, X_test = train_test_split(flattened_dataset, test_size=0.2, random_state=42, stratify=flattened_dataset[target_label])
    
    
    if implementation == 'sklearn':
        return CART_sklearn_decision_tree(flattened_dataset, param_path, target_label)
    else:
        return chefboost_decision_tree(flattened_dataset, param_path, algorithms)

def decision_tree_predict(module_path, instance):
    """
    moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
    instance = for example ['Sunny', 'Hot', 'High', 'Weak']
    """
    tree = chef.restoreTree(module_path)
    prediction = tree.findDecision(instance)
    return prediction