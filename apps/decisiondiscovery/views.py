import pandas as pd
import json
import os
from art import tprint
from core.settings import sep, decision_foldername, platform_name, flattening_phase_name, decision_model_discovery_phase_name, flattened_dataset_name
from .decision_trees import sklearn_decision_tree, chefboost_decision_tree
from .flattening import flat_dataset_row
from apps.chefboost import Chefboost as chef
from core.utils import read_ui_log_as_dataframe
from django.http import HttpResponseRedirect
from django.views.generic import ListView, DetailView, CreateView
from django.core.exceptions import ValidationError
from .models import DecisionTreeTraining, ExtractTrainingDataset
from .forms import DecisionTreeTrainingForm, ExtractTrainingDatasetForm
# import json
# import sys
# from django.shortcuts import render
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# import graphviz
# import matplotlib.pyplot as plt
# import matplotlib.image as plt_img
# from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
# from sklearn.metrics import accuracy_score

# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)

def extract_training_dataset(decision_point_activity,
        target_label,
        special_colnames,
        columns_to_drop,
        log_path="media/enriched_log_feature_extracted.csv", 
        path_dataset_saved="media/", 
        actions_columns=["Coor_X", "Coor_Y", "MorKeyb", "TextInput", "Click"]):
    """
    Iterate for every UI log row:
        For each case:
            Store in a map all the attributes of the activities until reaching the decision point
            Assuming the decision point is on activity D, the map would have the following structure:
        {
            "headers": ["timestamp", "MOUSE", "clipboard"...],
            "case1": {"CoorX_A": "value1", "CoorY_A": "value2", ..., "Checkbox_A: "value3",
                        "CoorX_B": "value1", "CoorY_B": "value2", ..., "Checkbox_B: "value3"
                        "CoorX_C": "value1", "CoorY_C": "value2", ..., "Checkbox_C: "value3"
                },
            ...
            
            "caseN": {"CoorX_A": "value1", "CoorY_A": "value2", ..., "Checkbox_A: "value3",
                        "CoorX_B": "value1", "CoorY_B": "value2", ..., "Checkbox_B: "value3"
                        "CoorX_C": "value1", "CoorY_C": "value2", ..., "Checkbox_C: "value3"
                },
        }
    Once the map is generated, for each case, we concatinate the header with the activity to name the columns and assign them the values
    For each case a new row in the dataframe is generated
    
    :param decision_point_activity: id of the activity inmediatly previous to the decision point which "why" wants to be discovered
    :type decision_point_activity: str
    :param target_label: name of the column where classification target label is stored
    :type target_label: str
    :param special_colnames: dict that contains the column names that refers to special columns: "Screenshot", "Variant", "Case"...
    :type special_colnames: dict
    :param columns_to_drop: Names of the colums to remove from the dataset
    :type columns_to_drop: list
    :param log_path: path where ui log to be processed is stored
    :type log_path: str
    :param path_dataset_saved: path where files that results from the flattening are stored
    :type path_dataset_saved: str
    :param actions_columns: list that contains column names that wont be added to the event information just before the decision point
    :type actions_columns: list
    """
    tprint("  " + platform_name + " - " + flattening_phase_name, "fancy60")
    print(log_path+"\n")

    log = read_ui_log_as_dataframe(log_path)

    # columns_to_drop = [special_colnames["Case"], special_colnames["Activity"], special_colnames["Timestamp"], special_colnames["Screenshot"], special_colnames["Variant"]]
    columns = list(log.columns)
    for c in columns_to_drop:
        if c in columns:
            columns.remove(c)
        
    # Stablish common columns and the rest of the columns are concatinated with "_" + activity
    flat_dataset_row(log, columns, target_label, path_dataset_saved, special_colnames["Case"], special_colnames["Activity"], 
                     special_colnames["Timestamp"], decision_point_activity, actions_columns)

                     
def decision_tree_training(cv, 
                           flattened_json_log_path="media/flattened_dataset.json",
                           path="media/", 
                           implementation="sklearn",
                           algorithms=['ID3', 'CART', 'CHAID', 'C4.5'],
                           columns_to_ignore=["Timestamp_start", "Timestamp_end"],
                           target_label='Variant',
                           one_hot_columns=['NameApp']):
    
    tprint(platform_name + " - " + decision_model_discovery_phase_name, "fancy60")
    print(flattened_json_log_path+"\n")
    
    flattened_dataset = pd.read_json(flattened_json_log_path, orient ='index')
    # flattened_dataset.to_csv(path + "flattened_dataset.csv")    
    
    path += decision_foldername + sep
    if not os.path.exists(path):
        os.mkdir(path)
    
    # for col in flattened_dataset.columns:
    #     if "Coor" in col:
    #         columns_to_ignore.append(col)  
    
    # TODO: get type of TextInput column using NLP: convert to categorical variable (conversation, name, email, number, date, etc)
    flattened_dataset = flattened_dataset.drop(columns_to_ignore, axis=1)
    flattened_dataset.to_csv(path + flattened_dataset_name)
    columns_len = flattened_dataset.shape[1]
    flattened_dataset = flattened_dataset.fillna('NaN')
    # tree_levels = {}
    
    if implementation == 'sklearn':
        # Default criterion = gini
        if not algorithms:
            algorithms = ["gini"]
        res, times = sklearn_decision_tree(flattened_dataset, path, algorithms, one_hot_columns, target_label, cv)
    elif implementation == 'chefboost':
        res, times = chefboost_decision_tree(flattened_dataset, path, algorithms, target_label, cv)
        # TODO: caculate number of tree levels automatically
        # for alg in algorithms:
            # rules_info = open(path+alg+'-rules.json')
            # rules_info_json = json.load(rules_info)
            # tree_levels[alg] = len(rules_info_json.keys())            
    else:
        raise Exception("Decision model chosen is not an option")
        
    return res, times, columns_len#, tree_levels
    
def decision_tree_predict(module_path, instance):
    """
    moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
    instance = for example ['Sunny', 'Hot', 'High', 'Weak']
    """
    tree = chef.restoreTree(module_path)
    prediction = tree.findDecision(instance)
    return prediction

class ExtractTrainingDatasetCreateView(CreateView):
    model = ExtractTrainingDataset
    form_class = ExtractTrainingDatasetForm
    template_name = "extract_training_dataset/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class ExtractTrainingDatasetListView(ListView):
    model = ExtractTrainingDataset
    template_name = "extract_training_dataset/list.html"
    paginate_by = 50

    def get_queryset(self):
        return ExtractTrainingDataset.objects.filter(user=self.request.user)
    
class DecisionTreeTrainingCreateView(CreateView):
    model = DecisionTreeTraining
    form_class = DecisionTreeTrainingForm
    template_name = "decision_tree_training/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class DecisionTreeTrainingListView(ListView):
    model = DecisionTreeTraining
    template_name = "decision_tree_training/list.html"
    paginate_by = 50

    def get_queryset(self):
        return DecisionTreeTraining.objects.filter(user=self.request.user)
