import re
import math
import numpy as np
from django.shortcuts import get_object_or_404
from django.utils.translation import gettext_lazy as _
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder #LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from apps.chefboost import Chefboost as chef
from .models import ExtractTrainingDataset, DecisionTreeTraining
import json
###########################################################################################################################
# case study get phases data  ###########################################################################################
###########################################################################################################################

def get_extract_training_dataset(case_study):
  return get_object_or_404(ExtractTrainingDataset, case_study=case_study, active=True)

def get_decision_tree_training(case_study):
  return get_object_or_404(DecisionTreeTraining, case_study=case_study, active=True)

def case_study_has_extract_training_dataset(case_study):
  return ExtractTrainingDataset.objects.filter(case_study=case_study, active=True).exists()

def case_study_has_decision_tree_training(case_study):
  return DecisionTreeTraining.objects.filter(case_study=case_study, active=True).exists()


###########################################################################################################################

def best_model_grid_search(X_train, y_train, tree_classifier, k_fold_cross_validation):
    # Define the hyperparameter grid for tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, cv=k_fold_cross_validation)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and train the final model
    best_tree_classifier = grid_search.best_estimator_
    print("Grid Search Best Params:\n", grid_search.best_params_)
    
    best_tree_classifier.fit(X_train, y_train)
    
    return best_tree_classifier, grid_search.best_params_

def cross_validation(X, y, config, target_label, library, model, k_fold_cross_validation):
    # Cross-validation: accurracy + f1 score
    accuracies = {}
    
    skf = StratifiedKFold(n_splits=k_fold_cross_validation)
    # skf.get_n_splits(X, y)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print("Fold {}:".format(i))
        print("Train: index={}".format(train_index))
        print("Test:  index={}".format(test_index))
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        if library == "chefboost":
            current_iteration_model, acc = chef.fit(X_train_fold+X_test_fold, config, target_label)
        elif library == "sklearn":
            current_iteration_model = model.fit(X_train_fold, y_train_fold)
        else:
            raise Exception("Decision Model Option Not Valid")

        metrics_acc = []
        metrics_precision = []
        metrics_recall = []
        metrics_f1 = []
        
        if library == "chefboost":
            y_pred = []
            for _, X_test_instance in X_test_fold.iterrows():
                y_pred.append(chef.predict(current_iteration_model, X_test_instance))
        elif library == "sklearn":
            y_pred = current_iteration_model.predict(X_test_fold)
        else:
            raise Exception("Decision Model Option Not Valid")
            
        metrics_acc.append(accuracy_score(y_test_fold, y_pred))
        metrics_precision.append(precision_score(y_test_fold, y_pred, average='weighted'))
        metrics_recall.append(recall_score(y_test_fold, y_pred, average='weighted'))
        metrics_f1.append(f1_score(y_test_fold, y_pred, average='weighted'))

    accuracies['accuracy'] = np.mean(metrics_acc)
    accuracies['precision'] = np.mean(metrics_precision)
    accuracies['recall'] = np.mean(metrics_recall)
    accuracies['f1_score'] = np.mean(metrics_f1)
    print("Stratified K-Fold:  accuracy={} f1_score={}".format(accuracies['accuracy'], accuracies['f1_score']))
    return accuracies



def preprocess_data(data):
  columns_to_drop = list(filter(lambda x:"TextInput" in x, data.columns))
  data = data.drop(columns=columns_to_drop)
  return data

def prev_preprocessor(X):
    # define type of columns
    # sta_columns = list(filter(lambda x:"sta_" in x, X.columns))

    X = X.loc[:, ~X.columns.str.contains('^Unnamed')] # Remove unnamed columns automatically generated
    # Identificar las columnas con todos los valores iguales
    columns_to_drop = X.columns[X.nunique() == 1]
    # Identificar las columnas con todos los valores nulos
    columns_to_drop = columns_to_drop.union(X.columns[X.isnull().all()])
    # Eliminar las columnas con todos los valores iguales o nulos
    X = X.drop(columns=columns_to_drop)

    if len(X.columns) == 0:
        return "No features left after preprocessing."

    return X

def def_preprocessor(X):
    
    
    mapping_dict = {"enabled": ['NaN', 'enabled', 'disabled'], "checked": ['unchecked', 'checked', '']}
    mapping_list = []
    sta_columns = []
    # Identificar las columnas que contienen "sta_" en su nombre
    for col in X.columns:
        if 'sta_' in col:
            sta_columns.append(col)
            if 'enabled' in col:
                mapping_list.append(list(mapping_dict['enabled']))
            elif 'checked' in col:
                mapping_list.append(list(mapping_dict['checked']))
            else:
                raise Exception("Not preprocessed column: " + str(col))
                
    one_hot_columns = list(X.select_dtypes(include=['object']).columns.drop(sta_columns))
    numeric_features = X.select_dtypes(include=['number']).columns

    # create each transformer
    status_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
                                    ('label_encoder', OrdinalEncoder(categories=list(mapping_list)))
                                    ])
    one_hot_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
                                    ('one_hot_encoder', OneHotEncoder())
                                    ])

    numeric_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='mean')),
                                    ])#('scaler',StandardScaler())

    # create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('one_hot_categorical', one_hot_transformer, one_hot_columns),
            ('status_categorical', status_transformer, sta_columns)
        ]
    )
    return preprocessor

def create_and_fit_pipeline(X,y, model):
  preprocessor = def_preprocessor(X, [])
  # create pipeline
  pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',model)
  ])

  # fit pipeline
  pipeline.fit(X,y)

  return pipeline


# Formating textual representation of decision trees
def parse_decision_tree(file_path):
    tree_structure = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    def parse_node(node_str, depth):
        match = re.match(r'\|   ' * (get_node_depth(node_str) - 1) + r'\|--- (.+) (<=|>=|>|<)\s+([0-9.-]+)', node_str)
        if match:
            feature, operator, threshold = match.groups()
            return [feature, operator, float(threshold)]

        match = re.match(r'\|   ' * (get_node_depth(node_str) - 1) + r'\|--- class: (.+)', node_str)
        if match:
            class_value = match.group(1)
            return f'class: {class_value}'

    def get_node_depth(node_str):
        return len(re.findall(r'\|', node_str))
    
    def build_tree(lines, index, depth, max_depth):
        if index < 0:
            node_depth = 0
            node = ['root', 'None', 'None']
        else:
            node_str = lines[index].strip()
            node_depth = get_node_depth(node_str)
            node = parse_node(node_str, node_depth)

        next_index = index + 1
        if node_depth == depth:

            if isinstance(node, list):
                children = []
                while next_index < len(lines):
                    child_depth = get_node_depth(lines[next_index].strip())
                    max_depth = child_depth if child_depth > max_depth else max_depth
                    
                    if child_depth > node_depth:
                        child, max_depth, next_index = build_tree(lines, next_index, child_depth, max_depth)
                        children.append(child)
                    else:
                        break
                node.append(children)
                return node, max_depth, next_index
            else:
                return node, max_depth, next_index
        else:
            return node, max_depth, index

    tree_structure, max_depth, index = build_tree(lines, -1, depth=0, max_depth=0)
  
    return tree_structure, max_depth
  
  
# Check path inside decision tree representation 
def points_distance(punto_x, punto_y):
    x1, y1 = punto_x
    x2, y2 = punto_y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def centroid_distance_checker(punto_x, punto_y, umbral):
    if not punto_x and not punto_y:
        return True
    else:
        distancia = points_distance(punto_x, punto_y)
        return distancia < umbral
  
def read_feature_column_name(column_name):
    
    # Buscar el patrón en la cadena de texto
    contains_centroid = bool(re.search(r'\d+\.\d+-\d+\.\d+', column_name))
    
    # Definimos la expresión regular para buscar los componentes del identificador
    if "__" in column_name and contains_centroid:
        pattern = r"(.*)__([a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+-\d+\.\d+)_(\d*_?[a-zA-Z])"
        aux1 = 1
        aux2 = 1
    elif "__" in column_name and not contains_centroid:
        pattern = r"(\w+)__(\w+)_(\w+_\w+)"
        centroid = None
        aux1 = 1
        aux2 = 0
    elif not "__" in column_name and not contains_centroid:
        pattern = r"(\w+)_(\w+_\w+)"
        suffix = None
        aux1 = 0
        centroid = None
        aux2 = 0
    elif not "__" in column_name and not contains_centroid: #one_hot_categorical__case:concept:name_1_1
        pattern = r"([a-zA-Z_]+)__([a-zA-Z:+]+)_(\d+_\d+)"
        suffix = None
        aux1 = 0
        aux2 = 1
    else:
        pattern = r"([a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+-\d+\.\d+)_(\d*_?[a-zA-Z])"
        suffix = None
        aux1 = 0
        aux2 = 1

    # Buscamos las coincidences en el identificador utilizando la expresión regular
    coincidences = re.match(pattern, column_name)

    if coincidences:
        if aux1 == 1:
            suffix = coincidences.group(1)
        feature = coincidences.group(aux1+1)
        if aux2 == 1:
            centroid = [float(coincidences.group(aux1+2).split("-")[0]), float(coincidences.group(aux1+2).split("-")[1])]
        activity = coincidences.group(aux1+aux2+2)
    else:
        raise Exception(_("The identifier does not follow the pattern"))

    return suffix, feature, centroid, activity
  
def find_path_in_decision_tree(tree, feature_values, target_class, centroid_threshold=250):
    def dt_condition_checker(parent, node_index, features_in_tree):
        feature_values = features_in_tree["feature_values"]
        node = parent[3][node_index]
        if isinstance(node, str):
            res = node.split(':')[-1].strip() == target_class, features_in_tree
            return res 
        
        feature_id, operator, threshold, branches = node
        
        suffix, feature, centroid, activity = read_feature_column_name(feature_id)
        
        exists_schema_aux = False
        for cond_feature in feature_values:
            if cond_feature[:7] == "or_cond":
                for or_cond_feature in feature_values[cond_feature]:
                    or_cond_feature_suffix, or_cond_feature_name, or_cond_feature_centroid, or_cond_feature_activity = read_feature_column_name(or_cond_feature)
                    if (feature == or_cond_feature_name and centroid_distance_checker(centroid, or_cond_feature_centroid, centroid_threshold)):
                        feature_value = feature_values[cond_feature][or_cond_feature]
                        features_in_tree[or_cond_feature] = features_in_tree[or_cond_feature] + 1 if or_cond_feature in features_in_tree else 1
                        exists_schema_aux = True
                        break  
            else:
                cond_feature_suffix, cond_feature_name, cond_feature_centroid, cond_feature_activity = read_feature_column_name(cond_feature)
                if cond_feature_name == feature and centroid_distance_checker(centroid, cond_feature_centroid, centroid_threshold):
                    feature_value = feature_values[cond_feature]
                    features_in_tree[cond_feature] = features_in_tree[cond_feature] + 1 if cond_feature in features_in_tree else 1
                    exists_schema_aux = True
                    break
        if not exists_schema_aux:
            return False, features_in_tree

        condition = eval(str(feature_value) + ' ' + operator + ' ' + str(threshold))

        if condition:
            next_parent = node
            next_node_index = 0
        else:
            next_parent = parent
            next_node_index = 1

        return dt_condition_checker(next_parent, next_node_index, features_in_tree)

    return dt_condition_checker(tree, 0, {"feature_values": feature_values})


########################################3


def find_prev_act(json_path, decision_point_id):
    
    try:
        with open(json_path, 'r') as file:
            traceability = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

    for dp in traceability.get("decision_points", []):
        if dp["id"] == decision_point_id:
            return dp["prevAct"]
    
    return None