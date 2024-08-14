import pickle
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
from sklearn.tree import _tree
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
    # Identificar las columnas con todos los valores nulos
    columns_to_drop = X.columns[X.isnull().all()].tolist()
    # Eliminar las columnas con todos los valores iguales o nulos
    X_drop = X.drop(columns=columns_to_drop)
    

    if len(X_drop.columns) == 0:
        return "No features left after preprocessing."

    return X_drop



def def_preprocessor(X):
    # Define el diccionario de mapeo para las columnas "enabled" y "checked"
    mapping_dict = {"enabled": ['NaN', 'enabled', 'disabled'], "checked": ['unchecked', 'checked', '']}
    
    mapping_list = []
    sta_columns = []
    
    # Identificar las columnas que contienen "rpa-us_" en su nombre
    for col in X.columns:
        if 'rpa-us_' in col:
            # sta_columns.append(col)
            if 'enabled' in col:
                mapping_list.append(mapping_dict['enabled'])
            elif 'checked' in col:
                mapping_list.append(mapping_dict['checked'])
            else:
                # Si la columna no coincide con ninguna categoría conocida, agregar un mapeo genérico
                unique_values = X[col].dropna().unique().tolist()
                if 'NaN' not in unique_values:
                    unique_values.append('NaN')
                mapping_list.append(unique_values)
                
    # Identificar columnas de tipo objeto y numéricas
    types_obj = X.select_dtypes(include=['object']).columns
    one_hot_columns = list(types_obj.drop(sta_columns, errors='ignore'))
    numeric_features = X.select_dtypes(include=['number']).columns

    # Crear cada transformador
    # status_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
    #     ('label_encoder', OrdinalEncoder(categories=mapping_list))
    # ])
    
    one_hot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('one_hot_encoder', OneHotEncoder())
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        # ('scaler',StandardScaler()) # Descomentar si se requiere escalado
    ])

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('one_hot_categorical', one_hot_transformer, one_hot_columns)
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

# def rename_nan_columns(df):
#     for col in df.columns:
#         if df[col].nunique() == 2 and 'nan' in col.lower():
#             # Obtener los nombres únicos en la columna, excluyendo 'NaN'
#             unique_values = df[col].unique().tolist()
#             non_nan_value = [val for val in unique_values if 'nan' not in str(val).lower()][0]
#             # Renombrar la columna
#             df.rename(columns={col: non_nan_value}, inplace=True)
#     return df

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
  
# def read_feature_column_name(column_name):
    
#     # Buscar el patrón en la cadena de texto
#     contains_centroid = bool(re.search(r'\d+\.\d+-\d+\.\d+', column_name))
    
#     # Definimos la expresión regular para buscar los componentes del identificador
#     if "__" in column_name and contains_centroid:
#         pattern = r"(.*)__([a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+-\d+\.\d+)_(\d*_?[a-zA-Z])"
#         aux1 = 1
#         aux2 = 1
#     elif "__" in column_name and not contains_centroid:
#         pattern = r"(\w+)__(\w+)_(\w+_\w+)"
#         centroid = None
#         aux1 = 1
#         aux2 = 0
#     elif not "__" in column_name and not contains_centroid:
#         pattern = r"(\w+)_(\w+_\w+)"
#         suffix = None
#         aux1 = 0
#         centroid = None
#         aux2 = 0
#     elif not "__" in column_name and not contains_centroid: #one_hot_categorical__case:concept:name_1_1
#         pattern = r"([a-zA-Z_]+)__([a-zA-Z:+]+)_(\d+_\d+)"
#         suffix = None
#         aux1 = 0
#         aux2 = 1
#     elif "__" in column_name and contains_centroid: #status_categorical__rpa-us_282.6106567382812-167.1426658630371_1
#         pattern = r"([a-zA-Z_]+)__([a-zA-Z-]+)_(\d+\.\d+-\d+\.\d+)_(\d+)"
#         suffix = None
#         aux1 = 0
#         aux2 = 1
#     else:
#         pattern = r"([a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+-\d+\.\d+)_(\d*_?[a-zA-Z])"
#         suffix = None
#         aux1 = 0
#         aux2 = 1

#     # Buscamos las coincidences en el identificador utilizando la expresión regular
#     coincidences = re.match(pattern, column_name)

#     if coincidences:
#         if aux1 == 1:
#             suffix = coincidences.group(1)
#         feature = coincidences.group(aux1+1)
#         if aux2 == 1:
#             centroid = [float(coincidences.group(aux1+2).split("-")[0]), float(coincidences.group(aux1+2).split("-")[1])]
#         activity = coincidences.group(aux1+aux2+2)
#     else:
#         raise Exception(_("The identifier does not follow the pattern"))

#     return suffix, feature, centroid, activity


# def read_feature_column_name(column_name):
#     contains_centroid = bool(re.search(r'\d+\.\d+-\d+\.\d+', column_name))

#     if contains_centroid:
#         pattern = r"([a-zA-Z_]+)__([a-zA-Z-]+)_(\d+\.\d+-\d+\.\d+)_(\d+)(_?[a-zA-Z]?)"
#     else:
#         #pattern = r"([a-zA-Z_]+)__([a-zA-Z_]+)_(\d+)"
#         pattern = r"([a-zA-Z_]+)__([a-zA-Z0-9_]+)_(\d+)(_?[a-zA-Z]?)"

#     # Intentamos encontrar coincidencias con el patrón definido
#     coincidences = re.match(pattern, column_name)

#     # Verificamos si hay coincidencias antes de intentar acceder a los grupos
#     if not coincidences:
#         raise Exception(f"The identifier '{column_name}' does not follow the pattern")

#     suffix = coincidences.group(1)
#     feature = coincidences.group(2)
#     if contains_centroid:
#         centroid = [float(coord) for coord in coincidences.group(3).split("-")]
#         activity = coincidences.group(4)
#     else:
#         centroid = None
#         activity = coincidences.group(3)

#     return suffix, feature, centroid, activity
#     # Si no coincide con ninguno de los patrones

def read_feature_column_name(column_name):
    # Patrón para los nombres de columna que contienen centroid
    pattern_with_centroid = r"([a-zA-Z_]+)__([a-zA-Z0-9_-]+)_(\d+\.\d+-\d+\.\d+)_(\d+)(_?[a-zA-Z]?)"
    # Patrón para los nombres de columna que no contienen centroid
    pattern_without_centroid = r"([a-zA-Z_]+)__([a-zA-Z0-9_]+)_(\d+)(_?[a-zA-Z]?)"
    # Patrón adicional para nombres de columna sin prefijo
    pattern_no_prefix = r"([a-zA-Z0-9_]+)_(\d+\.\d+-\d+\.\d+)_(\d+)(_?[a-zA-Z]?)"
    # Patroón para puntos de decisión
    # numeric__id6322e007-a58b-4b5a-b711-8f51d37c438f_1
    pattern_decision_point = r"([a-zA-Z_]+)__([a-zA-Z0-9-]+)_(\d+)(_?[a-zA-Z]?)"
    
    # Intentamos encontrar coincidencias con los patrones definidos
    coincidences = re.match(pattern_with_centroid, column_name)
    if coincidences:
        suffix = coincidences.group(1)
        feature = coincidences.group(2)
        centroid = [float(coord) for coord in coincidences.group(3).split("-")]
        activity = coincidences.group(4)
        if coincidences.group(5):  # Si hay un grupo 5 adicional (opcional)
            activity += coincidences.group(5)
        return suffix, feature, centroid, activity

    coincidences = re.match(pattern_without_centroid, column_name)
    if coincidences:
        suffix = coincidences.group(1)
        feature = coincidences.group(2)
        centroid = None
        activity = coincidences.group(3)
        if coincidences.group(4):  # Si hay un grupo 4 adicional (opcional)
            activity += coincidences.group(4)
        return suffix, feature, centroid, activity
    
    coincidences = re.match(pattern_no_prefix, column_name)
    if coincidences:
        suffix = None
        feature = coincidences.group(1)
        centroid = [float(coord) for coord in coincidences.group(2).split("-")]
        activity = coincidences.group(3)
        if coincidences.group(4):  # Si hay un grupo 4 adicional (opcional)
            activity += coincidences.group(4)
        return suffix, feature, centroid, activity

    coincidences = re.match(pattern_decision_point, column_name)
    if coincidences:
        suffix = coincidences.group(1)
        feature = coincidences.group(2)
        centroid = None
        activity = coincidences.group(3)
        if coincidences.group(4):  # Si hay un grupo 4 adicional (opcional)
            activity += coincidences.group(4)
        return suffix, feature, centroid, activity
    
    # Si no coincide con ninguno de los patrones
    raise Exception(f"The identifier '{column_name}' does not follow the pattern")

  
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

def extract_tree_rules(path_to_tree_file):
    try:
        with open('/screenrpa/' + path_to_tree_file, 'rb') as archivo:
            loaded_data = pickle.load(archivo)
        # Obtener el clasificador y los nombres de las características del diccionario cargado
        tree = loaded_data['classifier']
        feature_names = loaded_data['feature_names']
        classes = loaded_data['class_names']

    except FileNotFoundError:
        print(f"File not found: {path_to_tree_file}")
        return None

    """
    Función que recorre las ramas de un árbol de decisión y extrae las reglas
    obtenidas para clasificar cada una de las variables objetivo
    
    Parametros:
    - tree: El modelo árbol de decisión.
    - feature_names: Lista de los atributos del dataset.
    - classes: Clases posibles de la variable objetivo, ordenada ascendentemente
    """
    # Accede al objeto interno tree_ del árbol de decisión
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # Crear un diccionario para almacenar las reglas de cada clase
    rules_per_class = {cls: [] for cls in classes}

    def recurse(node, parent_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = parent_rule + [f"{name} <= {threshold:.2f}"]
            right_rule = parent_rule + [f"{name} > {threshold:.2f}"]

            recurse(tree_.children_left[node], left_rule)
            recurse(tree_.children_right[node], right_rule)
        else:
            rule = " & ".join(parent_rule)
            target_index = tree_.value[node].argmax()
            target = classes[target_index] if target_index < len(classes) else None
            if target:
                # Agregar la regla a la lista correspondiente de su clase en el diccionario
                rules_per_class[target].append(rule)

    recurse(0, [])

    # Filtrar las clases que no tienen reglas
    rules_per_class = {k: v for k, v in rules_per_class.items() if v}

    return rules_per_class

def truncar_a_dos_decimales(valor):
    cadena = str(valor)
    partes = cadena.split('.')
    if len(partes) == 2:
        return partes[0] + '.' + partes[1][:2]
    else:
        return cadena

def rename_columns_with_centroids(df):
    pattern_with_centroid = r"([a-zA-Z_]+)__([a-zA-Z0-9_-]+)_(\d+\.\d+-\d+\.\d+)_(\d+)(_?[a-zA-Z0-9]*)"
    pattern_no_prefix = r"([a-zA-Z0-9_]+)_(\d+\.\d+-\d+\.\d+)_(\d+)(_?[a-zA-Z0-9]*)"
    
    new_columns = []
    
    for column in df.columns:
        matches_with_centroid = re.match(pattern_with_centroid, column)
        matches_no_prefix = re.match(pattern_no_prefix, column)
        
        if matches_with_centroid:
            suffix = matches_with_centroid.group(1)
            feature = matches_with_centroid.group(2)
            centroid = [float(coord) for coord in matches_with_centroid.group(3).split("-")]
            activity = matches_with_centroid.group(4)
            extra = matches_with_centroid.group(5) if matches_with_centroid.group(5) else ""
            centroid_str = f"{truncar_a_dos_decimales(centroid[0])}-{truncar_a_dos_decimales(centroid[1])}"
            new_name = f"{suffix}__{feature}_{centroid_str}_{activity}{extra}"
            new_columns.append(new_name)
        elif matches_no_prefix:
            feature = matches_no_prefix.group(1)
            centroid = [float(coord) for coord in matches_no_prefix.group(2).split("-")]
            activity = matches_no_prefix.group(3)
            extra = matches_no_prefix.group(4) if matches_no_prefix.group(4) else ""
            centroid_str = f"{truncar_a_dos_decimales(centroid[0])}-{truncar_a_dos_decimales(centroid[1])}"
            new_name = f"{feature}_{centroid_str}_{activity}{extra}"
            new_columns.append(new_name)
        else:
            new_columns.append(column)
    
    df.columns = new_columns
    return df