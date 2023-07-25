from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import math

def preprocess_data(data):
  columns_to_drop = list(filter(lambda x:"TextInput" in x, data.columns))
  data = data.drop(columns=columns_to_drop)
  return data

def def_preprocessor(X):
  # define type of columns
  status_columns = list(filter(lambda x:"sta_" in x, X.columns))
  one_hot_columns = list(X.select_dtypes(include=['object']).columns.drop(status_columns))
  numeric_features = X.select_dtypes(include=['number']).columns

  # create each transformer
  status_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
                                ('label_encoder', OrdinalEncoder())
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
        ('status_categorical', status_transformer, status_columns)
    ]
  )
  return preprocessor

def create_and_fit_pipeline(X,y, model):
  preprocessor = def_preprocessor(X)
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
    
    def build_tree(lines, index, depth):
        if index < 0:
            node_depth = 0
            node = ['root']
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

                    if child_depth > node_depth:
                        child, next_index = build_tree(lines, next_index, child_depth)
                        children.append(child)
                    else:
                        break
                node.append(children)
                return node, next_index
            else:
                return node, next_index
        else:
           
            return node, index

    tree_structure, index = build_tree(lines, -1, depth=0)
  
    return tree_structure
  
  
# Check path inside decision tree representation 
def points_distance(punto_x, punto_y):
    x1, y1 = punto_x
    x2, y2 = punto_y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def centroid_distance_checker(punto_x, punto_y, umbral):
    distancia = points_distance(punto_x, punto_y)
    return distancia < umbral
  
def read_feature_column_name(column_name):
    # Definimos la expresión regular para buscar los componentes del identificador
    if "__" in column_name:
        pattern = r"(.*)__([a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+-\d+\.\d+)_(\d+_[a-zA-Z])"
        aux = 1
    else:
        pattern = r"([a-zA-Z]+_[a-zA-Z]+)_(\d+\.\d+-\d+\.\d+)_(\d+_[a-zA-Z])"
        suffix = None
        aux = 0

    # Buscamos las coincidences en el identificador utilizando la expresión regular
    coincidences = re.match(pattern, column_name)

    if coincidences:
        if aux == 1:
            suffix = coincidences.group(1)
        feature = coincidences.group(aux+1)
        centroid = [float(coincidences.group(aux+2).split("-")[0]), float(coincidences.group(aux+2).split("-")[1])]
        activity = coincidences.group(aux+3)
    else:
        raise Exception("El identificador no sigue el formato esperado.")

    return suffix, feature, centroid, activity
  
def find_path_in_decision_tree(tree, feature_values, target_class, centroid_threshold=5.0):
    def dt_condition_checker(parent, node_index, feature_values):
        node = parent[3][node_index]
        if isinstance(node, str):
            return int(node.split(':')[-1]) == target_class

        feature_id, operator, threshold, branches = node
        
        suffix, feature, centroid, activity = read_feature_column_name(feature_id)
        
        exists_schema_aux = True
        for cond_feature in feature_values:
            cond_feature_suffix, cond_feature_name, cond_feature_centroid, cond_feature_activity = read_feature_column_name(cond_feature)
            if cond_feature_name == feature and centroid_distance_checker(centroid, cond_feature_centroid, centroid_threshold):
                feature_value = feature_values[cond_feature]
                exists_schema_aux = False
                break
        if exists_schema_aux:
            return False

        condition = eval(str(feature_value) + ' ' + operator + ' ' + str(threshold))
        if condition:
            next_parent = node
            next_node_index = 0
        else:
            next_parent = parent
            next_node_index = 1

        return dt_condition_checker(next_parent, next_node_index, feature_values)

    return dt_condition_checker(tree, 0, feature_values)