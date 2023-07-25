from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

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


# read from text representation
import re

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