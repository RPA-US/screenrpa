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
from .flattening import flat_dataset_row


def extract_training_dataset(param_decision_point_activity, 
                             param_log_path="media/enriched_log_feature_extracted.csv", 
                             param_path_dataset_saved="media/", 
                             actions_columns=["Coor_X", "Coor_Y", "MorKeyb", "TextInput", "Click"], 
                             param_variant_column_name="Variant", 
                             param_case_column_name="Case", 
                             param_screenshot_column_name="Screenshot", 
                             param_timestamp_column_name="Timestamp", 
                             param_activity_column_name="Activity"):
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
    data_flattened.to_csv(param_path_dataset_saved + "preprocessed_dataset.csv")

# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)


# Ref. https://gist.github.com/j-adamczyk/dc82f7b54d49f81cb48ac87329dba95e#file-graphviz_disk_op-py
def plot_decision_tree(path: str,
                       clf: DecisionTreeClassifier,
                       feature_names: List[str],
                       class_names: List[str]) -> np.ndarray:
    # 1st disk operation: write DOT
    export_graphviz(clf, out_file=path+".dot",
                    feature_names=feature_names,
                    class_names=class_names,
                    label="all", filled=True, impurity=False,
                    proportion=True, rounded=True, precision=2)

    # 2nd disk operation: read DOT
    graph = graphviz.Source.from_file(path + ".dot")

    # 3rd disk operation: write image
    graph.render(path, format="png")

    # 4th disk operation: read image
    image = plt_img.imread(path + ".png")

    # 5th and 6th disk operations: delete files
    os.remove(path + ".dot")
    # os.remove("decision_tree.png")

    return image


def CART_sklearn_decision_tree(df, param_path, algorithms, target_label='Variant', implementation):
    times = {}
    # Training set
    X = df.drop(target_label, axis=1)
    # Test set
    y = df[[target_label]]
    
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)

    # criterion="gini", 
    # splitter="best", 
    # random_state=42, 
    # max_depth=3, 
    # min_samples_split=3,
    # min_samples_leaf=5)
    clf_model = DecisionTreeClassifier()
    # clf_model = RandomForestClassifier(n_estimators=100)
    
    times[implementation] = {"start": time.time()}
    clf_model.fit(X, y)  # change train set. X_train, y_train
    times[implementation]["finish"] = time.time()
    
    # Test dataset predictions
    y_predict = clf_model.predict(X)  # X_test

    # accuracy_score(y_test,y_predict))

    target = list(df[target_label].unique())
    feature_names = list(X.columns)

    target_casted = [str(t) for t in target]

    # estimator = clf_model.estimators_[5]

    # export_graphviz(estimator, out_file='tree.dot',
    #                 feature_names = feature_names,
    #                 class_names = target_casted,
    #                 rounded = True, proportion = False,
    #                 precision = 2, filled = True)

    # # Convert to png using system command (requires Graphviz)
    # from subprocess import call
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    text_representation = export_text(clf_model, feature_names=feature_names)
    print("\n\nDecision Tree Text Representation")
    print(text_representation)

    with open(param_path + "decision_tree.log", "w") as fout:
        fout.write(text_representation)

    # type(target_casted[0])

    if not autogeneration == 'autogeneration':
        img = plot_decision_tree(
            param_path + "decision_tree", clf_model, feature_names, target_casted)
        plt.imshow(img)
        plt.show()

    return accuracy_score(y, y_predict), times


def decision_tree_training(param_preprocessed_log_path="media/preprocessed_dataset.csv",
                           param_path="media/", 
                           implementation="sklearn", 
                           autogeneration="autogeneration", 
                           algorithms=['ID3', 'CART', 'CHAID', 'C4.5'], 
                           columns_one_hot_encoding=["NameApp"],
                           target_label='Variant',
                           columns_to_ignore=["Timestamp_start", "Timestamp_end", "TextInput"]):
    tprint(platform_name + " - " + decision_model_discovery_phase_name, "fancy60")
    print(param_preprocessed_log_path+"\n")
    
    
    df = pd.read_csv(param_preprocessed_log_path, index_col=0, sep=',')
    param_path += decision_foldername + sep
    if not os.path.exists(param_path):
        os.mkdir(param_path)
    one_hot_cols = []

    for c in df.columns:
        # TODO: remove hardcoded column names
        if all([item in c for item in columns_one_hot_encoding]):
            one_hot_cols.append(c)
        
        # if "TextInput" in c:
        #     columns_to_ignore.append(c)  # TODO: get type of field using NLP: convert to categorical variable (conversation, name, email, number, date, etc)
            
    df = pd.get_dummies(df, columns=one_hot_cols)
    df = df.drop(columns_to_ignore, axis=1)
    df = df.fillna(0.)
    
    
    if implementation == 'sklearn':
        return CART_sklearn_decision_tree(param_preprocessed_log_path, param_path, autogeneration, columns_to_ignore)
    else:
        return chefboost_decision_tree(param_preprocessed_log_path, param_path, algorithms, columns_to_ignore)

def decision_tree_predict(module_path, instance):
    """
    moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
    instance = for example ['Sunny', 'Hot', 'High', 'Weak']
    """
    tree = chef.restoreTree(module_path)
    prediction = tree.findDecision(instance)
    return prediction