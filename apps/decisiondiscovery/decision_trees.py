import os
import time
import shutil
import graphviz
import numpy as np
import pandas as pd
from django.utils.translation import gettext_lazy as _
from typing import List
import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import scipy
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from .utils import def_preprocessor, best_model_grid_search, cross_validation, prev_preprocessor
from apps.chefboost import Chefboost as chef
from core.settings import PLOT_DECISION_TREES, SEVERAL_ITERATIONS
import pickle

# def chefboost_decision_tree(df, param_path, algorithms, target_label):
#     """
    
#     config = {
#     		'algorithm' (string): ID3, 'C4.5, CART, CHAID or Regression
#     		'enableParallelism' (boolean): False
#     		'enableGBM' (boolean): True,
#     		'epochs' (int): 7,
#     		'learning_rate' (int): 1,
#     		'enableRandomForest' (boolean): True,
#     		'num_of_trees' (int): 5,
#     		'enableAdaboost' (boolean): True,
#     		'num_of_weak_classifier' (int): 4
#     	}
#     """
#     times = {}

#     for alg in list(algorithms):
#         df.rename(columns = {target_label:'Decision'}, inplace = True)
#         df['Decision'] = df['Decision'].astype(str) # which will by default set the length to the max len it encounters
#         enableParallelism = False
#         config = {'algorithm': alg, 'enableParallelism': enableParallelism, 'max_depth': 4}# 'num_cores': 2, 
#         if SEVERAL_ITERATIONS:
#             durations = []
#             for i in range(0, int(SEVERAL_ITERATIONS)):
#                 start_t = time.time()
#                 model, accuracy_score = chef.fit(df, config = config)
#                 durations.append(float(time.time()) - float(start_t))
#             durations_total = 0
#             for d in durations:
#                 durations_total+=d
#             times[alg] = { "duration": durations_total/len(durations) }
#         else:
#             start_t = time.time()
#             model, accuracy_score = chef.fit(df, config = config)
#             times[alg] = { "duration": float(time.time()) - float(start_t) }
#         # TODO: accurracy_score -> store evaluate terminar output
#         # accuracy_score = chef.evaluate(model, df, "Decision")
#         # model = chef.fit(df, config = config)
#         # output = subprocess.Popen( [chef.evaluate(model,df)], stdout=subprocess.PIPE ).communicate()[0]
#         # file = open(param_path+alg+'-results.txt','w')
#         # file.write(output)
#         # file.close()
#         # Saving model
#         # model = chef.fit(df, config = config, target_label = special_colnames["Variant"])
#         # chef.save_model(model, alg+'model.pkl')
#         # TODO: feature importance
#         fi = chef.feature_importance('outputs/rules/rules.py').set_index("feature")
#         fi.to_csv(param_path+alg+"-tree-feature-importance.csv")
#         # TODO: Graphical representation of feature importance
#         # fi.plot(kind="barh", title="Feature Importance")
#         shutil.move('outputs/rules/rules.py', param_path+alg+'-rules.py')
#         if enableParallelism:
#             shutil.move('outputs/rules/rules.json', param_path+alg+'-rules.json')
#     return accuracy_score, times


def chefboost_decision_tree(df, param_path, configuration, target_label, k_fold_cross_validation):
    """
    
    config = {
    		'algorithm' (string): ID3, 'C4.5, CART, CHAID or Regression
    		'enableParallelism' (boolean): False
    		'enableGBM' (boolean): True,
    		'epochs' (int): 7,
    		'learning_rate' (int): 1,
    		'enableRandomForest' (boolean): True,
    		'num_of_trees' (int): 5,
    		'enableAdaboost' (boolean): True,
    		'num_of_weak_classifier' (int): 4
    	}
    """
    times = {}
    accuracies = {}
    
    algorithms = configuration["algorithms"]

    for alg in list(algorithms):
        df_aux = df.copy()
        df.rename(columns = {target_label:'Decision'}, inplace = True)
        df['Decision'] = df['Decision'].astype(str) # which will by default set the length to the max len it encounters
        enableParallelism = False
        config = {'algorithm': alg, 'enableParallelism': enableParallelism }# 'num_cores': 2, 
        if SEVERAL_ITERATIONS:
            durations = []
            for i in range(0, int(SEVERAL_ITERATIONS)):
                start_t = time.time()
                model = chef.fit(df, config = config, target_label = "Decision")
                durations.append(float(time.time()) - float(start_t))
            durations_total = 0
            for d in durations:
                durations_total+=d
            times[alg] = { "duration": durations_total/len(durations) }
        else:
            start_t = time.time()
            model, acc = chef.fit(df, config = config, target_label = "Decision")
            times[alg] = { "duration": float(time.time()) - float(start_t) }
        # Saving model
        # chef.save_model(model, alg+'model.pkl')

        X = df_aux.drop(columns=[target_label])
        y = df_aux[target_label]
        accuracies[alg] = cross_validation(X, y, config, target_label, "chefboost", None, k_fold_cross_validation)
        
        # => Feature importance
        fi = chef.feature_importance('outputs/rules/rules.py').set_index("feature")
        fi.to_csv(param_path+alg+"-tree-feature-importance.csv")
        # TODO: Graphical representation of feature importance
        # fi.plot(kind="barh", title="Feature Importance")

        # shutil.move('outputs/rules/rules.py', param_path+alg+'-rules.py')
        if enableParallelism:
            shutil.move('outputs/rules/rules.json', param_path+alg+'-rules.json')


        print(param_path)
        
    return accuracies, times


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

def sklearn_decision_tree(df,prevact, param_path, special_colnames, configuration, one_hot_columns, target_label, k_fold_cross_validation):
    times = {}
    accuracies = {}
    
    # columns_to_encode_one_hot = []
    # for elem in one_hot_columns:
    #     for column_name in list(df.columns):
    #         if elem in column_name:
    #             columns_to_encode_one_hot.append(column_name)
    # if columns_to_encode_one_hot:
    #     df = pd.get_dummies(df, columns=columns_to_encode_one_hot)
    
    if "e50_" in param_path:
        k_fold_cross_validation = 2

    # Extract features and target variable
    X = df.drop(columns=[special_colnames["Variant"]])
    #X = X.astype(str)
    y = df[special_colnames["Variant"]]
    
    X = prev_preprocessor(X)
    if isinstance(X, str):
        raise Exception(X)
        return
    preprocessor = def_preprocessor(X)
    print(X)
    
    #df.head()
    X = preprocessor.fit_transform(X)
    
    # X es una sparse_matrix
    # if a dataframe has a high number of columns, may be it has to be treated as a sparse matrix
    if isinstance(X, scipy.sparse.spmatrix):
        X = X.toarray()
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X, columns=feature_names)

    X_df.to_csv(os.path.join(param_path, "preprocessed_df.csv"), header=feature_names)
    # Define the tree decision tree model
    tree_classifier = DecisionTreeClassifier()
    start_t = time.time()
    tree_classifier, best_params = best_model_grid_search(X, y, tree_classifier, k_fold_cross_validation)

    accuracies = cross_validation(X_df,pd.DataFrame(y),None,special_colnames["Variant"],"sklearn",tree_classifier,k_fold_cross_validation)
    times["sklearn"] = {"duration": float(time.time()) - float(start_t)}
    # times["sklearn"]["encoders"] = {
    #     "enabled": status_encoder.fit_transform(["enabled"])[0], 
    #     "checked": status_encoder.fit_transform(["checked"])[0],
    #      "__empty__": status_encoder.fit_transform([""])[0]
    # }
    
    # Assuming 'best_tree_tree' is already trained on the training data
    text_representation = export_text(tree_classifier, feature_names=list(feature_names))
    print("Decision Tree Rules:\n", text_representation)
    
    with open(os.path.join(param_path, "decision_tree_"+prevact+".log"), "w") as fout:
        fout.write(text_representation)
        
    # estimator = clf_model.estimators_[5]
    # export_graphviz(estimator, out_file='tree.dot',
    #                 feature_names = feature_names,
    #                 class_names = target_casted,
    #                 rounded = True, proportion = False,
    #                 precision = 2, filled = True)

    # # Convert to png using system command (requires Graphviz)
    # from subprocess import call
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    # # Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename = 'tree.png')

    
    saved_data = {
        'classifier': tree_classifier,
        'feature_names': feature_names,
        'class_names': np.unique(y),
    }
    with open(os.path.join(param_path, 'decision_tree_'+prevact+'.pkl'), 'wb') as fid:
        pickle.dump(saved_data, fid)

    if PLOT_DECISION_TREES:
        target = list(df[target_label].unique())
        target_casted = [str(t) for t in target]
        img = plot_decision_tree(
            os.path.join(param_path, 'decision_tree_'+prevact+'.pkl'), tree_classifier, feature_names, target_casted)
        plt.imshow(img)
        plt.show()

    # Grid Search
    accuracies["selected_params"] = best_params

    return accuracies, times
