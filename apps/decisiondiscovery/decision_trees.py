from typing import List
import os
import time
import shutil
import graphviz
import pandas as pd
import matplotlib.image as plt_img
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from apps.chefboost import Chefboost as chef
from .utils import preprocess_data, create_and_fit_pipeline, def_preprocessor
from core.settings import plot_decision_trees, several_iterations
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
#         if several_iterations:
#             durations = []
#             for i in range(0, int(several_iterations)):
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
#         # model = chef.fit(df, config = config, target_label = 'Variant')
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


def chefboost_decision_tree(df, param_path, configuration, target_label, cv):
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
        if several_iterations:
            durations = []
            for i in range(0, int(several_iterations)):
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
        accuracies[alg] = cross_validation(X, y, config, target_label, "chefboost", None, cv)
        
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

def sklearn_decision_tree(df, param_path, configuration, one_hot_columns, target_label, cv):
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
        cv = 2

    # Extract features and target variable
    X = df.drop(columns=['Variant'])
    y = df['Variant']
    
    preprocessor = def_preprocessor(X)
    X = preprocessor.fit_transform(X)
    X_df = pd.DataFrame(X)
    feature_names = list(preprocessor.get_feature_names_out())
    X_df.to_csv(param_path + "preprocessed_df.csv", header=feature_names)
    # Define the tree decision tree model
    tree_classifier = DecisionTreeClassifier()
    start_t = time.time()
    tree_classifier, best_params = best_model_grid_search(X, y, tree_classifier, cv)

    accuracies = cross_validation(X_df,pd.DataFrame(y),None,"Variant","sklearn",tree_classifier,cv)
    times["sklearn"] = {"duration": float(time.time()) - float(start_t)}
    # times["sklearn"]["encoders"] = {
    #     "enabled": status_encoder.fit_transform(["enabled"])[0], 
    #     "checked": status_encoder.fit_transform(["checked"])[0],
    #      "__empty__": status_encoder.fit_transform([""])[0]
    # }
    
    # Assuming 'best_tree_tree' is already trained on the training data
    text_representation = export_text(tree_classifier, feature_names=feature_names)
    print("Decision Tree Rules:\n", text_representation)
    
    with open(param_path + "decision_tree.log", "w") as fout:
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
    
    if plot_decision_trees:
        target = list(df[target_label].unique())
        target_casted = [str(t) for t in target]
        img = plot_decision_tree(
            param_path + "decision_tree", tree_classifier, feature_names, target_casted)
        plt.imshow(img)
        plt.show()

    # Grid Search
    accuracies["selected_params"] = best_params

    return accuracies, times

def best_model_grid_search(X_train, y_train, tree_classifier, cv):
    # Define the hyperparameter grid for tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and train the final model
    best_tree_classifier = grid_search.best_estimator_
    print("Grid Search Best Params:\n", grid_search.best_params_)
    
    best_tree_classifier.fit(X_train, y_train)
    
    return best_tree_classifier, grid_search.best_params_

def cross_validation(X, y, config, target_label, library, model, cv):
    # Cross-validation: accurracy + f1 score
    accuracies = {}
    
    skf = StratifiedKFold(n_splits=cv)
    # skf.get_n_splits(X, y)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
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
    
    print(f"  Stratified K-Fold:  accuracy={accuracies['accuracy']} f1_score={accuracies['f1_score']} ")
    return accuracies
