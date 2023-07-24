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
from core.settings import plot_decision_trees, several_iterations
from sklearn.model_selection import StratifiedKFold
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


def chefboost_decision_tree(df, param_path, algorithms, target_label):
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
    
    if "e30" in param_path:
        cv = 2
    else:
        cv = 5

    for alg in list(algorithms):

        df.rename(columns = {target_label:'Decision'}, inplace = True)
        df['Decision'] = df['Decision'].astype(str) # which will by default set the length to the max len it encounters
        enableParallelism = False
        config = {'algorithm': alg, 'enableParallelism': enableParallelism, 'max_depth': 4}# 'num_cores': 2, 
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

        
        # => Feature importance
        fi = chef.feature_importance('outputs/rules/rules.py').set_index("feature")
        fi.to_csv(param_path+alg+"-tree-feature-importance.csv")
        # TODO: Graphical representation of feature importance
        # fi.plot(kind="barh", title="Feature Importance")

        # shutil.move('outputs/rules/rules.py', param_path+alg+'-rules.py')
        if enableParallelism:
            shutil.move('outputs/rules/rules.json', param_path+alg+'-rules.json')


        # Cross-validation: accurracy + f1 score
        X = df.drop(columns=['Decision'])
        y = df['Decision']
        skf = StratifiedKFold(n_splits=cv)
        # skf.get_n_splits(X, y)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")
            X_train_fold, X_test_fold = df.iloc[train_index], df.iloc[test_index] # TODO: review X instead of df
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            # print("y_train_fold")
            # print(y_train_fold)
            # print("y_test_fold")
            # print(y_test_fold)
            current_iteration_model, acc = chef.fit(X_train_fold, config, target_label)
            
            current_iteration_model

            y_pred = []
            metrics_acc = []
            metrics_precision = []
            metrics_recall = []
            metrics_f1 = []
            
            for _, X_test_instance in X_test_fold.iterrows():
                y_pred.append(chef.predict(current_iteration_model, X_test_instance))
                
            metrics_acc.append(accuracy_score(y_test_fold, y_pred))
            metrics_precision.append(precision_score(y_test_fold, y_pred, average='macro'))
            metrics_recall.append(recall_score(y_test_fold, y_pred, average='macro'))
            metrics_f1.append(f1_score(y_test_fold, y_pred, average='macro'))

        accuracies['accuracy'] = np.mean(metrics_acc)
        accuracies['precision'] = np.mean(metrics_precision)
        accuracies['recall'] = np.mean(metrics_recall)
        accuracies['f1_score'] = np.mean(metrics_f1)
        
        print(param_path)
        print(f"  Stratified K-Fold:  accuracy={accuracies['accuracy']} f1_score={accuracies['f1_score']} ")
        
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

def CART_sklearn_decision_tree(df, param_path, one_hot_columns, target_label):
    
    columns_to_encode_one_hot = []
    
    for elem in one_hot_columns:
        for column_name in list(df.columns):
            if elem in column_name:
                columns_to_encode_one_hot.append(column_name)
    if columns_to_encode_one_hot:
        df = pd.get_dummies(df, columns=columns_to_encode_one_hot)

    
    times = {}
    X = df.drop(target_label, axis=1)
    y = df[[target_label]]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)

    # criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)
    clf_model = DecisionTreeClassifier()
    # clf_model = RandomForestClassifier(n_estimators=100)
    start_t = time.time()
    clf_model.fit(X, y)  # change train set. X_train, y_train
    times["sklearn"] = {"duration": float(time.time()) - float(start_t) }

    y_predict = clf_model.predict(X)  # X_test
    # print(accuracy_score(y_test,y_predict))

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

    # # Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename = 'tree.png')

    text_representation = export_text(clf_model, feature_names=feature_names)
    print("\n\nDecision Tree Text Representation")
    print(text_representation)

    with open(param_path + "decision_tree.log", "w") as fout:
        fout.write(text_representation)

    type(target_casted[0])

    if plot_decision_trees:
        img = plot_decision_tree(
            param_path + "decision_tree", clf_model, feature_names, target_casted)
        plt.imshow(img)
        plt.show()

    return accuracy_score(y, y_predict), times