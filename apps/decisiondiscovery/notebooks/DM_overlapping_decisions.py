#%%
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# columna_objetivo = 'variant'
# min_samples_split = 2

# # Carga tu dataset aquí
# df = pd.read_csv('overlap_test_data.csv', sep=';')

# # Elimina las columnas que no son features: contienen 'timestamp' en el nombre
# df = df[[col for col in df.columns if 'timestamp' not in col]]

# # Asegúrate de definir X (features) e y (target)
# X = df.drop(columna_objetivo, axis=1)
# y = df[columna_objetivo]

# # Divide el dataset en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Entrenar el modelo de árbol de decisión
# tree = DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, max_depth=5)  # Usando entropía para simular C4.5
# tree.fit(X_train, y_train)

# # Evaluar el modelo
# y_pred = tree.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # Si la precisión es del 100%, exporta el árbol completo
# tree_text = export_text(tree, feature_names=list(X.columns))
# with open('full_tree.txt', 'w') as f:
#     f.write(tree_text)

# # Manejo de los casos según la precisión
# if not accuracy == 1.0:
#     # Encuentra los errores de clasificación
#     misclassified = y_test != y_pred
#     misclassified_data = X_test[misclassified]
#     misclassified_labels = y_test[misclassified]
#     misclassified_labels = y_pred[misclassified]

#     # Entrena un árbol para cada clase con errores
#     for label in pd.unique(misclassified_labels):
        
#         subset = misclassified_data[misclassified_labels == label]
#         # Copiar el subset en una variable aux_subset
#         aux_subset = subset.copy()
        
#         subset_target = misclassified_labels[misclassified_labels == label]
#         if subset.empty:
#             continue
        
#         # Añadir al subset la columna objetivo
#         aux_subset[columna_objetivo] = y_pred[mi]
        
#         # Guardar el subset completo de las etiquetas mal clasificadas como "subset_for_class_{label}.csv"
#         aux_subset.to_csv(f'subset_for_class_{label}.csv', sep=';', index=False)
        
#         sub_tree = DecisionTreeClassifier(criterion='entropy')
#         sub_tree.fit(subset, subset_target)
#         sub_tree_text = export_text(sub_tree, feature_names=list(X.columns))
        
#         with open(f'tree_for_class_{label}.txt', 'w') as f:
#             f.write(sub_tree_text)


#%%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Carga tu dataset aquí

columna_objetivo = 'variant'
min_samples_split = 2

# Carga tu dataset aquí
df = pd.read_csv('overlap_test_data.csv', sep=';')

# Elimina las columnas que no son features: contienen 'timestamp' en el nombre
df = df[[col for col in df.columns if 'timestamp' not in col]]

# Función para obtener las reglas del árbol de decisión
def get_decision_path(tree, X_sample):
    feature_names = X_sample.columns
    node_indicator = tree.decision_path(X_sample)
    leaf_id = tree.apply(X_sample)
    paths = []
    for sample_id, node_index in enumerate(leaf_id):
        path_nodes = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
        path = []
        for node_id in path_nodes:
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:  # Es una hoja
                path.append(f"class: {tree.classes_[np.argmax(tree.tree_.value[node_id])]}")
                break
            else:
                if X_sample.iloc[sample_id, tree.tree_.feature[node_id]] <= tree.tree_.threshold[node_id]:
                    path.append(f"{feature_names[tree.tree_.feature[node_id]]} <= {tree.tree_.threshold[node_id]:.2f}")
                else:
                    path.append(f"{feature_names[tree.tree_.feature[node_id]]} > {tree.tree_.threshold[node_id]:.2f}")
        paths.append(" -> ".join(path))
    return paths

# Carga tu dataset aquí
# df = pd.read_csv('tu_dataset.csv')
X = df.drop(columna_objetivo, axis=1)
y = df[columna_objetivo]

# Divide el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de árbol de decisión
tree = DecisionTreeClassifier(criterion='entropy')  # Usando entropía para simular C4.5
tree.fit(X_train, y_train)

# Evaluar el modelo
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

if accuracy == 1.0:
    tree_text = export_text(tree, feature_names=list(X.columns))
    with open('full_tree.txt', 'w') as f:
        f.write(tree_text)
else:
    misclassified = y_test != y_pred
    misclassified_data = X_test[misclassified]
    misclassified_labels = y_test[misclassified]
    misclassified_preds = y_pred[misclassified]


    for label in pd.unique(misclassified_labels):
        subset = misclassified_data[misclassified_labels == label]
        subset_target = misclassified_labels[misclassified_labels == label]
        subset_preds = misclassified_preds[misclassified_labels == label]
        
        # Obtener las reglas de decisión para las instancias misclassified
        subset_paths = get_decision_path(tree, subset)
        
        if subset.empty:
            continue
        
        # Guarda el subset con predicciones, etiquetas reales y caminos de decisión
        subset['Predicción'] = subset_preds
        subset['Etiqueta Real'] = subset_target
        subset['Ruta de Decisión'] = subset_paths
        subset.to_csv(f'subset_for_class_{label}.csv', index=False)
        
        # Entrena un nuevo árbol con el subset
        sub_tree = DecisionTreeClassifier(criterion='entropy')
        sub_tree.fit(subset.drop(['Predicción', 'Etiqueta Real', 'Ruta de Decisión'], axis=1), subset_target)
        sub_tree_text = export_text(sub_tree, feature_names=list(subset.columns[:-3]))
        
        with open(f'tree_for_class_{label}.txt', 'w') as f:
            f.write(sub_tree_text)

