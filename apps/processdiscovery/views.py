from django.shortcuts import render

# Create your views here.
import numpy as np
import pandas as pd
import os

# State Discovery
import cv2
from tensorflow.keras.applications import VGG16
# from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Process Discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer


def state_level(log_path, root_path, special_colnames, configurations, skip, type):
    """
    Creates an identifier named "state_id" through clustering screenshots and assinging them an identifier 
    """
    if type == "screen-cluster":
        # Cargar el UI log en un DataFrame de Pandas
        ui_log = pd.read_csv(log_path)

        # Cargar el modelo preentrenado de VGG16
        vgg_model = VGG16(weights='imagenet', include_top=False)

        # Función para extraer características de cada screenshot
        def extract_features(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = vgg_model.predict(img)
            return img.flatten()

        # Extraer características de cada screenshot y guardarlas en un nuevo dataframe
        features_df = pd.DataFrame()
        for index, row in ui_log.iterrows():
            img_path = root_path + row[special_colnames['Screenshot']]
            features = extract_features(img_path)
            features_df = features_df.append(pd.Series(features), ignore_index=True)

        # Calcular la matriz de distancia entre las capturas
        distance_matrix = linkage(features_df, method='ward')

        # Dibujar el dendrograma para visualizar la jerarquía de clusters y guardarlo como una imagen
        if ("draw_dendogram" in configurations) and (configurations["draw_dendogram"] == "True"):
            plt.figure(figsize=(10, 5))
            dendrogram(distance_matrix)
            plt.title('Dendrogram')
            plt.xlabel('Screenshots')
            plt.ylabel('Distance')
            plt.savefig(os.path.join(root_path + 'ui_log_states_dendrogram.png'))

        # Obtener los clusters utilizando un umbral de distancia
        threshold = 0.1 * np.max(distance_matrix[:, 2])
        cluster_labels = fcluster(distance_matrix, threshold, criterion='distance')

        # Agregar los clusters al dataframe de UI log
        ui_log['Activity'] = cluster_labels
        # ui_log['trace_id'] = list(range(1, ui_log.shape[0]+1))
        
        # Guardar el resultado en un nuevo archivo CSV
        ui_log.to_csv(root_path + 'ui_log_clustered.csv', index=False)
    else:
        raise Exception("You selected a process discovery type that doesnt exist")

# def process_level(log_path, root_path, special_colnames, configurations, type):
    # # Cargar el UI log clustered como un dataframe de pandas
    # ui_log_clustered = pd.read_csv(root_path + 'ui_log_clustered.csv')

    # # Convertir el UI log clustered a un objeto log de pm4py
    # log = csv_importer.import_dataframe(ui_log_clustered)

    # # Descubrir el modelo BPMN utilizando el algoritmo inductive miner
    # net, initial_marking, final_marking = inductive_miner.apply(log)

    # # Visualizar el modelo BPMN y guardarlo como una imagen
    # bpmn_visualizer.apply(net, initial_marking, final_marking, output_file= root_path + 'bpmn_model.png')


def process_discovery(log_path, root_path, special_colnames, configurations, skip, type):
    if not skip:
        state_level(log_path, root_path, special_colnames, configurations, type)
        # process_level(log_path, root_path, special_colnames, configurations, type)