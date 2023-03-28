from django.shortcuts import render

# Create your views here.
from PIL import Image
import numpy as np
import pandas as pd
import cv2
# from sklearn.cluster import AgglomerativeClustering
import os
from tensorflow.keras.applications import VGG16
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


def process_discovery(log_path, root_path, special_colnames, configurations, skip, type):
    """
    Creates an identifier named "state_id" through clustering screenshots and assinging them an identifier 
    """
    if not skip:
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

            # Dibujar el dendrograma para visualizar la jerarquía de clusters
            # plt.figure(figsize=(10, 5))
            # dendrogram(distance_matrix)
            # plt.title('Dendrograma')
            # plt.xlabel('Capturas')
            # plt.ylabel('Distancia')
            # plt.show()

            # Obtener los clusters utilizando un umbral de distancia
            threshold = 0.1 * np.max(distance_matrix[:, 2])
            cluster_labels = fcluster(distance_matrix, threshold, criterion='distance')

            # Agregar los clusters al dataframe de UI log
            ui_log['state_id'] = cluster_labels

            # Guardar el resultado en un nuevo archivo CSV
            ui_log.to_csv(root_path + 'ui_log_clustered.csv', index=False)
        else:
            raise Exception("You selected a process discovery type that doesnt exist")