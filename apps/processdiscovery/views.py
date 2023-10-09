import os
import numpy as np
import pandas as pd
from django.shortcuts import render
from core.utils import read_ui_log_as_dataframe
# Front Platform Imports
from django.http import HttpResponseRedirect
from django.views.generic import ListView, CreateView
from django.core.exceptions import ValidationError
from .models import ProcessDiscovery
from .forms import ProcessDiscoveryForm
# State Discovery
import cv2
from tensorflow.keras.applications import VGG16
# from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
# Process Discovery
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from apps.analyzer.models import CaseStudy

def scene_level(log_path, root_path, special_colnames, configurations, skip, type):
    """
    Creates an identifier named "scene_id" through clustering screenshots and assinging them an identifier 
    """
    if type == "screen-cluster":
        # Cargar el UI log en un DataFrame de Pandas
        ui_log = read_ui_log_as_dataframe(log_path)


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
        if "similarity_th" in configurations:
            threshold = float(configurations["similarity_th"]) * np.max(distance_matrix[:, 2])
        else:
            threshold = 0.1 * np.max(distance_matrix[:, 2])
            print("Threshold: " + str(threshold))
        cluster_labels = fcluster(distance_matrix, threshold, criterion='distance')

        # Agregar los clusters al dataframe de UI log
        ui_log[special_colnames['Activity']] = cluster_labels
        # ui_log['trace_id'] = list(range(1, ui_log.shape[0]+1))
        
        # Guardar el resultado en un nuevo archivo CSV
        ui_log.to_csv(root_path + 'ui_log_clustered.csv', index=False)
    else:
        raise Exception("You selected a process discovery type that doesnt exist")


def process_level(log_path, root_path, special_colnames, configurations, type):
    dataframe = pm4py.format_dataframe(dataframe, case_id=special_colnames['Case'], activity_key=special_colnames['Activity'], timestamp_key=special_colnames['Timestamp'])
    event_log = pm4py.convert_to_event_log(dataframe)

    process_model = pm4py.discover_bpmn_inductive(event_log)
    # pm4py.view_bpmn(process_model)
    pm4py.save_vis_bpmn(process_model, "bpm.png", "white")
    
    

def process_discovery(log_path, root_path, special_colnames, configurations, skip, type):
    if not skip:
        scene_level(log_path, root_path, special_colnames, configurations, type)
        process_level(log_path, root_path, special_colnames, configurations, type)
        
        
    
class ProcessDiscoveryCreateView(CreateView):
    model = ProcessDiscovery
    form_class = ProcessDiscoveryForm
    template_name = "processdiscovery/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class ProcessDiscoveryListView(ListView):
    model = ProcessDiscovery
    template_name = "processdiscovery/list.html"
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super(ProcessDiscoveryListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = ProcessDiscovery.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset