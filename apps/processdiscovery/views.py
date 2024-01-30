import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# from sklearn.cluster import AgglomerativeClustering
from django.http import HttpResponseRedirect, HttpResponse
from django.views.generic import ListView, CreateView, DetailView
from django.core.exceptions import ValidationError
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from apps.analyzer.models import CaseStudy
from core.utils import read_ui_log_as_dataframe
from .models import ProcessDiscovery
from .forms import ProcessDiscoveryForm
from django.utils.translation import gettext_lazy as _
#Testing dependencies
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter

def scene_level(log_path, scenario_path, root_path, special_colnames, configurations, skip, type):
    """
    Creates an identifier named "scene_id" through clustering screenshots and assinging them an identifier 
    """

    ui_log = read_ui_log_as_dataframe(log_path)

    def extract_features_from_images(df, root_path, image_col):
        vgg_model = VGG16(weights='imagenet', include_top=False)
        img_path = os.path.join(root_path, 'Nano')
        def extract_features(img_path):
            if not os.path.exists(img_path):
                raise ValueError(f"La imagen no existe en {img_path}")

            img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {img_path}")

            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            return vgg_model.predict(img).flatten()
        
        df['features'] = df[image_col].apply(lambda x: extract_features(os.path.join(img_path, x)))
        return df
    
    def cluster_images(df):
        n_clusters = 4
        distance_matrix = linkage(df['features'].tolist(), method='ward')
        # Usar el criterio 'maxclust' para especificar un número fijo de clusters
        cluster_labels = fcluster(distance_matrix, n_clusters, criterion='maxclust')
        df['activity_label'] = cluster_labels
        df['activity_label'] = df['activity_label'].astype(str)
        return df, distance_matrix
    
    
    def auto_process_id_assignment(df):
        cluster_inicial = df['activity_label'].iloc[0]
        process_id = 1
        process_ids = []
        for index, row in df.iterrows():
            if row['activity_label'] == cluster_inicial and index !=0:
                process_id += 1
            process_ids.append(process_id)
        df['process_id'] = process_ids
        return df
    
    ui_log = extract_features_from_images(ui_log, root_path, 'ocel:screenshot:name')
    ui_log, distance_matrix = cluster_images(ui_log)
    ui_log = auto_process_id_assignment(ui_log)
    folder_path = os.path.join(root_path, 'results')
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        ui_log.to_csv(os.path.join(folder_path + 'ui_log_process_discovery.csv'), index=False)
    else:
        raise Exception(_("You selected a process discovery type that does not exist"))
    print(folder_path)
    return folder_path, ui_log


def process_level(folder_path, df):
    def petri_net_process(df):
        formatted_df = pm4py.format_dataframe(df, case_id='process_id', activity_key='activity_label', timestamp_key='ocel:timestamp')
        event_log = pm4py.convert_to_event_log(formatted_df)
        process_tree = inductive_miner.apply(event_log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
        dot = pn_visualizer.apply(net, initial_marking, final_marking)
        dot_path = os.path.join(folder_path, 'pn.dot')
        with open(dot_path, 'w') as f:
            f.write(dot.source)

    def bpmn_process(df):
        formatted_df = pm4py.format_dataframe(df, case_id='process_id', activity_key='activity_label', timestamp_key='ocel:timestamp')
        event_log = pm4py.convert_to_event_log(formatted_df)
        bpmn_model = pm4py.discover_bpmn_inductive(event_log)
        dot = bpmn_visualizer.apply(bpmn_model)
        dot_path = os.path.join(folder_path, 'bpmn.dot')
        with open(dot_path, 'w') as f:
            f.write(dot.source)
        bpmn_exporter.apply(bpmn_model, os.path.join(folder_path, 'bpmn.bpmn'))

    petri_net_process(df)
    bpmn_process(df)
    
    

def process_discovery(log_path, scenario_path, root_path, special_colnames, configurations, skip, type):
    if not skip:
        # log_path -> media/unzipped/exec_1/Nano/log.csv
        # scenario_path -> media/unzipped/exec_1/Nano/
        # root_path -> media/unzipped/exec_1/

        # root_path + "results/" + "ui_log_process_discovery.csv"
        folder_path, ui_log = scene_level(log_path, scenario_path, root_path, special_colnames, configurations, skip,  type)
        process_level(folder_path, ui_log)
        
    
class ProcessDiscoveryCreateView(CreateView):
    model = ProcessDiscovery
    form_class = ProcessDiscoveryForm
    template_name = "processdiscovery/create.html"
    
    def get_context_data(self, **kwargs):
        context = super(ProcessDiscoveryCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context    

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
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
    

class ProcessDiscoveryDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        process_discovery = get_object_or_404(ProcessDiscovery, id=kwargs["process_discovery_id"])
        return render(request, "processdiscovery/detail.html", {"process_discovery": process_discovery, "case_study_id": kwargs["case_study_id"]})

def set_as_process_discovery_active(request):
    process_discovery_id = request.GET.get("processdiscovery_id")
    case_study_id = request.GET.get("case_study_id")
    process_discovery_list = ProcessDiscovery.objects.filter(case_study_id=case_study_id)
    for m in process_discovery_list:
        m.active = False
        m.save()
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    process_discovery.active = True
    process_discovery.save()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))
    
def delete_process_discovery(request):
    process_discovery_id = request.GET.get("processdiscovery_id")
    case_study_id = request.GET.get("case_study_id")
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    if request.user.id != process_discovery.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    process_discovery.delete()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))
