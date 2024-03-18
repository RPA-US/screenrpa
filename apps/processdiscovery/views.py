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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


def scene_level(log_path, scenario_path, root_path, special_colnames, configurations, skip, type):
    """
    Creates an identifier named "scene_id" through clustering screenshots and assinging them an identifier 
    """

    ui_log = read_ui_log_as_dataframe(log_path)

    def extract_features_from_images(df, root_path, image_col, text_col, image_weight, text_weight):
        # Inicializar CLIP
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        combined_features = []

        for _, row in df.iterrows():
            img_path = os.path.join(root_path, 'Nano', row[image_col])
            text = row[text_col]
            image = Image.open(img_path)
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            image_features = outputs.image_embeds.cpu().numpy().flatten() * image_weight
            text_features = outputs.text_embeds.cpu().numpy().flatten() * text_weight
            combined_feature = np.hstack((image_features, text_features))
            combined_features.append(combined_feature)
        df['combined_features'] = combined_features
        return df

    

    def cluster_images(df, use_pca, n_components=0.95):
        if use_pca:
            pca = PCA(n_components=n_components)
            features = pca.fit_transform(np.array(df['combined_features'].tolist()))
        else:
            features = np.array(df['combined_features'].tolist())

        silhouette_scores = []
        best_davies_bouldin_score = float("inf")
        best_calinski_harabasz_score = -1
        for k in range(2, 11):
            clustering = AgglomerativeClustering(n_clusters=k).fit(features)
            labels = clustering.labels_
            silhouette_avg = silhouette_score(features, labels)
            silhouette_scores.append(silhouette_avg)
            davies_bouldin = davies_bouldin_score(features, labels)
            best_davies_bouldin_score = min(best_davies_bouldin_score, davies_bouldin)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            best_calinski_harabasz_score = max(best_calinski_harabasz_score, calinski_harabasz)
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        clustering = AgglomerativeClustering(n_clusters=optimal_clusters).fit(features)
        df['activity_label'] = clustering.labels_
        df['activity_label'] = df['activity_label'].astype(str)
        print(f"Mejor Coeficiente de Silueta: {max(silhouette_scores)} con {optimal_clusters} clusters")
        print(f"Mejor Davies-Bouldin Score: {best_davies_bouldin_score}")
        print(f"Mejor Calinski-Harabasz Score: {best_calinski_harabasz_score}")
        return df

    
    
    def auto_process_id_assignment(df):
        activity_inicial = df['activity_label'].iloc[0]
        process_id = 1
        process_ids = [process_id]  
        for index, row in df.iterrows():
            if index != 0:  
                if row['activity_label'] == activity_inicial:
                    process_id += 1
                process_ids.append(process_id)
            else:
                continue
        df['process_id'] = process_ids
        return df

    
    ui_log = extract_features_from_images(ui_log, root_path, 'ocel:screenshot:name', 'header', image_weight=1, text_weight=1)
    ui_log = cluster_images(ui_log, use_pca=False, n_components=0.95)
    ui_log = auto_process_id_assignment(ui_log)
    folder_path = os.path.join(root_path, 'results')
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        ui_log.to_csv(os.path.join(folder_path, 'ui_log_process_discovery.csv'), index=False)
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
    
    
def process_discovery(log_path, scenario_path, root_path, execution):
    special_colnames = execution.case_study.special_colnames,
    configurations = execution.process_discovery.configurations,
    type = execution.process_discovery.type,
    skip = execution.process_discovery.preloaded
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
    process_discovery_id = request.GET.get("process_discovery_id")
    case_study_id = request.GET.get("case_study_id")
    process_discovery_list = ProcessDiscovery.objects.filter(case_study_id=case_study_id)
    for m in process_discovery_list:
        m.active = False
        m.save()
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    process_discovery.active = True
    process_discovery.save()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))

def set_as_process_discovery_inactive(request):
    process_discovery_id = request.GET.get("process_discovery_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if ProcessDiscovery.objects.get(pk=process_discovery_id).user != request.user:  
        raise ValidationError(_("Monitoring doesn't belong to the authenticated user."))
    if ProcessDiscovery.objects.get(pk=process_discovery_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Monitoring doesn't belong to the Case Study."))
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    process_discovery.active = False
    process_discovery.save()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))
    
def delete_process_discovery(request):
    process_discovery_id = request.GET.get("process_discovery_id")
    case_study_id = request.GET.get("case_study_id")
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    if request.user.id != process_discovery.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    process_discovery.delete()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))
