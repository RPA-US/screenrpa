import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import ListView, CreateView, DetailView
from django.core.exceptions import ValidationError
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from apps.analyzer.models import CaseStudy, Execution
from core.utils import read_ui_log_as_dataframe
from .models import ProcessDiscovery
from .forms import ProcessDiscoveryForm
from django.utils.translation import gettext_lazy as _
#Testing dependencies
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel
import torch
from keras.applications.vgg16 import VGG16
from PIL import Image
import tensorflow as tf
from tqdm import tqdm


def scene_level(log_path, scenario_path, execution):
    """
    Feature extraction WorkFlow
    """

    def load_model(model_type):
        if model_type == 'clip':
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor
        elif model_type == 'vgg':
            tf.config.set_visible_devices([], 'GPU')
            model = VGG16(weights='imagenet', include_top=False)
            return model, None
    
    def process_image_clip(image, text, processor, model, image_weight, text_weight):
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        image_features = outputs.image_embeds.cpu().numpy().flatten() * image_weight
        text_features = outputs.text_embeds.cpu().numpy().flatten() * text_weight
        return np.hstack((image_features, text_features))
    
    def process_image_vgg(img_path, model):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found")
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)   
        features = model.predict(img).flatten()
        return features
    
    def extract_features_from_images(df, scenario_path, image_col, text_col, image_weight, text_weight, model_type):
        model, processor_or_transform = load_model(model_type)
        combined_features = []

        if model_type == 'clip':
            pbar = tqdm(total=df.shape[0], desc='Extracting features with CLIP from Image')
        
        for index, row in df.iterrows():
            img_path = os.path.join(scenario_path, row[image_col])
            image = Image.open(img_path).convert('RGB')

            if model_type == 'clip':
                text = row[text_col]
                features = process_image_clip(image, text, processor_or_transform, model, image_weight, text_weight)
                pbar.update(1)  
            elif model_type == 'vgg':
                features = process_image_vgg(img_path, model)
            
            combined_features.append(features)

        if model_type == 'clip':
            pbar.close()  

        df['combined_features'] = combined_features
        return df
    
    '''
    Clustering Workflow
    '''

    def apply_pca(features, n_components):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(features)

    def evaluate_clusters(features, labels):
        silhouette_avg = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        return silhouette_avg, davies_bouldin, calinski_harabasz

    def perform_clustering(features, method, n_clusters=2):
        if method == 'agglomerative':
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'divisive': 
            clustering_model = KMeans(n_clusters=n_clusters)
        else:
            raise ValueError("Unsupported clustering method. Choose 'agglomerative' or 'divisive'.")
        
        clustering_model.fit(features)
        labels = clustering_model.labels_
        return labels

    def find_optimal_clusters(features, clustering_type, min_clusters=2, max_clusters=10):
        best_score = -1
        optimal_clusters = min_clusters
    
        for k in tqdm(range(min_clusters, max_clusters + 1), desc='Finding optimal clusters'):
            labels = perform_clustering(features, clustering_type, n_clusters=k)
            silhouette_avg, _, _ = evaluate_clusters(features, labels)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                optimal_clusters = k
                    
        optimal_labels = perform_clustering(features, clustering_type, n_clusters=optimal_clusters)
        silhouette_avg, davies_bouldin, calinski_harabasz = evaluate_clusters(features, optimal_labels)
             
        tqdm.write(f"Method: {clustering_type}, Optimal Clusters: {optimal_clusters}, Best Silhouette Score: {silhouette_avg}")
        tqdm.write(f"Davies-Bouldin Score: {davies_bouldin}, Calinski-Harabasz Score: {calinski_harabasz}")

        return optimal_clusters, optimal_labels

    def cluster_images(df, use_pca, clustering_type, n_components=0.95):
        tqdm.write('Starting Clustering...')
        features = np.array(df['combined_features'].tolist())
        
        if use_pca:
            features = apply_pca(features, n_components)
        
        _, labels = find_optimal_clusters(features, clustering_type)
        df['activity_label'] = labels.astype(str)
        
        return df

    
    '''
    Labeling WorkFlow
    '''
    def auto_labeling(df, remove_loops):
        activity_inicial = df['activity_label'].iloc[0]
        process_id = 1
        process_ids = [process_id]
        for index, row in df.iterrows():
            if index != 0:
                if row['activity_label'] == activity_inicial:
                    process_id += 1
                process_ids.append(process_id)
        df['process_id'] = process_ids
        if remove_loops: df = remove_duplicate_activities(df, 'activity_label')
        return df

    def manual_labeling(df):
        # Placeholder for manual labeling logic.
        # For now, it simply returns the DataFrame unchanged.
        return df

    def process_id_assignment(df, remove_loops, labeling_mode='automatic'):
        if labeling_mode == 'automatic':
            df = auto_labeling(df, remove_loops)
        elif labeling_mode == 'manual':
            df = manual_labeling(df)
        else:
            raise ValueError("Unsupported labeling mode. Choose 'automatic' or 'manual'.")
        return df
    
    def remove_duplicate_activities(df, activity_column):
        to_remove = df[activity_column].eq(df[activity_column].shift())
        df = df[~to_remove]
        
        return df

    '''
    Dendrogram generation WorkFlow
    '''
    
    def generate_dendrogram(df, show_dendrogram, features_column='combined_features', folder_path='results'):
        if not show_dendrogram:
            print("Dendrogram generation has been skipped.")
            return
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        features = np.array(df[features_column].tolist())
        linked = linkage(features, 'ward')

        plt.figure(figsize=(10, 7))
        dendrogram(linked,
                orientation='top',
                labels=df.index.to_list(),
                distance_sort='descending',
                show_leaf_counts=True)

        dendrogram_path = os.path.join(root_path, 'results', 'dendrogram.png')
        plt.savefig(dendrogram_path)
        plt.close() 
        print(f"Dendrogram saved in: {dendrogram_path}")

    
    '''
    Special parameters for the execution of the process discovery
    '''
    root_path = execution.exp_folder_complete_path
    special_colnames = execution.case_study.special_colnames

    '''
    Process Discovery Execution Parameters
    '''
    process_discovery = execution.process_discovery
    model_type = process_discovery.model_type
    clustering_type = process_discovery.clustering_type
    labeling = process_discovery.labeling
    image_weight = process_discovery.image_weight if model_type == 'clip' else 0.5
    text_weight = process_discovery.text_weight if model_type == 'clip' else 0.5
    use_pca = process_discovery.use_pca
    n_components = process_discovery.n_components
    show_dendrogram = process_discovery.show_dendrogram
    remove_loops = process_discovery.remove_loops
    text_column = process_discovery.text_column
   
    '''
    Process Discovery Execution Main Workflow
    '''
    ui_log = read_ui_log_as_dataframe(log_path)
    ui_log = extract_features_from_images(ui_log, scenario_path, special_colnames["Screenshot"], text_column, image_weight=image_weight, text_weight=text_weight, model_type=model_type)
    ui_log = cluster_images(ui_log, use_pca, clustering_type, n_components)
    ui_log = process_id_assignment(ui_log, remove_loops, labeling_mode=labeling)

    folder_path = os.path.join(root_path, 'results')
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        ui_log.to_csv(os.path.join(folder_path, 'ui_log_process_discovery.csv'), index=False)
    else:
        raise Exception(_("You selected a process discovery type that does not exist"))
    
    generate_dendrogram(ui_log, show_dendrogram=show_dendrogram)

    print(folder_path)
    return folder_path, ui_log
    

def process_level(folder_path, df, execution):
    special_colnames = execution.case_study.special_colnames

    def petri_net_process(df, special_colnames):
        formatted_df = pm4py.format_dataframe(df, case_id='process_id', activity_key='activity_label', timestamp_key=special_colnames['Timestamp'])
        event_log = pm4py.convert_to_event_log(formatted_df)
        process_tree = inductive_miner.apply(event_log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
        dot = pn_visualizer.apply(net, initial_marking, final_marking)
        dot_path = os.path.join(folder_path, 'pn.dot')
        with open(dot_path, 'w') as f:
            f.write(dot.source)

    def bpmn_process(df, special_colnames):
        formatted_df = pm4py.format_dataframe(df, case_id='process_id', activity_key='activity_label', timestamp_key=special_colnames['Timestamp'])
        event_log = pm4py.convert_to_event_log(formatted_df)
        bpmn_model = pm4py.discover_bpmn_inductive(event_log)
        dot = bpmn_visualizer.apply(bpmn_model)
        dot_path = os.path.join(folder_path, 'bpmn.dot')
        with open(dot_path, 'w') as f:
            f.write(dot.source)
        bpmn_exporter.apply(bpmn_model, os.path.join(folder_path, 'bpmn.bpmn'))


    petri_net_process(df, special_colnames)
    try:
        bpmn_process(df, special_colnames)
    except Exception as e:
        print(f'Error generating BPMN: {e} Continuing with Petrinet...')
    
    
def process_discovery(log_path, scenario_path, execution):
    # log_path -> media/unzipped/exec_1/Nano/log.csv
    # scenario_path -> media/unzipped/exec_1/Nano/
    # root_path -> media/unzipped/exec_1/

    # root_path + "results/" + "ui_log_process_discovery.csv"
    #Pasar execution.process_discovery
    folder_path, ui_log = scene_level(log_path, scenario_path, execution)
    process_level(folder_path, ui_log, execution)
        
    
class ProcessDiscoveryCreateView(CreateView):
    model = ProcessDiscovery
    form_class = ProcessDiscoveryForm
    template_name = "processdiscovery/create.html"

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if 'ProcessDiscovery' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def post(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if 'ProcessDiscovery' in case_study.available_phases:
            return super().post(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    

    def get_form_kwargs(self):
        kwargs = super(ProcessDiscoveryCreateView, self).get_form_kwargs()
        case_study_id = self.kwargs.get('case_study_id')
        if case_study_id:
            try:
                case_study_instance = CaseStudy.objects.get(pk=case_study_id)
                kwargs['case_study'] = case_study_instance
            except CaseStudy.DoesNotExist:
                raise ValidationError(_("CaseStudy with the given ID does not exist."))
        return kwargs

    def get_context_data(self, **kwargs):
        context = super(ProcessDiscoveryCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context    

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        case_study_id = self.kwargs.get('case_study_id')
        if case_study_id:
            self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class ProcessDiscoveryListView(ListView):
    model = ProcessDiscovery
    template_name = "processdiscovery/list.html"
    paginate_by = 50

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if 'ProcessDiscovery' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))

    def get_context_data(self, **kwargs):
        context = super(ProcessDiscoveryListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos por case_study_id
        if search:
            queryset = ProcessDiscovery.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, title__icontains=search).order_by('-created_at')
        else:
            queryset = ProcessDiscovery.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset
    

class ProcessDiscoveryDetailView(DetailView):
    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):

        process_discovery = get_object_or_404(ProcessDiscovery, id=kwargs["process_discovery_id"])
        process_discovery_form = ProcessDiscoveryForm(read_only=True, instance=process_discovery)

        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
            if 'ProcessDiscovery' in case_study.available_phases: 
                context= {"process_discovery": process_discovery, 
                            "case_study_id": case_study.id,
                            "form": process_discovery_form,}
            else:
                return HttpResponseRedirect(reverse("analyzer:casestudy_list"))

        elif 'execution_id' in kwargs:
            execution = get_object_or_404(Execution, id=kwargs['execution_id'])
            if execution.process_discovery:
            
                context= {"process_discovery": process_discovery, 
                            "execution_id": execution.id,
                            "form": process_discovery_form,}
                    
                return render(request, "processdiscovery/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:execution_list"))
        

        
        

    

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

##########################################

class ProcessDiscoveryResultDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        scenario = request.GET.get('scenario')
        

        if scenario == None:
            #scenario = "1"
            scenario = execution.scenarios_to_study[0] # by default, the first one that was indicated
              
        path_to_bpmn_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "bpmn.bpmn")


        with open('/screenrpa/'+path_to_bpmn_file, 'r', encoding='utf-8') as file:
            bpmn_content = file.read()

        # Include CSV data in the context for the template
        context = {
            "execution_id": execution.id,
            "prueba": bpmn_content,  # Png to be used in the HTML template
            "scenarios": execution.scenarios_to_study,
            "scenario": scenario
            }
        return render(request, 'processdiscovery/result.html', context)
    


####################################################################

def ProcessDiscoveryDownload(request, execution_id):

    execution = get_object_or_404(Execution, pk=execution_id)
    #execution = get_object_or_404(Execution, id=request.kwargs["execution_id"])
    scenario = request.GET.get('scenario')
    
    if scenario is None:
        scenario = execution.scenarios_to_study[0]  # by default, the first one that was indicated
              
    path_to_bpmn_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "bpmn.bpmn")
    
    try:
        # Asegúrate de que la ruta absoluta sea correcta
        full_file_path = os.path.join('/screenrpa', path_to_bpmn_file)
        with open(full_file_path, 'rb') as archivo:
            response = HttpResponse(archivo.read(), content_type="application/octet-stream")
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(full_file_path)}-{scenario}"'
            return response
        
    except FileNotFoundError:

        print(f"File not found: {path_to_bpmn_file}")
        return HttpResponse("Sorry, the file was not found.", status=404)
    

    
####################################################################