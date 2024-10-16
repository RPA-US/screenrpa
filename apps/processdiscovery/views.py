import base64
import io
import os
from tempfile import NamedTemporaryFile
import zipfile
import cv2
from graphviz import Source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import ListView, CreateView, DetailView, UpdateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError, PermissionDenied
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from apps.analyzer.models import CaseStudy, Execution
from apps.decisiondiscovery.utils import rename_columns_with_centroids
from core.utils import read_ui_log_as_dataframe
from core.settings import PROCESS_DISCOVERY_LOG_FILENAME, ENRICHED_LOG_SUFFIX
from apps.processdiscovery.utils import Process, DecisionPoint, Branch, Rule
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
import pygraphviz as pgv 

def scene_level(log_path, scenario_path, execution):
    """
    Feature extraction WorkFlow
    """
    special_colnames= execution.case_study.special_colnames

    def load_model(model_type):
        if model_type == 'clip':
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor
        elif model_type == 'vgg':
            # tf.config.set_visible_devices([], 'GPU')
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
        df[special_colnames['Activity']] = labels.astype(str)
        
        return df

    
    '''
    Labeling WorkFlow
    '''
    def auto_labeling(df, fe_log, remove_loops):
        if remove_loops: df, fe_log = remove_duplicate_activities(df, fe_log, special_colnames['Activity'])
        activity_inicial = df[special_colnames['Activity']].iloc[0]
        trace_id = 1
        trace_ids = [trace_id]
        for index, row in df.iterrows():
            if index != 0:
                if row[special_colnames['Activity']] == activity_inicial:
                    trace_id += 1
                trace_ids.append(trace_id)
        df['trace_id'] = trace_ids
        return df, fe_log

    def manual_labeling(df):
        # Placeholder for manual labeling logic.
        # For now, it simply returns the DataFrame unchanged.
        return df

    def trace_id_assignment(df, fe_log, remove_loops, labeling_mode='automatic'):
        if labeling_mode == 'automatic':
            df, fe_log = auto_labeling(df, fe_log, remove_loops)
        elif labeling_mode == 'manual':
            df = manual_labeling(df)
        else:
            raise ValueError("Unsupported labeling mode. Choose 'automatic' or 'manual'.")
        return df, fe_log
    
    def remove_duplicate_activities(df, fe_log, activity_column):
        to_remove = df[activity_column].eq(df[activity_column].shift())
        index_to_remove = df[to_remove].index
        df = df[~to_remove]
        if fe_log is not None:
            fe_log = fe_log[~to_remove]
        
        return df, fe_log

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

        dendrogram_path = os.path.join(scenario_path + '_results', 'dendrogram.png')
        plt.savefig(dendrogram_path)
        plt.close() 
        print(f"Dendrogram saved in: {dendrogram_path}")
    
    '''
    Special parameters for the execution of the process discovery
    '''
    # root_path = execution.exp_folder_complete_path
    

    '''
    Process Discovery Execution Parameters
    '''
    process_discovery = execution.process_discovery
    model_type = process_discovery.model_type
    clustering_type = process_discovery.clustering_type
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
    if execution.feature_extraction_technique:
        fe_log = read_ui_log_as_dataframe(os.path.join(scenario_path + "_results", 'log' + ENRICHED_LOG_SUFFIX + '.csv'))
    else:
        fe_log = None
    ui_log = extract_features_from_images(ui_log, scenario_path, special_colnames["Screenshot"], text_column, image_weight=image_weight, text_weight=text_weight, model_type=model_type)
    ui_log = cluster_images(ui_log, use_pca, clustering_type, n_components)
    ui_log, fe_log = trace_id_assignment(ui_log, fe_log, remove_loops)

    folder_path = scenario_path + '_results'
    print(folder_path)
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    #ACORTARLE EL NOMBRE A LAS COLUMNAS DEL PD_LOG POR LEGIBILIDAD
    ui_log = rename_columns_with_centroids(ui_log)
    ui_log.to_csv(os.path.join(folder_path, PROCESS_DISCOVERY_LOG_FILENAME), index=False)
    
    generate_dendrogram(ui_log, show_dendrogram=show_dendrogram)

    return folder_path, ui_log, fe_log
    

def process_level(folder_path, df, fe_log, execution):
    special_colnames = execution.case_study.special_colnames

    def petri_net_process(df, special_colnames):
        formatted_df = pm4py.format_dataframe(df, case_id=special_colnames['Case'], activity_key=special_colnames['Activity'], timestamp_key=special_colnames['Timestamp'])
        event_log = pm4py.convert_to_event_log(formatted_df)
        process_tree = inductive_miner.apply(event_log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
        dot = pn_visualizer.apply(net, initial_marking, final_marking)
        dot_path = os.path.join(folder_path, 'pn.dot')
        with open(dot_path, 'w') as f:
            f.write(dot.source)

    def bpmn_process(df, fe_log, special_colnames):
        formatted_df = pm4py.format_dataframe(df, case_id=special_colnames['Case'], activity_key=special_colnames['Activity'], timestamp_key=special_colnames['Timestamp'])
        event_log = pm4py.convert_to_event_log(formatted_df)
        bpmn_model = pm4py.discover_bpmn_inductive(event_log)
        dot = bpmn_visualizer.apply(bpmn_model)
        dot_path = os.path.join(folder_path, 'bpmn.dot')
        with open(dot_path, 'w') as f:
            f.write(dot.source)
        bpmn_exporter.apply(bpmn_model, os.path.join(folder_path, 'bpmn.bpmn'))

        # Give name to gates without
        i = 0
        nodes = bpmn_model.get_nodes()
        for node in nodes:
            if type(node) == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway:
                if node.name == '':
                    # node.name is a propperty and cannot be set. The class attribute is __name
                    node._BPMNNode__name = f'xor_{i}'
                    i += 1
        
        # Get the decision points in the model (diverging exclusive gateways)
        node_start = list(filter(lambda node: type(node) == pm4py.objects.bpmn.obj.BPMN.StartEvent, nodes))[0]
        node_end = list(filter(lambda node: type(node) == pm4py.objects.bpmn.obj.BPMN.NormalEndEvent, nodes))[0]

        def explore_branch(node_start, visited, last_act) -> tuple[Branch, pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway|pm4py.objects.bpmn.obj.BPMN.NormalEndEvent, set]:
            branch_start = node_start
            cn = branch_start # Current node
            dps: list[DecisionPoint] = []
            max_iterations = 1000  # Define a reasonable maximum number of iterations
            iteration_count = 0

            # The loop continues until the current node (cn) is the end node
            while cn != node_end:
                iteration_count += 1
                if iteration_count > max_iterations:
                    raise Exception("Infinite loop detected in BPMN exploration. Exceeded maximum iterations.")

                # If the next node is a converging ExclusiveGateway or a NormalEndEvent, the branch is finished
                if type(cn) == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway and cn.get_gateway_direction() == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway.Direction.CONVERGING or type(cn) == pm4py.objects.bpmn.obj.BPMN.NormalEndEvent:
                    return Branch(branch_start.name, branch_start.id, dps), cn, visited

                # The next node is the target of the first outgoing arc from the current node
                next_node = cn.get_out_arcs()[0].target

                # If the next node has already been visited, an exception is raised
                # This is because the current implementation does not support loops in the BPMN model
                if cn in visited:
                    raise Exception("error: end node not reached. current bpmn_bfs does not support loops. state:" + cn.name + " " + cn.id + " " + type(cn))

                # If the next node is a Task or a StartEvent, it is added to the visited set and becomes the current node
                if type(cn) == pm4py.objects.bpmn.obj.BPMN.Task or type(cn) == pm4py.objects.bpmn.obj.BPMN.StartEvent:
                    visited.add(cn)
                    last_act = cn.name
                    cn = next_node

                # If the next node is a diverging ExclusiveGateway, it is added to the visited set
                # Then, for each outgoing arc from the gateway, the function recursively explores the branch
                # A Rule is created for each branch and added to the rules list
                # A DecisionPoint is created and added to the dps list
                elif type(cn) == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway and cn.get_gateway_direction() == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway.Direction.DIVERGING:
                    visited.add(cn)
                    branches: list[Branch] = []
                    rules: list[Rule] = []
                    for arc in cn.get_out_arcs():
                        ## Handle empty branches 
                        if type(arc.target) == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway and \
                        arc.target.get_gateway_direction() == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway.Direction.CONVERGING:
                            branch = Branch(arc.target.name, arc.target.id, [])
                            rule = Rule([], arc.target.name)
                            converge_gateway = arc.target
                        else:
                            branch, converge_gateway, visited = explore_branch(arc.target, visited, last_act)
                            rule = Rule([], arc.target.name)
                        rules.append(rule)
                        branches.append(branch)
                    dps.append(DecisionPoint(cn.id, last_act, branches, rules))

                    # If the gateway at the end of the branch is a NormalEndEvent, the function return because the BPMN discovery has finished
                    if type(converge_gateway) == pm4py.objects.bpmn.obj.BPMN.NormalEndEvent:
                        return Branch(branch_start.name, branch_start.id, dps), next_node, visited
                    else:
                        cn = converge_gateway.get_out_arcs()[0].target

                # Handling any other case
                else:
                    raise Exception(f"ScreenRPA does not currently support with processes containing elements of type {type(next_node)}")
            # Return at the end of the BPMN
            return Branch(branch_start.name, branch_start.id, dps), None, visited

        def bpmn_bfs(node_start, node_end) -> Process:
            return Process(explore_branch(node_start, set(), node_start.name)[0].decision_points)

        def add_trace_to_log(process, df) -> None:
            # Navigate the log and add to each row info about the branches taken on each decision point
            # The path is reset after each trace_id change
            # This is done by updating the id of the branch after reaching a "prevAct" on the decision points
            current_trace_id = 1
            current_dp = None # Use to track what dp a branch will belong to
            current_branches = {}

            # First we make columns for each of the decision points, with the id of the decision point as the column name
            dps = process.get_non_empty_dp_flattened()
            #execution.process_discovery.activities_before_dps = [dp for dp in dps]
            # execution.process_discovery.save()
            branches = process.get_all_branches_flattened()
            for dp in dps:
                df[dp.id] = None

            for index, row in df.iterrows():
                if str(row[special_colnames['Case']]) != str(current_trace_id):
                    # Insert the branches taken in the previous trace
                    current_trace_rows = df.loc[df[special_colnames['Case']] == current_trace_id]
                    for index, trace_row in current_trace_rows.iterrows():
                        for passed_dp in current_branches.keys():
                            df.at[index, passed_dp] = current_branches[passed_dp]
                    # Update trace info
                    current_trace_id = row[special_colnames['Case']]
                    current_branches = {}

                act_label = row[special_colnames['Activity']]
                if current_dp is not None:
                    if any(str(branch.label) == str(act_label) for branch in branches):
                        # Compute the value for the row on column dp_id
                        branch_label = (list(filter(lambda branch: str(branch.label) == str(act_label), branches))[0].label)
                        current_branches[current_dp] = branch_label
                for dp in dps:
                    if str(dp.prevAct) == str(act_label):
                        current_dp = dp.id

                # Register trace of the last Case or trace_id
                last_trace_row = df.loc[df[special_colnames['Case']] == current_trace_id]
                for index, trace_row in last_trace_row.iterrows():
                    for passed_dp in current_branches.keys():
                        df.at[index, passed_dp] = current_branches[passed_dp]
                        
            df = variant_column(df, execution.case_study.special_colnames)
            # Save log to csv
            df.to_csv(os.path.join(folder_path, PROCESS_DISCOVERY_LOG_FILENAME), index=False)
            return df

        try:
            process = bpmn_bfs(node_start, node_end)
            json.dump(process.to_json(), open(os.path.join(folder_path, 'traceability.json'), 'w'))
            df = add_trace_to_log(process, df)
        except Exception as e:
            print(e)

        if fe_log is not None:
            # Save full log (pd + fe)
            fe_log.drop(columns=df.columns, inplace=True, errors='ignore')
            df.drop(columns=["index"], inplace=True, errors='ignore')
            fe_log = fe_log.reset_index(drop=True)
            full_log = pd.concat([df, fe_log], axis=1)
            full_log.to_csv(os.path.join(folder_path, 'pipeline_log.csv'), index=False)

    petri_net_process(df, special_colnames)
    try:
        bpmn_process(df, fe_log, special_colnames)
    except Exception as e:
        print(f'Error generating BPMN: {e} Continuing with Petrinet...')
    
def process_discovery(log_path, scenario_path, execution):
    # log_path -> media/unzipped/exec_1/Nano/log.csv
    # scenario_path -> media/unzipped/exec_1/Nano/
    # root_path -> media/unzipped/exec_1/

    # root_path + "results/" + PROCESS_DISCOVERY_LOG_FILENAME
    #Pasar execution.process_discovery
    folder_path, ui_log, fe_log = scene_level(log_path, scenario_path, execution)
    process_level(folder_path, ui_log, fe_log, execution)
        
    
class ProcessDiscoveryCreateView(LoginRequiredMixin, CreateView):
    login_url = '/login/'
    model = ProcessDiscovery
    form_class = ProcessDiscoveryForm
    template_name = "processdiscovery/create.html"

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Case Study not found.")
        elif case_study.user != request.user:
            raise PermissionDenied("Case Study doesn't belong to the authenticated user.")

        if 'ProcessDiscovery' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def post(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Case Study not found.")
        elif case_study.user != request.user:
            raise PermissionDenied("Case Study doesn't belong to the authenticated user.")

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

class ProcessDiscoveryListView(LoginRequiredMixin, ListView):
    login_url = '/login/'
    model = ProcessDiscovery
    template_name = "processdiscovery/list.html"
    paginate_by = 50

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Case Study not found.")
        elif case_study.user != request.user:
            raise PermissionDenied("Case Study doesn't belong to the authenticated user.")

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
    

class ProcessDiscoveryDetailView(LoginRequiredMixin, UpdateView):
    login_url = '/login/'
    model = ProcessDiscovery
    form_class = ProcessDiscoveryForm
    template_name = "processdiscovery/detail.html"
    success_url = "/pd/bpmn/list/"

    def get_object(self, queryset=None):
        return get_object_or_404(ProcessDiscovery, id=self.kwargs.get('process_discovery_id'))
    
    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        if self.object.freeze:
            raise ValidationError("This object cannot be edited.")
        if not self.object.case_study.user == self.request.user:
            raise PermissionDenied("This object doesn't belong to the authenticated")
        self.object.save()
        return HttpResponseRedirect(self.get_success_url() + str(self.object.case_study.id))

    def get(self, request, *args, **kwargs):
        process_discovery = get_object_or_404(ProcessDiscovery, id=kwargs["process_discovery_id"])
        if process_discovery.case_study.user != request.user:
            raise PermissionDenied("Process Discovery doesn't belong to the authenticated user.")

        process_discovery_form = ProcessDiscoveryForm(read_only=process_discovery.freeze, instance=process_discovery)

        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
            if 'ProcessDiscovery' in case_study.available_phases: 
                context= {"process_discovery": process_discovery, 
                            "case_study_id": case_study.id,
                            "form": process_discovery_form,}
                return render(request, "processdiscovery/detail.html", context)
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


@login_required(login_url='/login/')
def set_as_process_discovery_active(request):
    process_discovery_id = request.GET.get("process_discovery_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found.")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        raise PermissionDenied("Case Study doesn't belong to the authenticated user.")
    elif not ProcessDiscovery.objects.get(pk=process_discovery_id):
        return HttpResponse(status=404, content="Process Discovery not found.")
    elif not ProcessDiscovery.objects.get(pk=process_discovery_id).case_study == CaseStudy.objects.get(pk=case_study_id):
        raise PermissionDenied("Case Study doesn't belong to the authenticated user.")
    process_discovery_list = ProcessDiscovery.objects.filter(case_study_id=case_study_id)
    for m in process_discovery_list:
        m.active = False
        m.save()
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    process_discovery.active = True
    process_discovery.save()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))

@login_required(login_url='/login/')
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
    
@login_required(login_url='/login/')
def delete_process_discovery(request):
    process_discovery_id = request.GET.get("process_discovery_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found.")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        raise PermissionDenied("Case Study doesn't belong to the authenticated user.")
    elif not ProcessDiscovery.objects.get(pk=process_discovery_id):
        return HttpResponse(status=404, content="Process Discovery not found.")
    elif not ProcessDiscovery.objects.get(pk=process_discovery_id).case_study == CaseStudy.objects.get(pk=case_study_id):
        raise PermissionDenied("Process Discovery doesn't belong to the Case Study.")
    process_discovery = ProcessDiscovery.objects.get(id=process_discovery_id)
    if request.user.id != process_discovery.user.id:
        raise PermissionDenied("This object doesn't belong to the authenticated user")
    process_discovery.delete()
    return HttpResponseRedirect(reverse("processdiscovery:processdiscovery_list", args=[case_study_id]))

##########################################

def dot_to_png_base64(dot_path):
    try:
        # Cargar el contenido del archivo .dot
        with open(dot_path, 'r') as file:
            dot_content = file.read()
        
        # Crear el gráfico a partir del contenido DOT
        graph = Source(dot_content)
        graph.format = 'png'
        
        # Guardar la imagen a un archivo temporal
        with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            graph_path = graph.render(filename=temp_file.name, format='png', cleanup=True)
        
        # Leer la imagen generada en formato binario
        with open(graph_path, 'rb') as image_file:
            png_image = image_file.read()
        
        # Convertir la imagen a base64
        image_base64 = base64.b64encode(png_image).decode('utf-8')
        
        return f'data:image/png;base64,{image_base64}'
    
    except Exception as e:
        print(f"Error al procesar el gráfico: {e}")
        return None
    
def variant_column(df, colnames):
    sequence_to_variant = {}
    next_variant_number = 1
    # Función para generar el mapeo de variantes y asignar valores a la columna auto_variant
    def assign_variant(trace):
        nonlocal next_variant_number  # Declarar next_variant_number como global
        cadena = ""
        for e in trace[colnames['Activity']].tolist():
            cadena = cadena + str(e)
        if cadena not in sequence_to_variant:
            sequence_to_variant[cadena] = next_variant_number
            next_variant_number += 1
        trace['auto_variant'] = sequence_to_variant[cadena]
        return trace

    # Aplicar la función a cada grupo de trace_id y asignar el resultado al DataFrame original
    df=df.groupby(colnames['Case']).apply(assign_variant).reset_index(drop=True)
    return df
    
    
           
        
def cambiar_color_nodos_y_caminos(labels, path_to_dot_file):
    # Cargar el contenido del archivo .dot
    with open(path_to_dot_file, 'r') as file:
        dot_content = file.read()
    # Crear un grafo desde el contenido del archivo .dot
    grafo = pgv.AGraph(string=dot_content)
    
    # Crear un conjunto de nodos que necesitamos colorear
    nodos_a_colorear = set()
    labels = [str(elemento) for elemento in labels]
    # Recorrer todos los nodos del grafo y colorear los nodos correspondientes
    for nodo in grafo.nodes():
        if nodo.attr['label'] in labels:
            nodo.attr['fillcolor'] = "yellow"
            nodo.attr['style'] = 'filled'
            nodos_a_colorear.add(nodo.name)
        elif nodo.attr['label'] == "":
            nodo.attr['label'] = " "
            nodo.attr['fillcolor'] = nodo.attr['fillcolor']
            nodo.attr['style'] = nodo.attr['style']
        else:
            nodo.attr['style'] = 'filled'
            

    # Recorrer todas las aristas del grafo y colorear las que conectan nodos en la lista
    # Identificar y colorear los nodos de tipo "diamond" entre los nodos etiquetados
    for edge in grafo.edges():
        if edge[0] in nodos_a_colorear and edge[1] not in nodos_a_colorear:
            nodo_destino = grafo.get_node(edge[1])
            if nodo_destino.attr['shape'] == 'diamond':
                nodo_destino.attr['fillcolor'] = "yellow"
                nodo_destino.attr['style'] = 'filled'
                nodos_a_colorear.add(nodo_destino.name)
        elif edge[1] in nodos_a_colorear and edge[0] not in nodos_a_colorear:
            nodo_origen = grafo.get_node(edge[0])
            if nodo_origen.attr['shape'] == 'diamond':
                nodo_origen.attr['fillcolor'] = "yellow"
                nodo_origen.attr['style'] = 'filled'
                nodos_a_colorear.add(nodo_origen.name)

    # Colorear las aristas conectadas entre los nodos coloreados
    for edge in grafo.edges():
        if edge[0] in nodos_a_colorear and edge[1] in nodos_a_colorear:
            edge.attr['color'] = "blue"
            edge.attr['penwidth'] = 2.0

    # Configurar atributos globales del grafo
    grafo.graph_attr.update(bgcolor='white', rankdir='LR')
    grafo.graph_attr['overlap'] = 'false'
    grafo.format = 'dot'

    # Guardar la imagen a un archivo temporal
    temp_file = NamedTemporaryFile(delete=False, suffix='.dot')
    grafo.write(temp_file.name)
    modified_dot_path = temp_file.name

    return modified_dot_path
    


class ProcessDiscoveryResultDetailView(LoginRequiredMixin, DetailView):
    login_url = '/login/'

    def get(self, request, *args, **kwargs):
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        scenario = request.GET.get('scenario')
        selected_variant = request.GET.get('variant')
        colnames= execution.case_study.special_colnames
        if not execution:
            return HttpResponse(status=404, content="Execution not found.")
        elif not execution.case_study.user == request.user:
            raise PermissionDenied("Execution doesn't belong to the authenticated user.")

        if scenario is None:
            scenario = execution.scenarios_to_study[0]  # por defecto, el primer escenario indicado
              
        path_to_bpmn_file = os.path.join(execution.exp_folder_complete_path, scenario + "_results", "bpmn.dot")

        df = read_ui_log_as_dataframe(os.path.join(execution.exp_folder_complete_path, scenario + '_results', PROCESS_DISCOVERY_LOG_FILENAME))
        
        variants = df[colnames['Variant']].unique().tolist()

        variant_image_base64 = None
        if selected_variant:
            labels_for_variant = df[df[colnames['Variant']] == int(selected_variant)][execution.case_study.special_colnames['Activity']].unique().tolist()
            modified_dot_path  = cambiar_color_nodos_y_caminos(labels_for_variant, path_to_bpmn_file)
            variant_image_base64 = dot_to_png_base64(modified_dot_path)
        else:
            selected_variant = None
            variant_image_base64=dot_to_png_base64(path_to_bpmn_file)

        # Incluir los datos CSV en el contexto para la plantilla
        context = {
            "execution_id": execution.id,
            "scenarios": execution.scenarios_to_study,
            "scenario": scenario,
            "variants": variants,
            "selected_variant": selected_variant,
            "variant_image": variant_image_base64
        }
        return render(request, 'processdiscovery/result.html', context)  


####################################################################

    
def ProcessDiscoveryDownload(request, execution_id):
    execution = get_object_or_404(Execution, pk=execution_id)
    scenario = request.GET.get('scenario')
    
    if scenario is None:
        scenario = execution.scenarios_to_study[0]  # by default, the first one that was indicated

    # Construir las rutas completas a los archivos
    path_to_bpmn_dot = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "bpmn.dot")
    path_to_bpmn_bpmn = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "bpmn.bpmn")
    
    # Nombre del archivo ZIP a crear
    #zip_filename = f"bpmn_files_{execution_id}_{scenario}.zip"
    zip_filename = f"{scenario}_pd_results.zip"

    try:
        # Crear un archivo ZIP en memoria
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for file_path, arcname in [(path_to_bpmn_dot, "bpmn.dot"), (path_to_bpmn_bpmn, "bpmn.bpmn")]:
                # Verificar si el archivo existe
                if os.path.exists(file_path):
                    zip_file.write(file_path, arcname)
                else:
                    print(f"File not found: {file_path}")
        
        zip_buffer.seek(0)
        
        # Crear la respuesta HTTP con el archivo ZIP para descargar
        response = HttpResponse(zip_buffer, content_type="application/zip")
        response['Content-Disposition'] = f'attachment; filename="{zip_filename}"'
        return response
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return HttpResponse("Sorry, there was an error processing your request.", status=500)
    
####################################################################
