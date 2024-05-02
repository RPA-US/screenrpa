import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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
            plt.title(_('Dendrogram'))
            plt.xlabel(_('Screenshots'))
            plt.ylabel(_('Distance'))
            plt.savefig(os.path.join(root_path + 'ui_log_states_dendrogram.png'))

        # Obtener los clusters utilizando un umbral de distancia
        if "similarity_th" in configurations:
            threshold = float(configurations["similarity_th"]) * np.max(distance_matrix[:, 2])
        else:
            threshold = 0.1 * np.max(distance_matrix[:, 2])
            print(_("Threshold: ") + str(threshold))
        cluster_labels = fcluster(distance_matrix, threshold, criterion='distance')

        # Agregar los clusters al dataframe de UI log
        ui_log[special_colnames['Activity']] = cluster_labels
        # ui_log['trace_id'] = list(range(1, ui_log.shape[0]+1))
        
        # Guardar el resultado en un nuevo archivo CSV
        ui_log.to_csv(root_path + 'ui_log_clustered.csv', index=False)
    else:
        raise Exception(_("You selected a process discovery type that does not exist"))


def process_level(log_path, root_path, special_colnames, configurations, type):
    dataframe = pm4py.format_dataframe(dataframe, case_id=special_colnames['Case'], activity_key=special_colnames['Activity'], timestamp_key=special_colnames['Timestamp'])
    event_log = pm4py.convert_to_event_log(dataframe)

    process_model = pm4py.discover_bpmn_inductive(event_log)
    # pm4py.view_bpmn(process_model)
    pm4py.save_vis_bpmn(process_model, "bpm.png", "white")
    
    

def process_discovery(log_path, root_path, execution):
    special_colnames = execution.case_study.special_colnames,
    configurations = execution.process_discovery.configurations,
    type = execution.process_discovery.type,
    skip = execution.process_discovery.preloaded
    if not skip:
        scene_level(log_path, root_path, special_colnames, configurations, type)
        process_level(log_path, root_path, special_colnames, configurations, type)
        
        
    
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
        
        
        form = ProcessDiscoveryForm(read_only=True, instance=process_discovery)  # Todos los campos estarán desactivados

        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])

            context= {"process_discovery": process_discovery, 
                  "case_study_id": case_study.id,
                  "form": form,}

        elif 'execution_id' in kwargs:
            execution = get_object_or_404(Execution, id=kwargs['execution_id'])

        context= {"process_discovery": process_discovery, 
                  "execution_id": execution.id,
                  "form": form,}
        return render(request, "processdiscovery/detail.html", context)



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
        return HttpResponse("Lo siento, el archivo no se encontró.", status=404)
    

    
####################################################################