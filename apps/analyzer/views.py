import csv
import os
import json
import random
import time
import threading
from tqdm import tqdm
from art import tprint
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User 
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.db import transaction
import zipfile
from django.http import HttpResponse, HttpResponseRedirect, FileResponse
from django.views.generic import ListView, DetailView, CreateView, FormView, DeleteView
from django.utils.translation import gettext_lazy as _
from rest_framework import generics, status, viewsets #, permissions
from rest_framework.response import Response
# Home pages imports
from django import template
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.template import loader
# Settings variables
from core.settings import PRIVATE_STORAGE_ROOT, DEFAULT_PHASES, SCENARIO_NESTED_FOLDER, ACTIVE_CELERY
# Apps imports
from apps.decisiondiscovery.views import decision_tree_training, extract_training_dataset
from apps.featureextraction.views import ui_elements_classification, feature_extraction_technique
from apps.featureextraction.SOM.detection import ui_elements_detection
from apps.featureextraction.relevantinfoselection.prefilters import prefilters
from apps.featureextraction.relevantinfoselection.postfilters import postfilters
from apps.processdiscovery.views import process_discovery
from apps.behaviourmonitoring.log_mapping.gaze_monitoring import monitoring
from apps.analyzer.models import CaseStudy, Execution   
from apps.behaviourmonitoring.models import Monitoring
from apps.featureextraction.models import Prefilters, UIElementsClassification, UIElementsDetection, Postfilters, FeatureExtractionTechnique
from apps.processdiscovery.models import ProcessDiscovery
from apps.decisiondiscovery.models import ExtractTrainingDataset, DecisionTreeTraining
from apps.analyzer.forms import CaseStudyForm
from apps.analyzer.serializers import CaseStudySerializer
from apps.featureextraction.serializers import PrefiltersSerializer, UIElementsDetectionSerializer, UIElementsClassificationSerializer, PostfiltersSerializer, FeatureExtractionTechniqueSerializer
from apps.behaviourmonitoring.serializers import MonitoringSerializer
from apps.processdiscovery.serializers import ProcessDiscoverySerializer
from apps.decisiondiscovery.serializers import DecisionTreeTrainingSerializer, ExtractTrainingDatasetSerializer
from apps.analyzer.tasks import celery_task_process_case_study
from apps.analyzer.utils import get_foldernames_as_list
from apps.analyzer.collect_results import experiments_results_collectors

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

def generate_case_study(execution, path_scenario, times):
    log_filename = 'log.csv'
    log_path = os.path.join(path_scenario, log_filename)
    
    n = 0
    for i, function_to_exec in enumerate(DEFAULT_PHASES):
        if getattr(execution, function_to_exec) is not None:
            phase_has_preloaded = getattr(execution, function_to_exec).preloaded
            if phase_has_preloaded:
                times[n] = {function_to_exec: {"duration": None, "preloaded": True}}
            else:
                times[n] = {}
                if function_to_exec == "decision_tree_training":
                    res, fe_checker, tree_times, columns_len = eval(function_to_exec)(log_path, path_scenario, execution)
                    times[n][function_to_exec] = tree_times
                    times[n][function_to_exec]["columns_len"] = columns_len
                    # times[n][function_to_exec]["tree_levels"] = tree_levels
                    times[n][function_to_exec]["accuracy"] = res
                    times[n][function_to_exec]["feature_checker"] = fe_checker
                elif function_to_exec == "feature_extraction_technique":
                    if (getattr(execution, function_to_exec).type == "SINGLE" and i == 5) or (getattr(execution, function_to_exec).type == "AGGREGATE" and i == 8):
                        start_t = time.time()
                        num_UI_elements, num_screenshots, max_ui_elements, min_ui_elements = eval(function_to_exec)(log_path, path_scenario, execution)
                        times[n][function_to_exec] = {"duration": float(time.time()) - float(start_t)}
                        # Additional feature extraction metrics
                        times[n][function_to_exec]["num_UI_elements"] = num_UI_elements
                        times[n][function_to_exec]["num_screenshots"] = num_screenshots
                        times[n][function_to_exec]["max_#UI_elements"] = max_ui_elements
                        times[n][function_to_exec]["min_#UI_elements"] = min_ui_elements
                elif function_to_exec == "prefilters" or function_to_exec == "postfilters" or function_to_exec == "ui_elements_detection":
                # elif function_to_exec == "prefilters" or function_to_exec == "postfilters" or (function_to_exec == "ui_elements_detection" and to_exec_args["ui_elements_detection"][-1] == False):
                    filtering_times = eval(function_to_exec)(log_path, path_scenario, execution)
                    times[n][function_to_exec] = filtering_times
                else:
                    start_t = time.time()
                    output = eval(function_to_exec)(log_path, path_scenario, execution)
                    times[n][function_to_exec] = {"duration": float(time.time()) - float(start_t)}

            n += 1
        
    return times

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

def case_study_generator_execution(user_id: int, case_study_id: int):
    """
    This function process input data and generates the execution for a case study. It executes all phases specified in 'to_exec' and it stores enriched log and decision tree extracted from the initial UI log in the same folder it is.

    Args:
        user_id (int): The user id of the user that is executing the case study
        case_study_id (int): The case study id of the case study to be executed
    """
    with transaction.atomic():
        execution = Execution(user=User.objects.get(id=user_id), case_study=CaseStudy.objects.get(id=case_study_id))
        execution.save()
        execution.check_preloaded_file()

        times = {}

        # year = datetime.now().date().strftime("%Y")
        tprint("RPA-US     SCREEN RPA", "tarty1")
        # tprint("Relevance Information Miner", "pepper")
        if execution:
            if len(execution.scenarios_to_study) > 0:
                aux_path = os.path.join(execution.exp_folder_complete_path, execution.scenarios_to_study[0])
            else:
                aux_path = execution.exp_folder_complete_path
            # if not os.path.exists(aux_path):
            #     os.makedirs(aux_path)
        else:
            aux_path = execution.exp_folder_complete_path
        
        # For BPM LOG GENERATOR (old AGOSUIRPA) files
        foldername_logs_with_different_size_balance = get_foldernames_as_list(aux_path)
        
        for scenario in tqdm(execution.scenarios_to_study, desc=_("Scenarios that have been processed: ")):
            # For BPM LOG GENERATOR (old AGOSUIRPA) files
            if SCENARIO_NESTED_FOLDER:
                path_scenario = os.path.join(execution.exp_folder_complete_path, scenario, n)
                for n in foldername_logs_with_different_size_balance:
                    generate_case_study(execution, path_scenario, times)
            else:
                path_scenario = os.path.join(execution.exp_folder_complete_path, scenario)
                generate_case_study(execution, path_scenario, times)
            execution.executed = (execution.scenarios_to_study.index(scenario) / len(execution.scenarios_to_study)) * 100
            execution.save()

        # Serializing json
        json_object = json.dumps(times, indent=4)
        # Writing to .json
        
        metadata_final_path = os.path.join(
            execution.exp_folder_complete_path,
            f"times-cs_{execution.case_study.id}-exec_{execution.id}-metainfo.json"
            )

        with open(metadata_final_path, "w") as outfile:
            outfile.write(json_object)
            
        print(f"Case study {execution.case_study.title} executed!!. Case study foldername: {execution.exp_foldername}.Metadata saved in: {metadata_final_path}")

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

def case_study_generator(data):
    transaction_works = False

    with transaction.atomic():

        # Introduce a default value for scencarios_to_study if there is none
        if not data['scenarios_to_study']:
            data['scenarios_to_study'] = get_foldernames_as_list(data['exp_folder_complete_path'])

        phases = data["phases_to_execute"].copy()
        cs_serializer = CaseStudySerializer(data=data)
        cs_serializer.is_valid(raise_exception=True)
        case_study = cs_serializer.save()

        # For each phase we want to execute, we create a database row for it and relate it with the case study
        for phase in phases:
            match phase:
                case "monitoring":
                    serializer = MonitoringSerializer(data=phases[phase])
                    # Guarda el objeto de Monitoring de serializer, y asigna el objeto de case_study al campo case_study de Monitoring y el user a user de Monitoring
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "prefilters":
                    serializer = PrefiltersSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "ui_elements_detection":
                    serializer = UIElementsDetectionSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "ui_elements_classification":
                    serializer = UIElementsClassificationSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "postfilters":
                    serializer = PostfiltersSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "feature_extraction_technique":
                    serializer = FeatureExtractionTechniqueSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.validated_data['type'] = "SINGLE"
                    serializer.save()
                case "process_discovery":
                    serializer = ProcessDiscoverySerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "extract_training_dataset":
                    serializer = ExtractTrainingDatasetSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case "aggregate_features_as_dataset_columns":
                    serializer = FeatureExtractionTechniqueSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.validated_data['type'] = "AGGREGATE"
                    serializer.save()
                case "decision_tree_training":
                    serializer = DecisionTreeTrainingSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    serializer.validated_data['case_study'] = case_study
                    serializer.validated_data['user'] = case_study.user
                    serializer.save()
                case _:
                    pass

        # Updating the case study with the foreign keys of the phases to execute
        # case_study.save()
        transaction_works = True
    with transaction.atomic():
        execution = Execution(user=case_study.user, case_study=case_study)
        execution.save()

        if ACTIVE_CELERY:
            celery_task_process_case_study.delay(case_study.user.id, case_study.id)
        else:
            case_study_generator_execution(case_study.user.id, case_study.id)
        
    return transaction_works, case_study

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
# Views themself
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

class CaseStudyCreateView(CreateView):
    model = CaseStudy
    form_class = CaseStudyForm
    template_name = "case_studies/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class CaseStudyListView(ListView, LoginRequiredMixin):
    model = CaseStudy
    template_name = "case_studies/list.html"
    paginate_by = 50

    def get_queryset(self):
        return CaseStudy.objects.filter(active=True, user=self.request.user).order_by("-created_at")


def executeCaseStudy(request):
    case_study_id = request.GET.get("id")
    cs = CaseStudy.objects.get(id=case_study_id)
    if request.user.id != cs.user.id:
        raise Exception(_("This case study doesn't belong to the authenticated user"))
    elif ACTIVE_CELERY:
        celery_task_process_case_study.delay(request.user.id, case_study_id)
    else:
        threading.Thread(target=case_study_generator_execution, args=(request.user.id, case_study_id,)).start()

    # Return a response immediately, without waiting for the execution to finish
    return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
def deleteCaseStudy(request):
    case_study_id = request.GET.get("id")
    cs = CaseStudy.objects.get(id=case_study_id)
    if request.user.id != cs.user.id:
        raise Exception(_("This case study doesn't belong to the authenticated user"))
    if cs.executed != 0:
        raise Exception(_("This case study cannot be deleted because it has already been excecuted"))
    cs.delete()
    return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
class CaseStudyDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        case_study = get_object_or_404(CaseStudy, id=kwargs["case_study_id"], active=True)
        context = {
            "case_study": case_study, 
            "single_fe": FeatureExtractionTechnique.objects.filter(case_study=case_study, type="SINGLE"), 
            "aggregate_fe": FeatureExtractionTechnique.objects.filter(case_study=case_study, type="AGGREGATE")
            }
        return render(request, "case_studies/detail.html", context)

class CaseStudyView(generics.ListCreateAPIView):
    # permission_classes = [IsAuthenticatedUser]
    serializer_class = CaseStudySerializer

    def get_queryset(self):
        return CaseStudy.objects.filter(user=self.request.user)

    def post(self, request, *args, **kwargs):

        #Before starting the async task we will check if the json fields values are valid
        case_study_serialized = CaseStudySerializer(data=request.data)
        st = status.HTTP_200_OK

        if not case_study_serialized.is_valid():
            response_content = case_study_serialized.errors
            st=status.HTTP_400_BAD_REQUEST
        else:
            # asignar el usuario al caso de estudio
            case_study_serialized.validated_data['user'] = request.user
            execute_case_study = True
            try:
                if not isinstance(case_study_serialized.data['phases_to_execute'], dict):
                    response_content = {"message": _("phases_to_execute must be of type dict!!!!! and must be composed by phases contained in %(DEFAULT_PHASES)s")} % {"DEFAULT_PHASES": DEFAULT_PHASES}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                if hasattr(case_study_serialized.data['phases_to_execute'], 'ui_elements_detection') and (not case_study_serialized.data['phases_to_execute']['ui_elements_detection']['type'] in ["legacy", "uied"]):
                    response_content = {"message": _("Elements Detection algorithm must be one of ['legacy', 'uied']")}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                if hasattr(case_study_serialized.data['phases_to_execute'], 'ui_elements_classification'):
                    if (not case_study_serialized.data['phases_to_execute']['ui_elements_classification']['type'] in ["legacy", "uied"]):
                        response_content = {"message": _("Elements Classification algorithm must be one of ['legacy', 'uied']")}
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                        execute_case_study = False
                        return Response(response_content, st)

                    for path in [case_study_serialized.data['phases_to_execute']['ui_elements_classification']['model'],
                             case_study_serialized.data['phases_to_execute']['ui_elements_classification']['model_properties']]:
                        if not os.path.exists(path):
                            response_content = {"message": _("The following file or directory does not exists: %(path)") % {"path": path}}
                            st = status.HTTP_422_UNPROCESSABLE_ENTITY
                            execute_case_study = False
                            return Response(response_content, st)

                if not os.path.exists(case_study_serialized.data['exp_folder_complete_path']):
                    response_content = {"message": _("The following file or directory does not exists: %(path)") % {"path": case_study_serialized.data['exp_folder_complete_path']}}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                for phase in dict(case_study_serialized.data['phases_to_execute']).keys():
                    if not(phase in DEFAULT_PHASES):
                        response_content = {"message": _("phases_to_execute must be composed by phases contained in %(DEFAULT_PHASES)s") % {"DEFAULT_PHASES": DEFAULT_PHASES}}
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                        execute_case_study = False
                        return Response(response_content, st)

                

                if execute_case_study:
                    # init_case_study_task.delay(request.data)
                    transaction_works, case_study = case_study_generator(case_study_serialized.data)
                    if not transaction_works:
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    else:    
                        response_content = {"message": _("Case study with id:%(id) is being generated ...") % {"id": case_study.id}}

            except Exception as e:
                response_content = {"message": _("Some of the attributes are invalid: ") + str(e) }
                st = status.HTTP_422_UNPROCESSABLE_ENTITY

        # # item = CaseStudy.objects.create(serializer)
        # # result = CaseStudySerializer(item)
        # # return Response(result.data, status=status.HTTP_201_CREATED)

        return Response(response_content, status=st)

class SpecificCaseStudyView(generics.ListCreateAPIView):
    def get(self, request, case_study_id, *args, **kwargs):
        st = status.HTTP_200_OK
        try:
            case_study = CaseStudy.objects.get(id=case_study_id)
            serializer = CaseStudySerializer(instance=case_study)
            response = serializer.data
            return Response(response, status=st)

        except Exception as e:
            response = {_("Case Study with id %(id) not found") % {"id": case_study_id}}
            st = status.HTTP_404_NOT_FOUND

        return Response(response, status=st)

class ResultCaseStudyView(generics.ListCreateAPIView):
    def get(self, request, execution_id, *args, **kwargs):
        st = status.HTTP_200_OK
        
        try:
            execution = Execution.objects.get(id=execution)
            if execution.executed:
                csv_data, csv_filename = experiments_results_collectors(execution, "descision_tree.log")
                response = HttpResponse(content_type="text/csv")
                response["Content-Disposition"] = 'attachment; filename="'+execution.case_study.title+'.csv"'
                csv_data.to_csv(response, index=False)
                return response
            else:
                response = {"message": _('The processing of this case study has not yet finished, please try again in a few minutes')}

        except Exception as e:
            response = {"message": _("Case Study with id %(id) raised an exception: ") % {"id": execution.case_study.id} + str(e)}
            st = status.HTTP_404_NOT_FOUND

        return Response(response, status=st)
    
# Home

@login_required(login_url="/login/")
def index(request):
    return HttpResponseRedirect(reverse("analyzer:casestudy_list"))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        # if ".html" not in load_template:
        #     load_template += ".html"
        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def exp_files(request):
    user = request.user
    case_studies = CaseStudy.objects.filter(user=user)
    exp_files = [(c.id, c.exp_file.name, c.exp_file.path, c.exp_file.size) for c in case_studies]
    return render(request, 'case_studies/exp_files.html', {'object_list': exp_files})

@login_required(login_url="/login/")
def exp_file_download(request, case_study_id):
    user = request.user
    cs = CaseStudy.objects.filter(user=user, id=case_study_id)
    if cs.exists():
        unzipped_folder = cs[0].exp_folder_complete_path
    else:
        raise Exception(_("You don't have permissions to access this files"))
    
    # Create a temporary zip file containing the contents of the unzipped folder
    zip_filename = os.path.basename(unzipped_folder) + ".zip"
    zip_file_path = os.path.join(PRIVATE_STORAGE_ROOT, zip_filename)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, dirs, files in os.walk(unzipped_folder):
            # Ignore executions folder
            if "executions" in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, unzipped_folder)
                zip_ref.write(file_path, arcname=rel_path)
    # Serve the zip file as a download response
    response = FileResponse(open(zip_file_path, "rb"), content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename="%s"' % zip_filename
    response['Access-Control-Expose-Headers'] = 'Content-Disposition'
    
    return response



# ============================================================================================================================
# Executions
# ============================================================================================================================
class ExecutionListView(ListView, LoginRequiredMixin):
    model = Execution
    template_name = "executions/list.html"
    paginate_by = 50

    def get_queryset(self):
        return Execution.objects.filter(user=self.request.user).order_by("-created_at")
        

  
def deleteExecution(request):
    execution_id = request.GET.get("id")
    cs = Execution.objects.get(id=execution_id)
    if request.user.id != cs.user.id:
        raise Exception(_("This case study doesn't belong to the authenticated user"))
    if cs.executed != 0:
        raise Exception(_("This case study cannot be deleted because it has already been excecuted"))
    cs.delete()
    return HttpResponseRedirect(reverse("analyzer:execution_list"))
    
class ExecutionDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])
        context = {
            "execution": execution, 
            "single_fe": FeatureExtractionTechnique.objects.filter(execution=execution, type="SINGLE"), 
            "aggregate_fe": FeatureExtractionTechnique.objects.filter(execution=execution, type="AGGREGATE")
            }
        return render(request, "executions/detail.html", context)

@login_required(login_url="/login/")
def exec_file_download(request, execution_id):
    user = request.user
    execution = Execution.objects.filter(user=user, id=execution_id)
    if execution.exists():
        # Build zip file from the execution folder in exp_folder_complete_path
        unzipped_folder = execution[0].exp_folder_complete_path
    else:
        raise Exception(_("You don't have permissions to access this files"))
    
    # Create a temporary zip file containing the contents of the unzipped folder
    zip_filename = os.path.basename(unzipped_folder) + ".zip"
    zip_file_path = os.path.join(PRIVATE_STORAGE_ROOT, zip_filename)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, dirs, files in os.walk(unzipped_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, unzipped_folder)
                zip_ref.write(file_path, arcname=rel_path)
    # Serve the zip file as a download response
    response = FileResponse(open(zip_file_path, "rb"), content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename="%s"' % zip_filename
    response['Access-Control-Expose-Headers'] = 'Content-Disposition'
    
    return response

#################################################################### PHASE EXECUTIONS RESULTS ####################################################################
    
class MonitoringResultDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        # Get the Execution object or raise a 404 error if not found
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        scenarioNumber = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenarioNumber == None:
            #scenarioNumber = "1"
            scenarioNumber = execution.scenarios_to_study[0] # by default, the first one that was indicated
            
        #path_to_csv_file = execution.exp_folder_complete_path + "/"+ scenarioNumber +"/log.csv"  
        path_to_csv_file = os.path.join(execution.exp_folder_complete_path, scenarioNumber, "log.csv")
        # CSV Download
        if path_to_csv_file and download=="True":
            return MonitoringResultDownload2(path_to_csv_file)  

        # CSV Reading and Conversion to JSON
        csv_data_json = read_csv_to_json(path_to_csv_file)

        # Include CSV data in the context for the template
        context = {
            "execution": execution,
            "csv_data": csv_data_json,  # Data to be used in the HTML template
            "scenarios": execution.scenarios_to_study,
            "scenarioNumber": scenarioNumber
            } 

        # Render the HTML template with the context including the CSV data
        return render(request, "monitoring/result.html", context)


##########################################
    
class FeatureExtractionResultDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        # Get the Execution object or raise a 404 error if not found
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        scenarioNumber = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenarioNumber == None:
            #scenarioNumber = "1"
            scenarioNumber = execution.scenarios_to_study[0] # by default, the first one that was indicated
      
        # TODO: Sujeto a cambios en la estructura de la carpeta
        #path_to_csv_file = execution.exp_folder_complete_path + "/"+ scenarioNumber +"/log.csv" #enriched_log.csv
        path_to_csv_file = os.path.join(execution.exp_folder_complete_path, scenarioNumber, "log.csv")
        # CSV Download
        if path_to_csv_file and download=="True":
            return MonitoringResultDownload2(path_to_csv_file)  
     
        # CSV Reading and Conversion to JSON
        csv_data_json = read_csv_to_json(path_to_csv_file)

        # Include CSV data in the context for the template
        context = {
            "execution": execution,
            "csv_data": csv_data_json,  # Data to be used in the HTML template
            "scenarios": execution.scenarios_to_study,
            "scenarioNumber": scenarioNumber
            }

        # Render the HTML template with the context including the CSV data
        return render(request, "feature_extraction_technique/result.html", context)

#########################################

    
class ProcessDiscoveryResultDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        # Get the Execution object or raise a 404 error if not found 
        # Render the HTML template with the context including the CSV data
        return render(request, "processdiscovery/result.html")
    
# class ProcessDiscoveryResultDetailView(DetailView):
#     def get(self, request, *args, **kwargs):
#         # Get the Execution object or raise a 404 error if not found
#         execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
#         scenarioNumber = request.GET.get('scenario')
#         download = request.GET.get('download')

#         if scenarioNumber == None:
#             #scenarioNumber = "1"
#             scenarioNumber = execution.scenarios_to_study[0] # by default, the first one that was indicated
      
#         path_to_csv_file = execution.exp_folder_complete_path + "/"+ scenarioNumber +"/log.csv" 

#         # CSV Download
#         if path_to_csv_file and download=="True":
#             return MonitoringResultDownload2(path_to_csv_file) 

#         # CSV Reading and Conversion to JSON
#         csv_data_json = read_csv_to_json(path_to_csv_file)

#         # Include CSV data in the context for the template
#         context = {
 
#########################################

class ExtractTrainingDatasetResultDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        # Get the Execution object or raise a 404 error if not found
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        scenarioNumber = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenarioNumber == None:
            #scenarioNumber = "1"
            scenarioNumber = execution.scenarios_to_study[0] # by default, the first one that was indicated
      
        #path_to_csv_file = execution.exp_folder_complete_path + "/"+ scenarioNumber +"/log.csv" #flattened_log.csv
        path_to_csv_file = os.path.join(execution.exp_folder_complete_path, scenarioNumber, "log.csv")
        # CSV Download
        if path_to_csv_file and download=="True":
            return MonitoringResultDownload2(path_to_csv_file) 

        # CSV Reading and Conversion to JSON
        csv_data_json = read_csv_to_json(path_to_csv_file)

        # Include CSV data in the context for the template
        context = {
            "execution": execution,
            "csv_data": csv_data_json,  # Data to be used in the HTML template
            "scenarios": execution.scenarios_to_study,
            "scenarioNumber": scenarioNumber
            }  

        # Render the HTML template with the context including the CSV data
        return render(request, "extract_training_dataset/result.html", context)

##############################################33
    

# def LogicPhasesResultDetailView(execution, scenarioNumber,path_to_csv_file):
   
#     # CSV Reading and Conversion to JSON
#     csv_data_json = read_csv_to_json(path_to_csv_file)

#     # Include CSV data in the context for the template
#     context = {
#             "execution": execution,
#             "csv_data": csv_data_json,  # Data to be used in the HTML template
#             "scenarios": execution.scenarios_to_study,
#             "scenarioNumber": scenarioNumber
#             }  
#         # Render the HTML template with the context including the CSV data
#         return render(request, "processdiscovery/result.html", context)
#         }
#     return context

#############################################33
def read_csv_to_json(path_to_csv_file):
    # Initialize a list to hold the CSV data converted into dictionaries
    csv_data = []       
    # Check if the path to the CSV file exists and read the data
    try:
        with open(path_to_csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                csv_data.append(row)
    except FileNotFoundError:
        print(f"File not found: {path_to_csv_file}")
    # Convert csv_data to JSON
    csv_data_json = json.dumps(csv_data)
    return csv_data_json
##########################################3
def MonitoringResultDownload2(path_to_csv_file):
    with open(path_to_csv_file, 'r', newline='') as csvfile:
        # Create an HTTP response with the content of the CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'inline; filename="{}"'.format(os.path.basename(path_to_csv_file))
        writer = csv.writer(response)
        reader = csv.reader(csvfile)
        for row in reader:
            writer.writerow(row)
        return response
    

#############################################################
    
class UIElementsDetectionResultDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        execution: Execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        scenario: str = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenario == None:
            scenario = execution.scenarios_to_study[0] # Select the first scenario by default

        # Create dictionary with images and their corresponding UI elements
        soms = dict()

        classes = execution.ui_elements_classification.model.classes
        colors = []
        for i in range(len(classes)):
            colors.append("#%06x" % random.randint(0, 0xFFFFFF))
        soms["classes"] = {k: v for k, v in zip(classes, colors)} 

        soms["soms"] = []

        for compo_json in os.listdir(os.path.join(execution.exp_folder_complete_path, scenario + "_results", "components_json")):
            with open(os.path.join(execution.exp_folder_complete_path, scenario + "_results", "components_json", compo_json), "r") as f:
                compos = json.load(f)
            # path is something like: asdsa/.../.../image.PNG.json
            img_name = compo_json.split("/")[-1].split(".json")[0]
            img_path = os.path.join(execution.case_study.exp_foldername, scenario, img_name)

            soms["soms"].append(
                {
                    "img": img_name,
                    "img_path": img_path,
                    "som": compos
                }
            )

        context = {
            "execution": execution,
            "scenarios": execution.scenarios_to_study,
            "soms": soms
        }

        #return HttpResponse(json.dumps(context), content_type="application/json")
        return render(request, "ui_elements_detection/results.html", context)



