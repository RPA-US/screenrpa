from multiprocessing.connection import wait
from celery import shared_task
from django.shortcuts import render, get_object_or_404

# Create your views here.
import os
import json
import time
from django.core.exceptions import ValidationError
from art import tprint
from tqdm import tqdm
from django.contrib.auth.models import User 
import time
from core.settings import times_calculation_mode, metadata_location, sep, decision_foldername, gui_quantity_difference, default_phases, scenario_nested_folder, FLATTENED_DATASET_NAME
from apps.decisiondiscovery.views import decision_tree_training, extract_training_dataset
from apps.featureextraction.views import ui_elements_classification, feature_extraction_technique
from apps.featureextraction.detection import ui_elements_detection
from apps.featureextraction.gaze_analysis import noise_filtering
# CaseStudyView
from rest_framework import generics, status, viewsets #, permissions
from rest_framework.response import Response
from apps.analyzer.forms import CaseStudyForm
from apps.analyzer.tasks import init_generate_case_study
# from rest_framework.pagination import PageNumberPagination
from .models import CaseStudy# , ExecutionManager
from .serializers import CaseStudySerializer
from apps.featureextraction.serializers import UIElementsClassificationSerializer, UIElementsDetectionSerializer, FeatureExtractionTechniqueSerializer, NoiseFilteringSerializer
from apps.decisiondiscovery.serializers import DecisionTreeTrainingSerializer, ExtractTrainingDatasetSerializer
from apps.featureextraction.models import UIElementsClassification, UIElementsDetection, FeatureExtractionTechnique, NoiseFiltering
from apps.decisiondiscovery.models import DecisionTreeTraining, ExtractTrainingDataset
from .collect_results import experiments_results_collectors
from apps.analyzer.utils import get_foldernames_as_list
from django.db import transaction
from django.contrib.auth import get_user_model
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import ListView, DetailView, CreateView, FormView, DeleteView
from django.contrib.auth.decorators import login_required

def generate_case_study(case_study, path_scenario, times, n):
    times[n] = {}
    to_exec_args = {
        'ui_elements_detection': (path_scenario + 'log.csv',
                                        path_scenario,
                                        case_study.special_colnames,
                                        case_study.ui_elements_detection.skip,
                                        case_study.ui_elements_detection.type,
                                        case_study.text_classname)
                                        # We check this phase is present in case_study to avoid exceptions
                                        if case_study.ui_elements_detection else None,
        'noise_filtering': (path_scenario +'log.csv',
                                        path_scenario,
                                        case_study.special_colnames,
                                        case_study.noise_filtering.type,
                                        case_study.noise_filtering.configurations)
                                        # We check this phase is present in case_study to avoid exceptions
                                        if case_study.noise_filtering else None,
        'ui_elements_classification': (case_study.ui_elements_classification.model,
                                        case_study.ui_elements_classification.model_properties,
                                        path_scenario + 'components_npy' + sep,
                                        path_scenario + 'components_json' + sep,
                                        path_scenario + 'log.csv',
                                        case_study.special_colnames["Screenshot"],
                                        case_study.text_classname,
                                        case_study.ui_elements_classification.skip,
                                        case_study.ui_elements_classification_classes,
                                        case_study.ui_elements_classification_image_shape,
                                        case_study.ui_elements_classification.type)
                                        # We check this phase is present in case_study to avoid exceptions
                                        if case_study.ui_elements_classification else None,
        'extract_training_dataset': (case_study.decision_point_activity, 
                                        case_study.target_label,
                                        case_study.special_colnames,
                                        case_study.extract_training_dataset.columns_to_drop,
                                        path_scenario + 'log.csv',
                                        path_scenario, 
                                        case_study.extract_training_dataset.columns_to_drop_before_decision_point,
                                    )
                                    # We check this phase is present in case_study to avoid exceptions
                                    if case_study.extract_training_dataset else None,
        'feature_extraction_technique': (case_study.ui_elements_classification_classes,
                                        case_study.decision_point_activity,
                                        case_study.special_colnames["Case"],
                                        case_study.special_colnames["Activity"],
                                        case_study.special_colnames["Screenshot"],
                                        path_scenario + 'components_json' + sep,
                                        path_scenario + 'flattened_dataset.json',
                                        path_scenario + 'log.csv',
                                        path_scenario + case_study.feature_extraction_technique.technique_name+'_enriched_log.csv',
                                        case_study.text_classname,
                                        case_study.feature_extraction_technique.identifier,
                                        case_study.feature_extraction_technique.skip,
                                        case_study.feature_extraction_technique.technique_name)
                                        # We check this phase is present in case_study to avoid exceptions
                                        if case_study.feature_extraction_technique else None,
        'decision_tree_training': (path_scenario + 'flattened_dataset.json', 
                                    path_scenario,
                                    case_study.decision_tree_training.library,
                                    case_study.decision_tree_training.algorithms,
                                    case_study.decision_tree_training.columns_to_drop_before_decision_point,
                                    case_study.target_label,
                                    case_study.decision_tree_training.one_hot_columns)
                                    # We check this phase is present in case_study to avoid exceptions
                                    if case_study.decision_tree_training  else None
        }

    # We go over the keys of to_exec_args, and call the corresponding functions passing the corresponding parameters
    for function_to_exec in [key for key in to_exec_args.keys() if to_exec_args[key] is not None]:
        if function_to_exec == "decision_tree_training" and case_study.decision_tree_training.library !='sklearn':
            res, tree_times, columns_len = eval(function_to_exec)(*to_exec_args[function_to_exec])
            times[n][function_to_exec] = tree_times
            times[n][function_to_exec]["columns_len"] = columns_len
            # times[n][function_to_exec]["tree_levels"] = tree_levels
            times[n][function_to_exec]["accuracy"] = res
        elif function_to_exec == "feature_extraction_technique":
            start_t = time.time()
            num_UI_elements, num_screenshots, max_ui_elements, min_ui_elements = eval(function_to_exec)(*to_exec_args[function_to_exec])
            times[n][function_to_exec] = {"duration": float(time.time()) - float(start_t)}
            times[n][function_to_exec]["num_UI_elements"] = num_UI_elements
            times[n][function_to_exec]["num_screenshots"] = num_screenshots
            times[n][function_to_exec]["max_#UI_elements"] = max_ui_elements
            times[n][function_to_exec]["min_#UI_elements"] = min_ui_elements
        else:
            start_t = time.time()
            output = eval(function_to_exec)(*to_exec_args[function_to_exec])
            times[n][function_to_exec] = {"duration": float(time.time()) - float(start_t)}

        # TODO: accurracy_score
        # if index == len(to_exec)-1:
        #     times[n][index]["decision_model_accuracy"] = output
        
    return times

def case_study_process_data(case_study_id):
    """
    This function process input data and generates the case study. It executes all phases specified in 'to_exec' and it stores enriched log and decision tree extracted from the initial UI log in the same folder it is.

    Args:
        exp_foldername (string): name of the folder where all case study data is stored. Example 'case_study_data'
        exp_folder_complete_path (string): complete path to the folder where all case study data is stored, including the name of the folder in this path. Example 'C:\\John\\Desktop\\case_study_data'
        decision_activity (string): activity where decision we want to study is taken. Example: 'B'
        scenarios (list): list with all foldernames corresponding to the differents scenarios that will be studied in this case study
        special_colnames (dict): a dict with the keys "Case", "Activity", "Screenshot", "Variant", "Timestamp", "eyetracking_recording_timestamp", "eyetracking_gaze_point_x", "eyetracking_gaze_point_y", specifiyng as their values each column name associated of your UI log.
        to_exec (list): list of the phases we want to execute. The possible phases to include are configured in settings.py: default_phases
    """
    case_study = CaseStudy.objects.get(id=case_study_id)
    times = {}
    metadata_path = metadata_location + sep # folder to store metadata that will be used in "results" mode

    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
 

    # year = datetime.now().date().strftime("%Y")
    tprint("RPA-US     RIM", "tarty1")
    # tprint("Relevance Information Miner", "pepper")

    foldername_logs_with_different_size_balance = get_foldernames_as_list(case_study.exp_folder_complete_path + sep + case_study.scenarios_to_study[0], sep)
    
    for scenario in tqdm(case_study.scenarios_to_study, desc="Scenarios that have been processed: "):
        time.sleep(.1)
        print("\nActual Scenario: " + str(scenario))
        # We check there is at least 1 phase to execute
        if case_study.ui_elements_detection or case_study.ui_elements_classification or case_study.feature_extraction_technique or case_study.noise_filtering or case_study.extract_training_dataset or case_study.decision_tree_training:
            if scenario_nested_folder == "TRUE":
                path_scenario = case_study.exp_folder_complete_path + sep + scenario + sep + n + sep 
                for n in foldername_logs_with_different_size_balance:
                    generate_case_study(case_study, path_scenario, times, n)
            else:
                path_scenario = case_study.exp_folder_complete_path + sep + scenario + sep
                generate_case_study(case_study, path_scenario, times, scenario)
                

    # Serializing json
    json_object = json.dumps(times, indent=4)
    # Writing to .json

    case_study.executed = 100
    case_study.save()
    
    metadata_final_path = metadata_path + str(case_study.id) + "-metainfo.json"
    with open(metadata_final_path, "w") as outfile:
        outfile.write(json_object)
        
    return "Case study '"+case_study.title+"' executed!!. Case study foldername: "+case_study.exp_foldername+". Metadata saved in: "+metadata_final_path
     
    # each experiment one csv line
    # store exceution times per each phase and experiment (30 per family)
    # execute only the experiments

# ========================================================================
# RUN CASE STUDY
# ========================================================================

def case_study_generator(data):
    '''
    Mandatory Attributes: title, exp_foldername, phases_to_execute, decision_point_activity, exp_folder_complete_path, gui_class_success_regex, gui_quantity_difference, scenarios_to_study, drop, special_colnames
    Example values:
    title = "case study 1"
    decision_point_activity = "D"
    path_to_save_experiment = None
    gui_class_success_regex = "CheckBox_D or ImageView_D or TextView_D" # "(CheckBox_D or ImageView_D or TextView_D) and (ImageView_B or TextView_B)"
    gui_quantity_difference = 1
    drop = None  # Example: ["Advanced_10_Balanced", "Advanced_10_Imbalanced"]
    interactive = False
    phases_to_execute = {'ui_elements_detection': {},
                   'ui_elements_classification': {},
                   'extract_training_dataset': {},
                   'decision_tree_training': {}
                   }
    scenarios = None # ["scenario_10","scenario_11","scenario_12","scenario_13"]
    '''
    transaction_works = False

    with transaction.atomic():

        # Introduce a default value for scencarios_to_study if there is none
        if not data['scenarios_to_study']:
            data['scenarios_to_study'] = get_foldernames_as_list(data['exp_folder_complete_path'], sep)

        # if "ui_elements_detection_id" in data  and data["ui_elements_detection_id"] :
        #     aux["phases_to_execute"]["ui_elements_detection"] = cs.ui_elements_detection.__dict__
        # if "noise_filtering_id" in data and data["noise_filtering_id"]:
        #     aux["phases_to_execute"]["noise_filtering"] = cs.noise_filtering.__dict__
        # if "feature_extraction_technique_id" in data and data["feature_extraction_technique_id"]:
        #     aux["phases_to_execute"]["feature_extraction_technique"] = cs.feature_extraction_technique.__dict__
        # if "extract_training_dataset_id" in data and data["extract_training_dataset_id"]:
        #     aux["phases_to_execute"]["extract_training_dataset"] = cs.extract_training_dataset.__dict__
        # if "decision_tree_training_id" in data and data["decision_tree_training_id"]:
        #     aux["phases_to_execute"]["decision_tree_training"] = cs.decision_tree_training.__dict__


        phases = data["phases_to_execute"].copy()
        cs_serializer = CaseStudySerializer(data=data)
        cs_serializer.is_valid(raise_exception=True)
        case_study = cs_serializer.save()

        # For each phase we want to execute, we create a database row for it and relate it with the case study
        for phase in phases:
            match phase:
                case "ui_elements_detection":
                    serializer = UIElementsDetectionSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.ui_elements_detection = serializer.save()
                case "noise_filtering":
                    serializer = NoiseFilteringSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.noise_filtering = serializer.save()
                case "ui_elements_classification":
                    serializer = UIElementsClassificationSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.ui_elements_classification = serializer.save()
                case "feature_extraction_technique":
                    serializer = FeatureExtractionTechniqueSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.feature_extraction_technique = serializer.save()
                case "extract_training_dataset":
                    serializer = ExtractTrainingDatasetSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.extract_training_dataset = serializer.save()
                case "decision_tree_training":
                    serializer = DecisionTreeTrainingSerializer(data=phases[phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.decision_tree_training = serializer.save()
                case _:
                    pass

        # Updating the case study with the foreign keys of the phases to execute
        case_study.save()
        transaction_works = True
    init_generate_case_study.delay(case_study.id)

    return transaction_works, case_study

class CaseStudyCreateView(CreateView):
    model = CaseStudy
    form_class = CaseStudyForm
    template_name = "case_studies/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class CaseStudyListView(ListView):
    model = CaseStudy
    template_name = "case_studies/list.html"
    paginate_by = 50

    def get_queryset(self):
        # return CaseStudy.objects.filter(active=True)
        return CaseStudy.objects.all()

class ExecuteCaseStudyView(ListView):
    model = CaseStudy
    template_name = "case_studies/list.html"
    paginate_by = 50

    def get_queryset(self):
        cs = CaseStudy.objects.get(id=self.kwargs["case_study_id"])
        aux = cs.__dict__
        aux["user"] = cs.user.id
        case_study_generator(aux)
        return CaseStudy.objects.all()
    
class CaseStudyDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        case_study = get_object_or_404(CaseStudy, id=kwargs["case_study_id"], active=True)
        context = {"case_study": case_study}
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
            execute_case_study = True
            try:
                if not isinstance(case_study_serialized.data['phases_to_execute'], dict):
                    response_content = {"message": f"phases_to_execute must be of type dict!!!!! and must be composed by phases contained in {default_phases}"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                if hasattr(case_study_serialized.data['phases_to_execute'], 'ui_elements_detection') and (not case_study_serialized.data['phases_to_execute']['ui_elements_detection']['type'] in ["legacy", "uied"]):
                    response_content = {"message": "Elements Detection algorithm must be one of ['legacy', 'uied']"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                if hasattr(case_study_serialized.data['phases_to_execute'], 'ui_elements_classification'):
                    if (not case_study_serialized.data['phases_to_execute']['ui_elements_classification']['type'] in ["legacy", "uied"]):
                        response_content = {"message": "Elements Classification algorithm must be one of ['legacy', 'uied']"}
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                        execute_case_study = False
                        return Response(response_content, st)

                    for path in [case_study_serialized.data['phases_to_execute']['ui_elements_classification']['model'],
                             case_study_serialized.data['phases_to_execute']['ui_elements_classification']['model_properties']]:
                        if not os.path.exists(path):
                            response_content = {"message": f"The following file or directory does not exists: {path}"}
                            st = status.HTTP_422_UNPROCESSABLE_ENTITY
                            execute_case_study = False
                            return Response(response_content, st)

                if not os.path.exists(case_study_serialized.data['exp_folder_complete_path']):
                    response_content = {"message": f"The following file or directory does not exists: {case_study_serialized.data['exp_folder_complete_path']}"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                for phase in dict(case_study_serialized.data['phases_to_execute']).keys():
                    if not(phase in default_phases):
                        response_content = {"message": f"phases_to_execute must be composed by phases contained in {default_phases}"}
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                        execute_case_study = False
                        return Response(response_content, st)

                

                if execute_case_study:
                    # init_case_study_task.delay(request.data)
                    transaction_works, case_study = case_study_generator(case_study_serialized.data)
                    if not transaction_works:
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    else:    
                        response_content = {"message": f"Case study with id:{case_study.id} is being generated ..."}

            except Exception as e:
                response_content = {"message": "Some of the attributes are invalid: " + str(e) }
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
            response = {f"Case Study with id {case_study_id} not found"}
            st = status.HTTP_404_NOT_FOUND

        return Response(response, status=st)

class ResultCaseStudyView(generics.ListCreateAPIView):
    def get(self, request, case_study_id, *args, **kwargs):
        st = status.HTTP_200_OK
        
        try:
            case_study = CaseStudy.objects.get(id=case_study_id)
            if case_study.executed:
                csv_data, csv_filename = experiments_results_collectors(case_study, "descision_tree.log")
                response = HttpResponse(content_type="text/csv")
                response["Content-Disposition"] = 'attachment; filename="'+case_study.title+'.csv"'
                csv_data.to_csv(response, index=False)
                return response
            else:
                response = {"message": 'The processing of this case study has not yet finished, please try again in a few minutes'}

        except Exception as e:
            response = {"message": f"Case Study with id {case_study_id} raise an exception: " + str(e)}
            st = status.HTTP_404_NOT_FOUND

        return Response(response, status=st)