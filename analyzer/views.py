from multiprocessing.connection import wait
from celery import shared_task
from django.shortcuts import render

# Create your views here.
import os
import json
import time
import csv
import pandas as pd
import re
from art import tprint
from tqdm import tqdm
from time import sleep
from datetime import datetime
from rim.settings import times_calculation_mode, metadata_location, sep, decision_foldername, gui_quantity_difference
from decisiondiscovery.views import decision_tree_training, extract_training_dataset
from featureextraction.views import ui_elements_classification, feature_extraction
from featureextraction.detection import ui_elements_detection
# CaseStudyView
from rest_framework import generics, status, viewsets #, permissions
from rest_framework.response import Response
from analyzer.tasks import init_generate_case_study
# from rest_framework.pagination import PageNumberPagination
from .models import CaseStudy# , ExecutionManager
from .serializers import CaseStudySerializer
from featureextraction.serializers import UIElementsClassificationSerializer, UIElementsDetectionSerializer, FeatureExtractionTechniqueSerializer, NoiseFilteringSerializer
from decisiondiscovery.serializers import DecisionTreeTrainingSerializer, ExtractTrainingDatasetSerializer
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db import transaction
from django.forms.models import model_to_dict
from asgiref.sync import sync_to_async

def get_foldernames_as_list(path, sep):
    folders_and_files = os.listdir(path)
    foldername_logs_with_different_size_balance = []
    for f in folders_and_files:
        if os.path.isdir(path+sep+f):
            foldername_logs_with_different_size_balance.append(f)
    return foldername_logs_with_different_size_balance

def generate_case_study(case_study_id):
    """
    Generate case study. This function executes all phases specified in 'to_exec' and it stores enriched log and decision tree extracted from the initial UI log in the same folder it is.

    Args:
        exp_foldername (string): name of the folder where all case study data is stored. Example 'case_study_data'
        exp_folder_complete_path (string): complete path to the folder where all case study data is stored, including the name of the folder in this path. Example 'C:\\John\\Desktop\\case_study_data'
        decision_activity (string): activity where decision we want to study is taken. Example: 'B'
        scenarios (list): list with all foldernames corresponding to the differents scenarios that will be studied in this case study
        special_colnames (dict): a dict with the keys "Case", "Activity", "Screenshot", "Variant", "Timestamp", "eyetracking_recording_timestamp", "eyetracking_gaze_point_x", "eyetracking_gaze_point_y", specifiyng as their values each column name associated of your UI log.
        to_exec (list): list of the phases we want to execute. The possible phases to include in this list are ['ui_elements_detection','ui_elements_classification','noise_filtering','extract_training_dataset','decision_tree_training']
    """
    case_study = CaseStudy.objects.get(id=case_study_id)
    times = {}
    foldername_logs_with_different_size_balance = get_foldernames_as_list(case_study.exp_folder_complete_path + sep + case_study.scenarios_to_study[0], sep)
    # DEPRECATED versions: exp_folder_complete_path + sep + "metadata" + sep
    metadata_path = metadata_location + sep + case_study.exp_foldername + "_metadata" + sep # folder to store metadata that will be used in "results" mode

    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    # year = datetime.now().date().strftime("%Y")
    # tprint("RPA-US", "rnd-xlarge")
    tprint("RPA-US     RIM", "tarty1")
    # tprint("Relevance Information Miner. Copyright " + year + ".", "pepper")
    tprint("Relevance Information Miner", "cybermedium")

    for scenario in tqdm(case_study.scenarios_to_study, desc="Scenarios that have been processed: "):
        sleep(.1)
        print("\nActual Scenario: " + str(scenario))
        param_path = case_study.exp_folder_complete_path + sep + scenario + sep
        # We check there is at least 1 phase to execute
        if case_study.ui_elements_detection or case_study.ui_elements_classification or case_study.extract_training_dataset or case_study.decision_tree_training:
            for n in foldername_logs_with_different_size_balance:
                times[n] = {}
                to_exec_args = {
                    'ui_elements_detection': (param_path+n+sep+'log.csv',
                                                 param_path+n+sep,
                                                 case_study.special_colnames,
                                                 case_study.ui_elements_detection.add_words_columns,
                                                 case_study.ui_elements_detection.skip,
                                                 case_study.ui_elements_detection.type)
                                                 # We check this phase is present in case_study to avoid exceptions
                                                 if case_study.ui_elements_detection else None,
                    'noise_filtering': (param_path+n+sep+'log.csv',
                                                 param_path+n+sep,
                                                 case_study.special_colnames,
                                                 case_study.noise_filtering.configurations,
                                                 case_study.noise_filtering.type)
                                                 # We check this phase is present in case_study to avoid exceptions
                                                 if case_study.noise_filtering else None,
                    'ui_elements_classification': (case_study.ui_elements_classification.model_weights,
                                                  case_study.ui_elements_classification.model_properties,
                                                  param_path + n + sep + 'components_npy' + sep,
                                                  param_path + n + sep + 'components_json' + sep,
                                                  param_path+n+sep + 'log.csv',
                                                  case_study.special_colnames["Screenshot"],
                                                  case_study.text_classes,
                                                  case_study.ui_elements_classification.skip,
                                                  case_study.ui_elements_classification_classes,
                                                  case_study.ui_elements_classification_image_shape,
                                                  case_study.ui_elements_classification.classifier)
                                                 # We check this phase is present in case_study to avoid exceptions
                                                  if case_study.ui_elements_classification else None,
                    'feature_extraction': (case_study.feature_extraction_technique.name,
                                                case_study.ui_elements_classification_classes,
                                                  param_path+n+sep+'enriched_log.csv',
                                                  case_study.feature_extraction_technique.skip)
                                                 # We check this phase is present in case_study to avoid exceptions
                                                  if case_study.feature_extraction_technique else None,
                    'extract_training_dataset': (case_study.decision_point_activity, param_path + n + sep + 'enriched_log.csv',
                                                param_path + n + sep, case_study.extract_training_dataset.columns_to_ignore,
                                                case_study.special_colnames["Variant"], case_study.special_colnames["Case"],
                                                case_study.special_colnames["Screenshot"], case_study.special_colnames["Timestamp"],
                                                case_study.special_colnames["Activity"])
                                                # We check this phase is present in case_study to avoid exceptions
                                                if case_study.extract_training_dataset else None,
                    'decision_tree_training': (param_path+n+sep + 'preprocessed_dataset.csv', param_path+n+sep,
                                                case_study.decision_tree_training.library,
                                                case_study.decision_tree_training.mode,
                                                case_study.decision_tree_training.algorithms,
                                                case_study.decision_tree_training.columns_to_ignore)
                                                # We check this phase is present in case_study to avoid exceptions
                                                if case_study.decision_tree_training  else None # 'autogeneration' -> to plot tree automatically
                    }

                # We go over the keys of to_exec_args, and call the corresponding functions passing the corresponding parameters
                for function_to_exec in [key for key in to_exec_args.keys() if to_exec_args[key] is not None]:
                    if function_to_exec == "decision_tree_training" and case_study.decision_tree_training.library!='sklearn':
                        res, tree_times = eval(function_to_exec)(*to_exec_args[function_to_exec])
                        times[n][function_to_exec] = tree_times
                    else:
                        times[n][function_to_exec] = {"start": time.time()}
                        output = eval(function_to_exec)(*to_exec_args[function_to_exec])
                        times[n][function_to_exec]["finish"] = time.time()

                    # TODO: accurracy_score
                    # if index == len(to_exec)-1:
                    #     times[n][index]["decision_model_accuracy"] = output

            # if not os.path.exists(scenario+sep):
            #     os.makedirs(scenario+sep)

            # Serializing json
            json_object = json.dumps(times, indent=4)
            # Writing to .json
            with open(metadata_path+scenario+"-metainfo.json", "w") as outfile:
                outfile.write(json_object)

            msg = "Case study '"+case_study.title+"' executed!!. Case study foldername: "+case_study.exp_foldername+". Metadata saved in: "+metadata_path+scenario+"-metainfo.json"
            case_study.executed = True
            case_study.save()
        else:
            msg = "None phases were set to be executed"
    return msg
    # each experiment one csv line
    # store exceution times per each phase and experiment (30 per family)
    # execute only the experiments


def times_duration(times_dict):
    if times_calculation_mode == "formatted":
        format = "%H:%M:%S.%fS"
        difference = datetime.strptime(times_dict["finish"], format) - datetime.strptime(times_dict["start"], format)
        res = difference.total_seconds()
    else:
        res = float(times_dict["finish"]) - float(times_dict["start"])
    return res


def calculate_accuracy_per_tree(decision_tree_path, expression, algorithm):
    res = {}
    # This code is useful if we want to get the expresion like: [["TextView", "B"],["ImageView", "B"]]
    # if not isinstance(levels, list):
    #     levels = [levels]
    levels = expression.replace("(", "")
    levels = levels.replace(")", "")
    levels = levels.split(" ")
    for op in ["and","or"]:
      while op in levels:
        levels.remove(op)

    if not algorithm:
        f = open(decision_tree_path + "decision_tree.log", "r").read()
        for gui_component_name_to_find in levels:
        # This code is useful if we want to get the expresion like: [["TextView", "B"],["ImageView", "B"]]
        # for gui_component_class in levels:
            # if len(gui_component_class) == 1:
            #     gui_component_name_to_find = gui_component_class[0]
            # else:
            #     gui_component_name_to_find = gui_component_class[0] + \
            #         "_"+gui_component_class[1]
            position = f.find(gui_component_name_to_find)
            res[gui_component_name_to_find] = "False"
            if position != -1:
                positions = [m.start() for m in re.finditer(gui_component_name_to_find, f)]
                number_of_nodes = int(len(positions)/2)
                if len(positions) != 2:
                    print("Warning: GUI component appears more than twice")
                for n_nod in range(0, number_of_nodes):
                    res_partial = {}
                    for index, position_i in enumerate(positions):
                        position_i += 2*n_nod
                        position_aux = position_i + len(gui_component_name_to_find)
                        s = f[position_aux:]
                        end_position = s.find("\n")
                        quantity = f[position_aux:position_aux+end_position]
                        for c in '<>= ':
                            quantity = quantity.replace(c, '')
                            res_partial[index] = quantity
                    if float(res_partial[0])-float(res_partial[1]) > gui_quantity_difference:
                        print("GUI component quantity difference greater than the expected")
                        res[gui_component_name_to_find] = "False"
                    else:
                        res[gui_component_name_to_find] = "True"
    else:
        json_f = open(decision_tree_path + decision_foldername + sep + algorithm + "-rules.json")
        decision_tree_decision_points = json.load(json_f)
        for gui_component_name_to_find in levels:
            # res_partial = []
            # gui_component_to_find_index = 0
            res_aux = False
            for node in decision_tree_decision_points:
                res_aux = res_aux or (node['return_statement'] == 0 and ('x0_'+gui_component_name_to_find in node['feature_name']))
                    # return_statement: filtering return statements (only conditions evaluations)
                    # feature_name: filtering feature names as the ones contained on the expression
            #         feature_complete_id = 'obj['+str(node['feature_idx'])+']'
            #         pos1 = node['rule'].find(feature_complete_id) + len(feature_complete_id)
            #         pos2 = node['rule'].find(':')
            #         quantity = node['rule'][pos1:pos2]
            #         res_partial.append(quantity)
            #         for c in '<>= ':
            #             quantity = quantity.replace(c, '')
            #             res_partial[gui_component_to_find_index] = quantity
            #         gui_component_to_find_index +=1
            # if res_partial and len(res_partial) == 2:
            #     res_aux = (float(res_partial[0])-float(res_partial[1]) <= quantity_difference)
            #     if not res_aux:
            #         print("GUI component quantity difference greater than the expected: len->" + str(len(res_partial)))
            # else:
            #     res_aux = False
            res[gui_component_name_to_find] = str(res_aux)

    s = expression
    print(res)
    for gui_component_name_to_find in levels:
        s = s.replace(gui_component_name_to_find, res[gui_component_name_to_find])

    res = eval(s)

    if not res:
      print("Condition " + str(expression) + " is not fulfilled")
    return int(res)


def experiments_results_collectors(case_study, decision_tree_filename):
    """
    Calculates the results of a given case_study

    :param case_study: Case study to analyze
    :type case_study: CaseStudy
    :returns: Path leading to the csv containing the results
    :rtype: str
    """
    csv_filename = case_study.exp_folder_complete_path + sep + case_study.exp_foldername + "_results.csv"

    times_info_path = metadata_location + sep + case_study.exp_foldername + "_metadata" + sep
    preprocessed_log_filename = "preprocessed_dataset.csv"

    # print("Scenarios: " + str(scenarios))
    family = []
    balanced = []
    log_size = []
    scenario_number = []
    log_column = []
    phases_info = {}
    # detection_time = []
    # classification_time = []
    # flat_time = []
    # tree_training_time = []
    # tree_training_accuracy = []

    decision_tree_algorithms = case_study.decision_tree_training.algorithms if (case_study.decision_tree_training and
                                                                                case_study.decision_tree_training.algorithms) else None

    if decision_tree_algorithms:
        accuracy = {}
    else:
        accuracy = []

    # TODO: new experiment files structure
    for scenario in tqdm(case_study.scenarios_to_study,
                         desc="Experiment results that have been processed"):
        sleep(.1)
        scenario_path = case_study.exp_folder_complete_path + sep + scenario
        family_size_balance_variations = get_foldernames_as_list(
            scenario_path, sep)
        # if case_study.drop and (case_study.drop in family_size_balance_variations):
        #     family_size_balance_variations.remove(case_study.drop)
        json_f = open(times_info_path+scenario+"-metainfo.json")
        times = json.load(json_f)
        for n in family_size_balance_variations:
            metainfo = n.split("_")
            # path example of decision tree specification: rim\CSV_exit\resources\version1637144717955\scenario_1\Basic_10_Imbalanced\decision_tree.log
            decision_tree_path = scenario_path + sep + n + sep

            with open(scenario_path + sep + n + sep + preprocessed_log_filename, newline='') as f:
                csv_reader = csv.reader(f)
                csv_headings = next(csv_reader)
            log_column.append(len(csv_headings))

            family.append(metainfo[0])
            log_size.append(metainfo[1])
            scenario_number.append(scenario.split("_")[1])
            # 1 == Balanced, 0 == Imbalanced
            balanced.append(1 if metainfo[2] == "Balanced" else 0)

            phases = [phase for phase in ["ui_elements_detection", "ui_elements_classification",
                      "extract_training_dataset", "decision_tree_training"] if getattr(case_study, phase) is not None]
            for phase in phases:
                if not (phase == 'decision_tree_training' and decision_tree_algorithms):
                    if phase in phases_info:
                        phases_info[phase].append(times_duration(times[n][phase]))
                    else:
                        phases_info[phase] = [times_duration(times[n][phase])]

            # TODO: accurracy_score
            # tree_training_accuracy.append(times[n]["3"]["decision_model_accuracy"])

            if decision_tree_algorithms:
                for alg in decision_tree_algorithms:
                    if (alg+'_accuracy') in accuracy:
                        accuracy[alg+'_tree_training_time'].append(times_duration(times[n]['decision_tree_training'][alg]))
                        accuracy[alg+'_accuracy'].append(calculate_accuracy_per_tree(decision_tree_path, case_study.gui_class_success_regex, alg))
                    else:
                        accuracy[alg+'_tree_training_time'] = [times_duration(times[n]['decision_tree_training'][alg])]
                        accuracy[alg+'_accuracy'] = [calculate_accuracy_per_tree(decision_tree_path, case_study.gui_class_success_regex, alg)]
            else:
                # Calculate level of accuracy
                accuracy.append(calculate_accuracy_per_tree(decision_tree_path, case_study.gui_class_success_regex, None))

    dict_results = {
        'family': family,
        'balanced': balanced,
        'log_size': log_size,
        'scenario_number': scenario_number,
        'log_column': log_column,
        # TODO: accurracy_score
        # 'tree_training_accuracy': tree_training_accuracy,
    }


    if isinstance(accuracy, dict):
        for sub_entry in accuracy.items():
            dict_results[sub_entry[0]] = sub_entry[1]

    for phase in phases_info.keys():
        dict_results[phase] = phases_info[phase]

    df = pd.DataFrame(dict_results)
    df.to_csv(csv_filename)

    return csv_filename

# ========================================================================
# RUN CASE STUDY
# ========================================================================

# EXAMPLE JSON REQUEST BODY
# {
#     "title": "Test Case Study",
#     "mode": "results",
#     "exp_foldername": "Advanced_10_30",
#     "phases_to_execute": {
#         "extract_training_dataset": {
#             "columns_to_ignore": ["Coor_X", "Coor_Y"]
#         },
#         "decision_tree_training": {
#             "library": "chefboost",
#             "algorithms": ["ID3", "CART", "CHAID", "C4.5"],
#             "mode": "autogeneration",
#             "columns_to_ignore": ["Timestamp_start", "Timestamp_end"]
#         }
#     },
#     "decision_point_activity": "B",
#     "exp_folder_complete_path": "C:\\Users\\Antonio\\Desktop\\caise data\\Advanced_10_30",
#     "gui_class_success_regex": "CheckBox_B or ImageView_B or TextView_B",
#     "gui_quantity_difference": 1,
#     "scenarios_to_study": null,
#     "drop": null
# }

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

        cs_serializer = CaseStudySerializer(data=data)
        cs_serializer.is_valid(raise_exception=True)
        case_study = cs_serializer.save()

        # For each phase we want to execute, we create a database row for it and relate it with the case study
        for phase in data['phases_to_execute']:
            match phase:
                case "ui_elements_detection":
                    serializer = UIElementsDetectionSerializer(data=data['phases_to_execute'][phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.ui_elements_detection = serializer.save()
                case "noise_filtering":
                    serializer = NoiseFilteringSerializer(data=data['phases_to_execute'][phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.noise_filtering = serializer.save()
                case "ui_elements_classification":
                    serializer = UIElementsClassificationSerializer(data=data['phases_to_execute'][phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.ui_elements_classification = serializer.save()
                case "feature_extraction_technique":
                    serializer = FeatureExtractionTechniqueSerializer(data=data['phases_to_execute'][phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.feature_extraction_technique = serializer.save()
                case "extract_training_dataset":
                    serializer = ExtractTrainingDatasetSerializer(data=data['phases_to_execute'][phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.extract_training_dataset = serializer.save()
                case "decision_tree_training":
                    serializer = DecisionTreeTrainingSerializer(data=data['phases_to_execute'][phase])
                    serializer.is_valid(raise_exception=True)
                    case_study.decision_tree_training = serializer.save()
                case _:
                    pass

        # Updating the case study with the foreign keys of the phases to execute
        case_study.save()
        transaction_works = True
    init_generate_case_study.delay(case_study.id)

    return transaction_works, case_study

class CaseStudyView(generics.ListCreateAPIView):
    # permission_classes = [IsAuthenticatedUser]
    serializer_class = CaseStudySerializer

    def get_queryset(self):
        return CaseStudy.objects.filter(shopper=self.request.user)

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
                    response_content = {"message": "phases_to_execute must be of type dict!!!!! and must be composed by phases contained in ['ui_elements_detection','ui_elements_classification','noise_filtering', 'feature_extraction_technique','extract_training_dataset','decision_tree_training']"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                if not case_study_serialized.data['phases_to_execute']['ui_elements_detection']['type'] in ["legacy", "uied"]:
                    response_content = {"message": "Elements Detection algorithm must be one of ['legacy', 'uied']"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                if not case_study_serialized.data['phases_to_execute']['ui_elements_classification']['type'] in ["legacy", "uied"]:
                    response_content = {"message": "Elements Classification algorithm must be one of ['legacy', 'uied']"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return Response(response_content, st)

                for phase in dict(case_study_serialized.data['phases_to_execute']).keys():
                    if not(phase in ['ui_elements_detection','ui_elements_classification','noise_filtering','feature_extraction_technique','extract_training_dataset','decision_tree_training']):
                        response_content = {"message": "phases_to_execute must be composed by phases contained in ['ui_elements_detection','ui_elements_classification','noise_filtering','feature_extraction_technique','extract_training_dataset','decision_tree_training']"}
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                        execute_case_study = False
                        return Response(response_content, st)

                for path in [case_study_serialized.data['phases_to_execute']['ui_elements_classification']['model'],
                             case_study_serialized.data['phases_to_execute']['ui_elements_classification']['model_properties'],
                             case_study_serialized.data['exp_folder_complete_path']]:
                    if not os.path.exists(path):
                        response_content = {"message": f"The following file or directory does not exists: {path}"}
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                        execute_case_study = False
                        return Response(response_content, st)

                if execute_case_study:
                    # init_case_study_task.delay(request.data)
                    transaction_works, case_study = case_study_generator(case_study_serialized.data)
                    if not transaction_works:
                        st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    else:    
                        response_content = {"message": f"Case study with id:{case_study.id} generated"}

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
                experiments_results_collectors(case_study, "descision_tree.log")
                response = {"message": case_study.exp_foldername + ' case study results collected!'}
            else:
                response = {"message": 'The processing of this case study has not yet finished, please try again in a few minutes'}

        except Exception as e:
            response = {"message": f"Case Study with id {case_study_id} not found"}
            st = status.HTTP_404_NOT_FOUND

        return Response(response, status=st)