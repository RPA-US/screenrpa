from celery import shared_task
from analyzer.models import CaseStudy
from analyzer.serializers import CaseStudySerializer
import analyzer.views as analyzer
from rest_framework import status

from multiprocessing.connection import wait
from celery import shared_task
from django.shortcuts import render

# Create your views here.
import os
import json
import time
from art import tprint
from tqdm import tqdm
from time import sleep
from rim.settings import metadata_location, sep
from decisiondiscovery.views import decision_tree_training, extract_training_dataset
from featureextraction.views import ui_elements_classification, feature_extraction
from featureextraction.detection import ui_elements_detection
# CaseStudyView
from rest_framework import generics, status, viewsets #, permissions
from rest_framework.response import Response
# from rest_framework.pagination import PageNumberPagination
from .serializers import CaseStudySerializer

# Functions in this file with the shared_task decorator will be picked up by celery and executed asynchronously

@shared_task()
def generate_case_study(case_study_id):
    """
    Generate case study. This function executes all phases specified in 'to_exec' and it stores enriched log and decision tree extracted from the initial UI log in the same folder it is.

    Args:
        exp_foldername (string): name of the folder where all case study data is stored. Example 'case_study_data'
        exp_folder_complete_path (string): complete path to the folder where all case study data is stored, including the name of the folder in this path. Example 'C:\\John\\Desktop\\case_study_data'
        decision_activity (string): activity where decision we want to study is taken. Example: 'B'
        scenarios (list): list with all foldernames corresponding to the differents scenarios that will be studied in this case study
        special_colnames (dict): a dict with the keys "Case", "Activity", "Screenshot", "Variant", "Timestamp", "eyetracking_recording_timestamp", "eyetracking_gaze_point_x", "eyetracking_gaze_point_y", specifiyng as their values each column name associated of your UI log.
        to_exec (list): list of the phases we want to execute. The possible phases to include in this list are ['ui_elements_detection','ui_elements_classification','extract_training_dataset','decision_tree_training']
    """
    case_study = CaseStudy.objects.get(id=case_study_id)
    times = {}
    foldername_logs_with_different_size_balance = analyzer.get_foldernames_as_list(case_study.exp_folder_complete_path + sep + case_study.scenarios_to_study[0], sep)
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
                                                 case_study.ui_elements_detection.eyetracking_log_filename,
                                                 case_study.ui_elements_detection.add_words_columns,
                                                 case_study.ui_elements_detection.overwrite_info,
                                                 case_study.ui_elements_detection.algorithm)
                                                 # We check this phase is present in case_study to avoid exceptions
                                                 if case_study.ui_elements_detection else None,
                    'ui_elements_classification': (case_study.ui_elements_classification.model_weights,
                                                  case_study.ui_elements_classification.model_properties,
                                                  param_path + n + sep + 'components_npy' + sep,
                                                  param_path + n + sep + 'components_json' + sep,
                                                  param_path+n+sep + 'log.csv',
                                                  case_study.special_colnames["Screenshot"],
                                                  case_study.ui_elements_classification.overwrite_info,
                                                  case_study.ui_elements_classification.ui_elements_classification_classes,
                                                  case_study.ui_elements_classification.ui_elements_classification_shape,
                                                  case_study.ui_elements_classification.classifier)
                                                 # We check this phase is present in case_study to avoid exceptions
                                                  if case_study.ui_elements_classification else None,
                    'feature_extraction': (case_study.feature_extraction_technique.name,
                                                  param_path+n+sep+'enriched_log.csv',
                                                  case_study.feature_extraction_technique.overwrite_info)
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
        else:
            msg = "None phases were set to be executed"
    return msg
    # each experiment one csv line
    # store exceution times per each phase and experiment (30 per family)
    # execute only the experiments