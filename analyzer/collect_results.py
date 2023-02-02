import json
import time
import csv
import pandas as pd
import re
from tqdm import tqdm
import time
from datetime import datetime
from rim.settings import times_calculation_mode, metadata_location, sep, decision_foldername, gui_quantity_difference, default_phases, scenario_nested_folder, FLATTENED_DATASET_NAME, several_iterations
import csv
from rim.utils import get_foldernames_as_list


def times_duration(times, phases_info, scenario, phase):
    if several_iterations:
        phases_info[phase].append(times[scenario][phase]["duration"])
        
    else:
        phases_info[phase].append(times_duration(float(times[scenario][phase]["finish"]) - float(times[scenario][phase]["start"])))
    return phases_info


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


def experiments_results_collectors_old_structure(case_study, decision_tree_filename):
    """
    Calculates the results of a given case_study

    :param case_study: Case study to analyze
    :type case_study: CaseStudy
    :returns: Path leading to the csv containing the results
    :rtype: str
    """
    exp_foldername_timestamp = case_study.exp_foldername + '_' + str(case_study.id)
    csv_filename = case_study.exp_folder_complete_path + sep + exp_foldername_timestamp + "_results.csv"
    times_info_path = metadata_location + sep + exp_foldername_timestamp + "_metadata" + sep
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
        time.sleep(.1)
        scenario_path = case_study.exp_folder_complete_path + sep + scenario
        family_size_balance_variations = get_foldernames_as_list(scenario_path, sep)
        # if case_study.drop and (case_study.drop in family_size_balance_variations):
        #     family_size_balance_variations.remove(case_study.drop)
        json_f = open(times_info_path+"-metainfo.json")
        times = json.load(json_f)
        for n in family_size_balance_variations:
            metainfo = n.split("_")
            # path example of decision tree specification: rim\CSV_exit\resources\version1637144717955\scenario_1\Basic_10_Imbalanced\decision_tree.log
            decision_tree_path = scenario_path + sep + n + sep

            with open(scenario_path + sep + n + sep + preprocessed_log_filename, newline='') as f:
                csv_reader = csv.reader(f)
                csv_headings = next(csv_reader)
            log_column.append(len(csv_headings))

            family.append(metainfo[2])
            log_size.append(metainfo[3])
            scenario_number.append(scenario.split("_")[1])
            # 1 == Balanced, 0 == Imbalanced
            balanced.append(1 if metainfo[2] == "Balanced" else 0)

            phases = [phase for phase in default_phases if getattr(case_study, phase) is not None]
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

    return df, csv_filename

def experiments_results_collectors(case_study, decision_tree_filename):
    """
    Calculates the results of a given case_study

    :param case_study: Case study to analyze
    :type case_study: CaseStudy
    :returns: Path leading to the csv containing the results
    :rtype: str
    """
    exp_foldername_timestamp = case_study.exp_foldername + '_' + str(case_study.id)
    csv_filename = case_study.exp_folder_complete_path + sep + exp_foldername_timestamp + "_results.csv"
    times_info_path = metadata_location + sep
    
    # print("Scenarios: " + str(scenarios))
    balanced = []
    log_size = []
    scenario_number = []
    phases_info = {}
    num_UI_elements = []
    num_screenshots = []
    max_ui_elements_number = []
    min_ui_elements_number = []
    columns_len = []
    density = []
    tree_levels = []
    # detection_time = []
    # classification_time = []
    # flat_time = []
    # tree_training_time = []
    tree_training_accuracy = []

    decision_tree_algorithms = case_study.decision_tree_training.algorithms if (case_study.decision_tree_training and
                                                                                case_study.decision_tree_training.algorithms) else None

    if decision_tree_algorithms:
        accuracy = {}
    else:
        accuracy = []

    # TODO: new experiment files structure
    scenarios = get_foldernames_as_list(case_study.exp_folder_complete_path, sep)
    for scenario in tqdm(scenarios, desc="Experiment results that have been processed"):
        time.sleep(.1)
        scenario_path = case_study.exp_folder_complete_path + sep + scenario
        json_f = open(times_info_path+str(case_study.id)+"-metainfo.json")
        times = json.load(json_f)
        metainfo = scenario.split("_")
        decision_tree_path = scenario_path + sep

        log_size.append(metainfo[2])
        scenario_number.append(metainfo[1])
        # 1 == Balanced, 0 == Imbalanced
        balanced.append(1 if metainfo[3] == "Balanced" else 0)

        if case_study.feature_extraction_technique:
            num_UI_elements.append(times[scenario]["feature_extraction_technique"]["num_UI_elements"])
            num_screenshots.append(times[scenario]["feature_extraction_technique"]["num_screenshots"])
            max_ui_elements_number.append(times[scenario]["feature_extraction_technique"]["max_#UI_elements"])
            min_ui_elements_number.append(times[scenario]["feature_extraction_technique"]["min_#UI_elements"])
            density.append(times[scenario]["feature_extraction_technique"]["num_UI_elements"]/times[scenario]["feature_extraction_technique"]["num_screenshots"])
        if case_study.decision_tree_training:
            columns_len.append(times[scenario]["decision_tree_training"]["columns_len"])
            tree_training_accuracy.append(times[scenario]["decision_tree_training"]["accuracy"])
            
        phases = [phase for phase in default_phases if getattr(case_study, phase) is not None]
        for phase in phases:
            if not (phase == 'decision_tree_training' and decision_tree_algorithms):
                if phase in phases_info:
                    phases_info = times_duration(times, phases_info, scenario, phase)
                else:
                    if several_iterations:
                        phases_info[phase] = [times[scenario][phase]["duration"]]
                    else:
                        phases_info[phase] = [times_duration(times[scenario][phase])]

        # TODO: accurracy_score
        if decision_tree_algorithms:
            for alg in decision_tree_algorithms:
                if (alg+'_tree_training_time') in accuracy:
                    if several_iterations:
                        accuracy[alg+'_tree_training_time'].append(times[scenario]['decision_tree_training'][alg]["duration"])
                    else:
                        accuracy[alg+'_tree_training_time'].append(times_duration(times[scenario]['decision_tree_training'][alg]))
                    # accuracy[alg+'_tree_training_time'].append(times[scenario]['decision_tree_training'][alg]["tree_levels"])
                    # accuracy[alg+'_accuracy'].append(calculate_accuracy_per_tree(decision_tree_path, case_study.gui_class_success_regex, alg))
                else:
                    if several_iterations:
                        accuracy[alg+'_tree_training_time'] = [times[scenario]['decision_tree_training'][alg]["duration"]]
                    else:
                        accuracy[alg+'_tree_training_time'] = [times_duration(times[scenario]['decision_tree_training'][alg])]
                    # accuracy[alg+'_tree_training_time'] = [times[scenario]['decision_tree_training'][alg]["tree_levels"]]
                    # accuracy[alg+'_accuracy'] = [calculate_accuracy_per_tree(decision_tree_path, case_study.gui_class_success_regex, alg)]
        else:
            print("DECISION TREE results collection fail! :(")
            # Calculate level of accuracy
            # accuracy.append(calculate_accuracy_per_tree(decision_tree_path, case_study.gui_class_success_regex, None))
    
    family_id = 'cs_' + str(case_study.id) + '_' + case_study.exp_foldername
    
    dict_results = {
        family_id: scenarios,
        'scenario_number': scenario_number,
        'balanced': balanced,
        'log_size': log_size
    }
    
    
    if case_study.feature_extraction_technique:
        dict_results['num_UI_elements'] =  num_UI_elements
        dict_results['num_screenshots'] =  num_screenshots
        dict_results['density'] =  density
        dict_results['max_#UI_elements'] =  max_ui_elements_number
        dict_results['min_#UI_elements'] =  min_ui_elements_number
        
    if case_study.decision_tree_training:
        dict_results['columns_len'] =  columns_len
        dict_results['tree_training_accuracy'] = tree_training_accuracy
          

    if isinstance(accuracy, dict):
        for sub_entry in accuracy.items():
            dict_results[sub_entry[0]] = sub_entry[1]

    for phase in phases_info.keys():
        dict_results[phase] = phases_info[phase]

    df = pd.DataFrame(dict_results)
    df.to_csv(csv_filename)

    return df, csv_filename
