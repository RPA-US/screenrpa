import os
import json
import pandas as pd
import logging
import re
import datetime
import email
import base64
import lxml.etree as ET
from lxml import html
from core.settings import sep
from apps.featureextraction.utils import case_study_has_feature_extraction_technique, get_feature_extraction_technique, case_study_has_ui_elements_detection, get_ui_elements_detection, case_study_has_ui_elements_classification, get_ui_elements_classification, case_study_has_info_postfiltering, get_info_postfiltering, case_study_has_info_prefiltering, get_info_prefiltering
from apps.behaviourmonitoring.utils import get_monitoring, case_study_has_monitoring
from apps.processdiscovery.utils import get_process_discovery, case_study_has_process_discovery
from apps.decisiondiscovery.utils import get_extract_training_dataset, case_study_has_extract_training_dataset, get_decision_tree_training, case_study_has_decision_tree_training
from apps.featureextraction.UIFEs.feature_extraction_techniques import *
from apps.featureextraction.UIFEs.aggregate_features_as_dataset_columns import *
from django.utils.translation import gettext_lazy as _

def get_foldernames_as_list(path, sep):
    folders_and_files = os.listdir(path)
    foldername_logs_with_different_size_balance = []
    for f in folders_and_files:
        if os.path.isdir(path+sep+f):
            foldername_logs_with_different_size_balance.append(f)
    return foldername_logs_with_different_size_balance


###########################################################################################################################
# MHT to XES/CSV ##########################################################################################################
###########################################################################################################################

def store_screenshots(payload, path_to_store_screenshots):
  images_position = 0
  for index, p in enumerate(payload):
    if "image" in p.get_content_type():
        images_position = index
        break
  for i in range(images_position, len(payload)):
    part = payload[i]
    if "Content-Location" in payload[i]:
      filename = part["Content-Location"]
    else:
      logging.exception(_("analyzer/utils/store_screenshots. line 49. MIME Html format not contains Content-Location header in screenshots"))
      raise Exception(_("MIME Html format not contains Content-Location header in screenshots"))
    image_data = part.get_payload()

    # Decode the image data from base64 encoding
    image_data_decoded = base64.b64decode(image_data)

    if not os.path.exists(path_to_store_screenshots):
        os.mkdir(path_to_store_screenshots)

    # Save the image to a file
    with open(path_to_store_screenshots + filename, 'wb') as f:
        f.write(image_data_decoded)
        print(f"Saved image file {filename}")

def from_html_to_xes(org_resource, myhtml, root_file_path, output_filename):
    root = html.fromstring(myhtml)
    myxml = root.xpath("//script[@id='myXML']")[0].text_content()
    myxml = myxml.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
    
    my_xml_doc = ET.fromstring(myxml)

    # Create an XES document
    xes = ET.Element('log', {'ocel.version': '0.1'})
    trace = ET.SubElement(xes, 'trace')
    ET.SubElement(trace, 'string', {'key': 'concept:name'}).text = 'Object Centric Event Log'
    ET.SubElement(trace, 'string', {'key': 'ocel:ordering'}).text = 'timestamp'
    global_log = ET.SubElement(trace, 'global', {'scope': 'log'})
    obj_types = ET.SubElement(global_log, 'list', {'key': 'ocel:object-types'})
    ET.SubElement(obj_types, 'string', {'key': 'object-type'}).text = 'customers'
    att_names = ET.SubElement(global_log, 'list', {'key': 'ocel:attribute-names'})
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'FileName'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'FileCompany'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'FileDescription'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'CommandLine'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'ActionNumber'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'Pid'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'ProgramId'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'FileId'
    ET.SubElement(att_names, 'string', {'key': 'attribute-name'}).text = 'FileVersion'

    recorded_session = my_xml_doc.find("UserActionData").find("RecordSession")
    for i, each_action in enumerate(recorded_session.getchildren()):
      event = ET.SubElement(trace, 'event')
      children = each_action.getchildren()
      ET.SubElement(event, 'string', {'key': 'ocel:eid'}).text = str(i)
      ET.SubElement(event, 'string', {'key': 'ocel:org:resource'}).text = org_resource
      ET.SubElement(event, 'string', {'key': 'ocel:concept:name'}).text = children[0].text
      ET.SubElement(event, 'string', {'key': 'ocel:type:event'}).text = children[1].text
      coords = children[2].text.strip().split(",")
      ET.SubElement(event, 'float', {'key': 'ocel:click:coorX'}).text = coords[0]
      ET.SubElement(event, 'float', {'key': 'ocel:click:coorY'}).text = coords[1]
      screen_coords = children[3].text.strip().split(",")
      ET.SubElement(event, 'float', {'key': 'ocel:screenshot:screenW'}).text = screen_coords[2]
      ET.SubElement(event, 'float', {'key': 'ocel:screenshot:screenH'}).text = screen_coords[3]
      if len(children) > 5:
        ET.SubElement(event, 'string', {'key': 'ocel:screenshot:name'}).text = children[5].text
      else:
        ET.SubElement(event, 'string', {'key': 'ocel:screenshot:name'}).text = "None"
      ET.SubElement(event, 'date', {'key': 'ocel:timestamp'}).text = each_action.get("Time")
      ET.SubElement(event, 'string', {'key': 'FileName'}).text = each_action.get("FileName")
      ET.SubElement(event, 'string', {'key': 'FileCompany'}).text = each_action.get("FileCompany")
      ET.SubElement(event, 'string', {'key': 'FileDescription'}).text = each_action.get("FileDescription")
      ET.SubElement(event, 'string', {'key': 'CommandLine'}).text = each_action.get("CommandLine")
      ET.SubElement(event, 'string', {'key': 'ActionNumber'}).text = each_action.get("ActionNumber")
      ET.SubElement(event, 'string', {'key': 'Pid'}).text = each_action.get("Pid")
      ET.SubElement(event, 'string', {'key': 'ProgramId'}).text = each_action.get("ProgramId")
      ET.SubElement(event, 'string', {'key': 'FileId'}).text = each_action.get("FileId")
      ET.SubElement(event, 'string', {'key': 'FileVersion'}).text = each_action.get("FileVersion")

    res_path = root_file_path + output_filename + '.xes'
    # Write the XES document to a file
    with open(res_path, 'wb') as f:
        f.write(ET.tostring(xes, pretty_print=True))
        
    return res_path
        

def from_html_to_csv(org_resource, myhtml, root_file_path, output_filename):
    root = html.fromstring(myhtml)
    myxml = root.xpath("//script[@id='myXML']")[0].text_content()
    myxml = myxml.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
    
    my_xml_doc = ET.fromstring(myxml)

    events = []
    recorded_session = my_xml_doc.find("UserActionData").find("RecordSession")
    for i, each_action in enumerate(recorded_session.getchildren()):
        event = {}
        event['ocel:eid'] = str(i)
        event['ocel:org:resource'] = org_resource
        event['ocel:concept:name'] = each_action[0].text
        event['ocel:type:event'] = each_action[1].text
        coords = each_action[2].text.strip().split(",")
        event['ocel:click:coorX'] = coords[0]
        event['ocel:click:coorY'] = coords[1]
        screen_coords = each_action[3].text.strip().split(",")
        event['ocel:screenshot:screenW'] = screen_coords[2]
        event['ocel:screenshot:screenH'] = screen_coords[3]
        if len(each_action) > 5:
            event['ocel:screenshot:name'] = each_action[5].text
        else:
            event['ocel:screenshot:name'] = "None"
        event['ocel:timestamp'] = each_action.get("Time")
        event['FileName'] = each_action.get("FileName")
        event['FileCompany'] = each_action.get("FileCompany")
        event['FileDescription'] = each_action.get("FileDescription")
        event['CommandLine'] = each_action.get("CommandLine")
        event['ActionNumber'] = each_action.get("ActionNumber")
        event['Pid'] = each_action.get("Pid")
        event['ProgramId'] = each_action.get("ProgramId")
        event['FileId'] = each_action.get("FileId")
        event['FileVersion'] = each_action.get("FileVersion")
        events.append(event)

    res_path = root_file_path + output_filename + '.csv'

    # Write the events to a CSV file
    df = pd.DataFrame(events)
    df.to_csv(res_path, index=False)
    
    return res_path
    
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

def format_mht_file(mht_file_path, output_format, output_path, output_filename, org_resource):
  with open(mht_file_path) as mht_file: 
    msg = email.message_from_file(mht_file)
    myhtml = msg.get_payload()[0].get_payload()
    
  store_screenshots(msg.get_payload(), output_path)
  
  if output_format == "mht_xes":
    res_path = from_html_to_xes(org_resource, myhtml, output_path, output_filename)
  elif output_format == "mht_csv":
    res_path = from_html_to_csv(org_resource, myhtml, output_path, output_filename)
  else:
    logging.exception(_("analyzer/utils/format_mht_file. line 187. MHT file format selected doesnt exists"))
    raise Exception(_("You select a format mht file that doesnt exists"))
  
  return res_path
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
def get_format_pattern(datetime_parenthesis, sep):
    if "â€Ž" in datetime_parenthesis:
      format_pattern = 'â€Ž%d'+sep+'â€Ž%m'+sep+'â€Ž%Y %H:%M:%S'
    elif "/" in datetime_parenthesis:
      format_pattern = '\u200e%d'+sep+'\u200e%m'+sep+'\u200e%Y %H:%M:%S'
    return format_pattern


def get_mht_log_start_datetime(mht_file_path, pattern):
    with open(mht_file_path) as mht_file: 
        msg = email.message_from_file(mht_file)
        myhtml = msg.get_payload()[0].get_payload()

    root = html.fromstring(myhtml)
    myxml = root.xpath("//div[@id='Step1']")[0].text_content()

    patron = r'Step 1:\s*\((.*?)\)'

    dateRegistered = re.search(patron, myxml)

    if dateRegistered:
        datetime_parenthesis = dateRegistered.group(1)
    else:
        logging.exception(_("analyzer/utils/format_mht_file. line 211. The MHT file doesnt follows the format:'Step 1: (datetime)'"))
        raise Exception(_("The MHT file doesnt have '(datetime)' after 'Step 1:'"))
      
    if pattern:
      format_pattern = pattern
    elif "/" in datetime_parenthesis:
      format_pattern = get_format_pattern(datetime_parenthesis, "/")
    elif "-" in datetime_parenthesis: 
      format_pattern = get_format_pattern(datetime_parenthesis, "-")
    else:
      raise Exception(_("The MHT file doesnt have a valid datetime format"))

    return datetime.datetime.strptime(datetime_parenthesis, format_pattern)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

def phases_to_execute_specs(execution, path_scenario, path_results):
    to_exec_args = {}
    # We check this phase is present in execution.case_study to avoid exceptions
    if execution.monitoring:
        to_exec_args['monitoring'] = (path_scenario +'log.csv',
                                        path_scenario,
                                        execution.case_study.special_colnames,
                                        execution.monitoring)
        
    if execution.prefilters:
        to_exec_args['info_prefiltering'] =  (path_scenario +'log.csv',
                                        path_scenario,
                                        execution.case_study.special_colnames,
                                        execution.prefilters.configurations,
                                        execution.prefilters.skip,
                                        execution.prefilters.type)
        
    if execution.ui_elements_detection:

        to_exec_args['ui_elements_detection'] = (path_scenario +'log.csv',
                                        path_scenario,
                                        execution.ui_elements_detection.input_filename,
                                        execution.case_study.special_colnames,
                                        execution.ui_elements_detection.configurations,
                                        execution.ui_elements_detection.skip,
                                        execution.ui_elements_detection.type,
                                        execution.ui_elements_detection.ocr)
                                        
    if execution.ui_elements_classification:
        to_exec_args['ui_elements_classification'] = (execution.ui_elements_classification.model.path, # specific extractors
                                        path_results + 'components_npy' + sep,
                                        path_results + 'components_json' + sep,
                                        path_scenario + 'log.csv',
                                        execution.case_study.special_colnames["Screenshot"],
                                        execution.ui_elements_classification.model.text_classname,
                                        execution.ui_elements_classification.skip,
                                        execution.ui_elements_classification.model.classes,
                                        execution.ui_elements_classification.model.image_shape,
                                        execution.ui_elements_classification.type)
        
    if execution.postfilters:
        to_exec_args['info_postfiltering'] = (path_scenario +'log.csv',
                                        path_scenario,
                                        execution.case_study.special_colnames,
                                        execution.postfilters.configurations,
                                        execution.postfilters.skip,
                                        execution.postfilters.type)
        
    if execution.feature_extraction_technique:
        to_exec_args['feature_extraction_technique'] = (execution.ui_elements_classification_classes,
                                        execution.feature_extraction_technique.decision_point_activity,
                                        execution.case_study.special_colnames["Case"],
                                        execution.case_study.special_colnames["Activity"],
                                        execution.case_study.special_colnames["Screenshot"],
                                        path_scenario + 'components_json' + sep,
                                        path_scenario + 'flattened_dataset.json',
                                        path_scenario + 'log.csv',
                                        path_scenario + execution.feature_extraction_technique.technique_name+'_enriched_log.csv',
                                        execution.case_study.text_classname,
                                        execution.feature_extraction_technique.consider_relevant_compos,
                                        execution.feature_extraction_technique.relevant_compos_predicate,
                                        execution.feature_extraction_technique.identifier,
                                        execution.feature_extraction_technique.skip,
                                        execution.feature_extraction_technique.technique_name)
        
    if execution.process_discovery:
        to_exec_args['process_discovery'] = (path_scenario +'log.csv',
                                        path_scenario,
                                        execution.case_study.special_colnames,
                                        execution.process_discovery.configurations,
                                        execution.process_discovery.skip,
                                        execution.process_discovery.type)
        
    if execution.extract_training_dataset:
        to_exec_args['extract_training_dataset'] = (execution.feature_extraction_technique.decision_point_activity, 
                                        execution.case_study.target_label,
                                        execution.case_study.special_colnames,
                                        execution.extract_training_dataset.columns_to_drop,
                                        path_scenario + 'log.csv',
                                        path_scenario, 
                                        execution.extract_training_dataset.columns_to_drop_before_decision_point)
        
    if execution.feature_extraction_technique:
        to_exec_args['aggregate_features_as_dataset_columns'] = (execution.ui_elements_classification_classes,
                                        execution.feature_extraction_technique.decision_point_activity,
                                        execution.case_study.special_colnames["Case"],
                                        execution.case_study.special_colnames["Activity"],
                                        execution.case_study.special_colnames["Screenshot"],
                                        path_scenario,
                                        path_scenario + 'flattened_dataset.json',
                                        path_scenario + 'log.csv',
                                        path_scenario + execution.feature_extraction_technique.technique_name+'_enriched_log.csv',
                                        execution.case_study.text_classname,
                                        execution.feature_extraction_technique.consider_relevant_compos,
                                        execution.feature_extraction_technique.relevant_compos_predicate,
                                        execution.feature_extraction_technique.identifier,
                                        execution.feature_extraction_technique.skip,
                                        execution.feature_extraction_technique.technique_name)
        
    if execution.decision_tree_training:
      to_exec_args['decision_tree_training'] = (execution.case_study, path_scenario)
        
    return to_exec_args