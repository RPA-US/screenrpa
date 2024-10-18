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
import pytz
from dateutil.parser import parse
# from apps.featureextraction.utils import case_study_has_feature_extraction_technique, get_feature_extraction_technique, case_study_has_ui_elements_detection, get_ui_elements_detection, case_study_has_ui_elements_classification, get_ui_elements_classification, case_study_has_postfilters, get_postfilters, case_study_has_prefilters, get_prefilters
# from apps.behaviourmonitoring.utils import get_monitoring, case_study_has_monitoring
# from apps.processdiscovery.utils import get_process_discovery, case_study_has_process_discovery
# from apps.decisiondiscovery.utils import get_extract_training_dataset, case_study_has_extract_training_dataset, get_decision_tree_training, case_study_has_decision_tree_training
from apps.featureextraction.UIFEs.single_feature_extraction_techniques import *
from apps.featureextraction.UIFEs.aggregate_features_as_dataset_columns import *
from django.utils.translation import gettext_lazy as _

def get_foldernames_as_list(path):
    folders_and_files = os.listdir(path)
    foldername_logs_with_different_size_balance = []
    for f in folders_and_files:
        # path+sep+f
        aux = os.path.join(path, f)
        if os.path.isdir(aux):
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
    with open(os.path.join(path_to_store_screenshots ,filename), 'wb') as f:
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

    res_path =os.path.join( root_file_path,output_filename + '.xes')
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

    res_path = os.path.join(root_file_path, output_filename + '.csv')

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
  
  
def get_csv_log_start_datetime(csv_file_path,pattern):
  log_csv = read_ui_log_as_dataframe(csv_file_path)
  first_timestamp = parse(log_csv["time:timestamp"].iloc[0])
  timezone = pytz.timezone("Etc/GMT")
  first_timestamp = first_timestamp.astimezone(timezone)
  first_timestamp = first_timestamp.replace(tzinfo=None) 
  return first_timestamp

def convert_timestamps_and_clean_screenshot_name_in_csv(csv_file_path):
    df = read_ui_log_as_dataframe(csv_file_path)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp']).dt.strftime('%H:%M:%S')
    df['screenshot'] = df['screenshot'].apply(lambda x: re.split(r'[\\/]+', x)[-1])
    # Save and apply this changes
    df.to_csv(csv_file_path, index=False)
    return df
    

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
