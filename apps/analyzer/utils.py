import json
import os
from core.settings import FE_EXTRACTORS_FILEPATH
from apps.featureextraction.feature_extraction_techniques import *

import email
import lxml.etree as ET
from lxml import html
import base64
import pandas as pd
from django.shortcuts import render

def detect_fe_function(text):
    '''
    Selecting a function in the system by means of a keyword
    args:
        text: function to be detected
    '''
    # Search the function by key in the json
    f = open(FE_EXTRACTORS_FILEPATH)
    json_func = json.load(f)
    return eval(json_func[text])

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
      raise Exception("MIME Html format not contains Content-Location header in screenshots")
    image_data = part.get_payload()

    # Decode the image data from base64 encoding
    image_data_decoded = base64.b64decode(image_data)

    if not os.path.exists(path_to_store_screenshots):
        os.mkdir(path_to_store_screenshots)

    # Save the image to a file
    with open(path_to_store_screenshots + filename, 'wb') as f:
        f.write(image_data_decoded)
        print(f"Saved image file {filename}")

def from_html_to_xes(myhtml, root_file_path, output_filename):
    root = html.fromstring(myhtml)
    myxml = root.xpath("//script[@id='myXML']")[0].text_content()
    myxml = myxml.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
    
    my_xml_doc = ET.fromstring(myxml)

    # Create an XES document
    xes = ET.Element('log', {'xes.version': '2.0'})
    trace = ET.SubElement(xes, 'trace')
    ET.SubElement(trace, 'string', {'key': 'concept:name'}).text = 'Example event'
    ET.SubElement(trace, 'string', {'key': 'description'}).text = 'Process instance from MHT file'

    recorded_session = my_xml_doc.find("UserActionData").find("RecordSession")
    for each_action in recorded_session.getchildren():
      event = ET.SubElement(trace, 'event')
      children = each_action.getchildren()
      ET.SubElement(event, 'string', {'key': 'org:resource'}).text = 'User1'
      ET.SubElement(event, 'string', {'key': 'description'}).text = children[0].text
      ET.SubElement(event, 'string', {'key': 'concept:name'}).text = children[1].text
      coords = children[2].text.strip().split(",")
      ET.SubElement(event, 'float', {'key': 'coorX'}).text = coords[0]
      ET.SubElement(event, 'float', {'key': 'coorY'}).text = coords[1]
      screen_coords = children[3].text.strip().split(",")
      ET.SubElement(event, 'float', {'key': 'screenW'}).text = screen_coords[2]
      ET.SubElement(event, 'float', {'key': 'screenH'}).text = screen_coords[3]
      if len(children) > 5:
        ET.SubElement(event, 'string', {'key': 'screenshot'}).text = children[5].text
      else:
        ET.SubElement(event, 'string', {'key': 'screenshot'}).text = "None"
      ET.SubElement(event, 'date', {'key': 'time:timestamp'}).text = each_action.get("Time")
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
        

def from_html_to_csv(myhtml, root_file_path, output_filename):
    root = html.fromstring(myhtml)
    myxml = root.xpath("//script[@id='myXML']")[0].text_content()
    myxml = myxml.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
    
    my_xml_doc = ET.fromstring(myxml)

    events = []
    recorded_session = my_xml_doc.find("UserActionData").find("RecordSession")
    for each_action in recorded_session.getchildren():
        event = {}
        event['org:resource'] = 'User1'
        event['description'] = each_action[0].text
        event['concept:name'] = each_action[1].text
        coords = each_action[2].text.strip().split(",")
        event['coorX'] = coords[0]
        event['coorY'] = coords[1]
        screen_coords = each_action[3].text.strip().split(",")
        event['screenW'] = screen_coords[2]
        event['screenH'] = screen_coords[3]
        if len(each_action) > 5:
            event['screenshot'] = each_action[5].text
        else:
            event['screenshot'] = "None"
        event['time:timestamp'] = each_action.get("Time")
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

def format_mht_file(mht_file_path, output_format, output_path, output_filename):
  with open(mht_file_path) as mht_file: 
    msg = email.message_from_file(mht_file)
    myhtml = msg.get_payload()[0].get_payload()
    
  store_screenshots(msg.get_payload(), output_path)
  
  if output_format == "mht_xes":
    res_path = from_html_to_xes(myhtml, output_path, output_filename)
  elif output_format == "mht_csv":
    res_path = from_html_to_csv(myhtml, output_path, output_filename)
  else:
    raise Exception("You select a format mht file that doesnt exists")
  
  return res_path