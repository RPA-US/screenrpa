import os
import pandas as pd
from core.settings import gaze_analysis_threshold
import urllib.parse
import json
from dateutil.parser import parse
from lxml import html
import email
import re
import pandas as pd
import datetime
from dateutil import tz


def get_seconds(startDateTime_ui_log, current_timestamp):
  return (startDateTime_ui_log - datetime.datetime.strptime(current_timestamp, '%H:%M:%S')).total_seconds()

def gaze_analyzer(ui_log, gaze_log, special_colnames, startDateTime_ui_log, startDateTime_gaze_tz):
  startDateTime_gaze = startDateTime_gaze_tz.replace(tzinfo=None)

  ui_log_timedelta=0
  gaze_log_timedelta=0

  if startDateTime_ui_log > startDateTime_gaze:
    ui_log_timedelta = (startDateTime_ui_log - startDateTime_gaze).total_seconds()
  elif startDateTime_ui_log > startDateTime_gaze:
    gaze_log_timedelta = (startDateTime_gaze - startDateTime_ui_log).total_seconds()
  else:
    print("==> Gaze Log and UI Log sinchronized!")


  # ordenar los dataframes por timestamp
  ui_log = ui_log.sort_values(special_colnames["Timestamp"])
  gaze_log = gaze_log.sort_values("Timestamp")
  # https://imotions.com/release-notes/imotions-9-1-7/
  # We added a new channel to LSL data called “LSL Timestamp”. This timestamp is 
  # calculated by the LSL library. Each sample is timestamped in seconds, representing
  # the relative elapsed time since system boot-up or 1-1-1970

  fixation_points = {}

  last_gaze_log_row = 0
  last_fixation_index = 0

  # recorrer cada fila del UI log
  for j in range(len(ui_log)-1):
      # obtener el timestamp actual y el siguiente del UI log
      current_timestamp = ui_log.iloc[j][special_colnames["Timestamp"]]
      current_timestamp = get_seconds(startDateTime_ui_log, current_timestamp) + ui_log_timedelta
      next_timestamp = ui_log.iloc[j+1][special_colnames["Timestamp"]]
      next_timestamp = get_seconds(startDateTime_ui_log, next_timestamp) + ui_log_timedelta
      

      fixation_points[ui_log.iloc[j]["screenshot"]] = { 'fixation_points': {} }
      key = None
      
      for i in range(last_gaze_log_row, len(gaze_log)-1):
        print(gaze_log.iloc[i]["Timestamp"])

        last_fixation_index = gaze_log.iloc[i]["Fixation Index by Stimulus"]
        if key and gaze_log.iloc[i]["Fixation Index by Stimulus"] == last_fixation_index:
          fixation_points[ui_log.iloc[j]["screenshot"]]["fixation_points"][key] += 1
        else:
          if gaze_log.iloc[i]["Fixation Start"] and (float(gaze_log.iloc[i]["Fixation Start"]) + gaze_log_timedelta) >= current_timestamp:
            # datetime.datetime.fromtimestamp(gaze_log.iloc[i]["Fixation Start"], tzinfo=startDateTime_gaze_tz.astimezone(tz.UTC))
            key = str(gaze_log["Fixation X"] + gaze_log_timedelta) + "," + str(gaze_log["Fixation Y"] + gaze_log_timedelta)
            fixation_points[ui_log.iloc[j]["screenshot"]]["fixation_points"][key] = 1
          else:
            print("No fixation point in " + str(gaze_log.iloc[i]["RowNumber"]))


        if float(gaze_log.iloc[i]["Timestamp"]) >= next_timestamp:
          last_gaze_log_row = i
          break
  return fixation_points


def get_mht_log_start_datetime(mht_file_path):
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
        raise Exception("The MHT file doesnt have '(datetime)' after 'Step 1:'")

    return datetime.datetime.strptime(datetime_parenthesis, '\u200e%d/\u200e%m/\u200e%Y %H:%M:%S')

def gaze_analysis(log_path, root_path, special_colnames, gaze_analysis_type, gaze_analysis_configurations):
    # eyetracking_log , log, img_index, image_names, special_colnames, timestamp_start, timestamp_end, last_upper_limit, init_value_ui_log_timestamp):
    
    ui_log = pd.read_csv(log_path, sep=",")
    
    sep = gaze_analysis_configurations["separator"]
    
    eyetracking_log_filename = gaze_analysis_configurations["eyetracking_log_filename"]
    uilog_starttime = float(gaze_analysis_configurations["uilog_startdatetime"])
    
    if eyetracking_log_filename and os.path.exists(root_path + eyetracking_log_filename):
        gazeanalysis_log = pd.read_csv(root_path + eyetracking_log_filename, sep=sep)
    else:
        raise Exception("Eyetracking log cannot be read")

    if gaze_analysis_type == "imotions":
        gaze_log, metadata = decode_imotions_gaze_analysis(gazeanalysis_log)
        startDateTime_gaze_tz = decode_imotions_native_slideevents(root_path, gaze_analysis_configurations["native_slide_events"], sep)
        startDateTime_ui_log = get_mht_log_start_datetime(root_path + gaze_analysis_configurations["mht_log_filename"])

        fixation_p = gaze_analyzer(ui_log, gaze_log, special_colnames, startDateTime_ui_log, startDateTime_gaze_tz)
        
        # Serializing json
        json_object = json.dumps(fixation_p, indent=4)
        with open(root_path + "fixation.json", "w") as outfile:
            outfile.write(json_object)
    else:
        raise Exception("You select a gaze analysis that is not available in the system")
        
    return root_path + "fixation.json"


# def gaze_events_associated_to_event_time_range(eyetracking_log, special_colnames, timestamp_start, timestamp_end, last_upper_limit):
#     # timestamp starts from 0
    
#     eyetracking_log_timestamp = eyetracking_log[special_colnames['eyetracking_timestamp']]
#     if eyetracking_log_timestamp[0] != 0:
#         raise ValidationError(
#             "Recording timestamp in eyetracking log must starts from 0")    

#     lower_limit = 0
#     upper_limit = 0

#     if timestamp_end != "LAST":
#         for index, time in enumerate(eyetracking_log_timestamp):
#             if time > timestamp_start:
#                 lower_limit = eyetracking_log_timestamp[index-1]
#                 if time >= timestamp_end:
#                     upper_limit = eyetracking_log_timestamp[index]
#                     break
#     else:
#         upper_limit = len(eyetracking_log_timestamp)
#         lower_limit = last_upper_limit

#     return eyetracking_log.loc[lower_limit:upper_limit, [special_colnames['eyetracking_gaze_point_x'], special_colnames['eyetracking_gaze_point_y']]], upper_limit


def decode_imotions_gaze_analysis(gazeanalysis_log):
    # Find row index where "#DATA" is located
    data_index = gazeanalysis_log.index[gazeanalysis_log.iloc[:, 0] == '#DATA'][0]

    # Split dataframe into two separate dataframes
    metadata_df = gazeanalysis_log.iloc[:data_index, :]
    data_df = gazeanalysis_log.iloc[data_index+1:, :]


    # Set the headers of data_df
    headers = data_df.iloc[0]
    data_df = data_df[1:].rename(columns=headers).reset_index(drop=True)
    
    return data_df, metadata_df

def decode_imotions_native_slideevents(native_slideevents_path, native_slideevents_filename, sep):
    # Leer el archivo completo
    with open(native_slideevents_path + native_slideevents_filename, 'r') as file:
        data = file.readlines()
    # Encontrar los índices donde están las etiquetas #METADATA y #DATA
    metadata_index = data.index('#METADATA\n')
    encodedStr = data[metadata_index-1:metadata_index][0]
    native_properties= urllib.parse.unquote(encodedStr)
    native_properties= json.loads(native_properties)
    with open(native_slideevents_path + "native_properties.json", 'w') as f:
            json.dump(native_properties, f, indent=4)
    # data_index = data.index('#DATA\n')
    # # Crear dos listas separadas para los metadatos y los datos
    # metadata = [line.strip().split(sep) for line in data[metadata_index+1:data_index]]
    # data_csv = [line.strip().split(sep) for line in data[data_index+1:]]
    # # Convertir la lista de datos en un dataframe de pandas
    # df = pd.DataFrame(data_csv[1:], columns=data_csv[0])
    return parse(native_properties["SlideShowStartDateTime"])