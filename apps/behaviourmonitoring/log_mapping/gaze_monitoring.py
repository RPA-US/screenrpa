import os
import logging
import pandas as pd
from core.settings import gaze_monitoring_threshold, CDLR
import json
import pandas as pd
import datetime
from dateutil import tz
from apps.analyzer.utils import get_mht_log_start_datetime
from apps.analyzer.utils import format_mht_file
from apps.behaviourmonitoring.log_mapping.eyetracker_log_decoders import decode_imotions_monitoring, decode_imotions_native_slideevents

import math

def euclidean_distance(x1, y1, x2, y2):
  """
  Define a function to calculate the Euclidean distance between two points
  """
  return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def fixation_dispersion(fixations, gaze_log, x_column_name="Gaze X", y_column_name="Gaze Y"):
    """
    Define a function to calculate the fixation dispersion given a list of fixations
    """
    
    # Calculate the mean x and y coordinates of the fixations
    total_x = 0
    total_y = 0
    for index in fixations:
        total_x += float(gaze_log.loc[index,x_column_name])
        total_y += float(gaze_log.loc[index,y_column_name])
    mean_x = total_x / len(fixations)
    mean_y = total_y / len(fixations)

    # Calculate the sum of squared distances from each fixation to the mean fixation
    ssd = sum(euclidean_distance(float(gaze_log.loc[index,x_column_name]), float(gaze_log.loc[index,y_column_name]), mean_x, mean_y)**2 for index in fixations)

    # Divide the sum of squared distances by the number of fixations minus 1 to get the variance
    variance = ssd / (len(fixations) - 1)

    # Take the square root of the variance to get the standard deviation, which is the fixation dispersion
    fixation_dispersion = math.sqrt(variance)

    return fixation_dispersion
  
def calculate_dispersion(gaze_log, metrics, last_index):
    dispersion = fixation_dispersion(range(metrics["start_index"], last_index + 1), gaze_log, "Gaze X", "Gaze Y")
    
    metrics["last_index"] = last_index
    metrics["dispersion"] = dispersion
    return metrics

def get_timestamp(time_begining, start_datetime, current_timestamp, pattern):
  if pattern == "ms":
    ms = float(current_timestamp) / 1000
    min = int(ms/60)
    hs = int(min/60)
    ms -= min*60
    min -= hs*60
    # Datetime can read 6 digits as miliseconds
    if len(str(float(current_timestamp) / 1000)) > 6:
      ms = str(ms)[:6]
    ms = str(hs)+"-"+str(min)+"-"+str(ms)
    time = datetime.datetime.strptime(ms, '%H-%M-%S.%f').time()
    res = start_datetime + datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond)
  else:
    time =  datetime.datetime.strptime(current_timestamp, pattern).time()
    res = datetime.datetime.combine(start_datetime, time)
  return (res - time_begining).total_seconds()


def format_fixation_point_key(i, gaze_log):
  return str(gaze_log.iloc[i]["Fixation X"]) + "#" + str(gaze_log.iloc[i]["Fixation Y"])

def gaze_log_get_key(i, gaze_log):
  init = {  
    "#events": 1,
    "start_index": i,
    "ms_start": gaze_log.iloc[i]["Fixation Start"],
    "ms_end": gaze_log.iloc[i]["Fixation End"],
    "duration": gaze_log.iloc[i]["Fixation Duration"],
    "imotions_dispersion": gaze_log.iloc[i]["Fixation Dispersion"]
  }
  return format_fixation_point_key(i, gaze_log), init

def update_previous_screenshots_in_splitted_events(fixation_points, j, key, init, ui_log, last_counter, special_colnames):
  for k in range(1, last_counter+1):
    if not (ui_log.iloc[j-k][special_colnames["Screenshot"]] in fixation_points):
      # Initialize gaze metrics
      fixation_points[ui_log.iloc[j-k][special_colnames["Screenshot"]]] = { 'fixation_points': { key: init} }
    elif key in fixation_points[ui_log.iloc[j-k][special_colnames["Screenshot"]]]["fixation_points"]:
      fixation_points[ui_log.iloc[j-k][special_colnames["Screenshot"]]]["fixation_points"][key]["#events"] += 1
    else:
      # Initialize gaze metrics
      fixation_points[ui_log.iloc[j-k][special_colnames["Screenshot"]]]["fixation_points"][key] = init
    fixation_points[ui_log.iloc[j-k][special_colnames["Screenshot"]]]["fixation_points"][key]["intersectioned"] = "True"
  
  return fixation_points 

def update_fixation_points(j, i, key, fixation_points, gaze_log, ui_log, last_fixation_index, last_gaze_log_row, last_fixation_index_row, last_ui_log_index_row, starting_point, initial_timestamp, current_timestamp, startDateTime_ui_log, startDateTime_gaze_tz, special_colnames):
  """
  
  Update fixation points values
  
  j: ui_log index
  i: gaze_log index
  fixation_points
  gaze_log: gaze_log content
  ui_log: ui_log content
  last_fixation_index: fixation index of the last row
  last_gaze_log_row: last gaze log row 
  last_fixation_index_row: gaze row index of the last fixation point detected
  last_ui_log_index_row: ui log index that point to last fixation related event
  starting_point: the start time to measure timestamps
  initial_timestamp: first timestamp in the log
  current_timestamp: current timestamp
  startDateTime_ui_log: start datetime in ui log
  startDateTime_gaze_tz: start datetime in gaze log
  special_colnames: json indicating special columns names
  """
  gaze_fixation_start = None
  
  # When a fixation slot ends, dispersion is calculated
  if last_ui_log_index_row != -1 and (not pd.isnull(last_fixation_index)) and last_fixation_index != 0 and (not pd.isnull(gaze_log.iloc[i]["Fixation Index"])) and gaze_log.iloc[i]["Fixation Index"] != last_fixation_index:
    screenshot_name = ui_log.iloc[last_ui_log_index_row][special_colnames["Screenshot"]]
    gaze_metrics = fixation_points[screenshot_name]["fixation_points"][format_fixation_point_key(last_fixation_index_row, gaze_log)]
    metrics_aux = calculate_dispersion(gaze_log, gaze_metrics, last_fixation_index_row)
    fixation_points[screenshot_name]["fixation_points"][format_fixation_point_key(last_fixation_index_row, gaze_log)] = metrics_aux
  
  if key and (key in fixation_points[ui_log.iloc[j][special_colnames["Screenshot"]]]["fixation_points"]) and \
    gaze_log.iloc[i]["Fixation Index"] == last_fixation_index:
    fixation_points[ui_log.iloc[j][special_colnames["Screenshot"]]]["fixation_points"][key]["#events"] += 1
  # if an UI log event splits a fixation cluster:  
  # if it is the first iteration, and 
  # the row contains fixation, and 
  # dict contains fixation key, and 
  # ui log event screenshot is distinct that the screenshot of the last event
  elif (i == last_gaze_log_row) and \
    gaze_log.iloc[i]["Fixation Start"] and (not pd.isnull(gaze_log.iloc[i]["Fixation Start"])) and \
    (ui_log.iloc[j-1][special_colnames["Screenshot"]] in fixation_points and \
    "fixation_points" in fixation_points[ui_log.iloc[j-1][special_colnames["Screenshot"]]] and \
    key in fixation_points[ui_log.iloc[j-1][special_colnames["Screenshot"]]]["fixation_points"]) and \
      ui_log.iloc[j][special_colnames["Screenshot"]] != ui_log.iloc[j-1][special_colnames["Screenshot"]]:
      logging.exception("behaviourmonitoring/monitoring/update_fixation_points line:65. UI Log row " + str(j) + ". Fixation cluster splitted by two UI Log event!")
      raise Exception("Fixation cluster splitted by two UI Log event!")
  else:
    fixation_start = gaze_log.iloc[i]["Fixation Start"]
    if fixation_start and (not pd.isnull(fixation_start)):
      gaze_fixation_start = get_timestamp(starting_point, startDateTime_gaze_tz, fixation_start, 'ms') # + gaze_log_timedelta
      key, init = gaze_log_get_key(i, gaze_log)
      last_counter = 1
      aux_index = j-last_counter if j != 0 else j
      last_timestamp = get_timestamp(starting_point, startDateTime_ui_log, ui_log.iloc[aux_index][special_colnames["Timestamp"]], '%H:%M:%S')
      
      while last_timestamp == current_timestamp and j != 0:
        last_timestamp = get_timestamp(starting_point, startDateTime_ui_log, ui_log.iloc[j-last_counter][special_colnames["Timestamp"]], '%H:%M:%S')
        logging.info("behaviourmonitoring/monitoring/update_fixation_points line:76. UI Log row " + str(j) + ". Finding the last timestamp change: " + str(last_counter) + " event before")
        last_counter+=1
      
      if gaze_fixation_start >= current_timestamp:
        if key in fixation_points[ui_log.iloc[j][special_colnames["Screenshot"]]]["fixation_points"]:
          fixation_points[ui_log.iloc[j][special_colnames["Screenshot"]]]["fixation_points"][key]["#events"] += 1
        else:
          # Initialize gaze metrics
          fixation_points[ui_log.iloc[j][special_colnames["Screenshot"]]]["fixation_points"][key] = init
      elif (last_timestamp <= gaze_fixation_start) and (gaze_fixation_start < current_timestamp):
        fixation_points = update_previous_screenshots_in_splitted_events(fixation_points, j, key, init, ui_log, last_counter, special_colnames)
      else:
        logging.info("behaviourmonitoring/monitoring/update_fixation_points line:87. Gaze log event PREVIOUS to UI log capture, gaze log row number:" + str(gaze_log.iloc[i]["RowNumber"]))
        
        if key and (key in fixation_points["previous"]):
          fixation_points["previous"][key]["#events"] += 1
        else:
          # Initialize gaze metrics
          fixation_points["previous"][key] = init
    
    else:
      if gaze_log.iloc[i]["Saccade Index"] and (not pd.isnull(gaze_log.iloc[i]["Saccade Index"])):
        msg = "Row " + str(gaze_log.iloc[i]["RowNumber"] + ": Saccade movement - Index " + str(gaze_log.iloc[i]["Saccade Index"]))
      else:
        msg = "No fixation point in " + str(gaze_log.iloc[i]["RowNumber"] + ". Saccade: " + str(gaze_log.iloc[i]["Saccade Index"]))
      logging.info(msg)
      print(msg)
  
  if not pd.isnull(gaze_log.iloc[i]["Fixation Index"]):
    last_fixation_index = gaze_log.iloc[i]["Fixation Index"]
    last_fixation_index_row = i
    if gaze_fixation_start and gaze_fixation_start >= current_timestamp:
      last_ui_log_index_row = j
  
  return fixation_points, key, last_fixation_index, last_fixation_index_row, last_ui_log_index_row, last_gaze_log_row

def gaze_log_mapping(ui_log, gaze_log, special_colnames, startDateTime_ui_log, startDateTime_gaze_tz, monitoring_configurations):
  # https://imotions.com/release-notes/imotions-9-1-7/
  startDateTime_gaze_tz = startDateTime_gaze_tz.replace(tzinfo=None)

  if startDateTime_ui_log > startDateTime_gaze_tz:
    starting_point = startDateTime_gaze_tz
  elif startDateTime_ui_log < startDateTime_gaze_tz:
    starting_point = startDateTime_ui_log
  else:
    starting_point = startDateTime_gaze_tz
    logging.info("behaviourmonitoring/monitoring/gaze_log_mapping line:109. Gaze Log and UI Log already synchronized!")

  # TODO: we suppose that dataframe timestamps are ordered
  # ui_log = ui_log.sort_values(special_colnames["Timestamp"])
  # gaze_log = gaze_log.sort_values("Timestamp")

  fixation_points = {"previous": {}, "subsequent": {} }
  
  last_gaze_log_row = 0
  last_fixation_index = 0
  last_fixation_index_row = -1
  last_ui_log_index_row = -1
  
  initial_timestamp = get_timestamp(starting_point, startDateTime_ui_log, ui_log.iloc[0][special_colnames["Timestamp"]], '%H:%M:%S')
  
  # Loop: Each UI Log row
  for j in range(len(ui_log)-1):
      # Obtain current event timestamp and next event timestamp 
      current_timestamp = ui_log.iloc[j][special_colnames["Timestamp"]]
      current_timestamp = get_timestamp(starting_point, startDateTime_ui_log, current_timestamp, '%H:%M:%S')# + ui_log_timedelta
      next_timestamp = ui_log.iloc[j+1][special_colnames["Timestamp"]]
      next_timestamp = get_timestamp(starting_point, startDateTime_ui_log, next_timestamp, '%H:%M:%S')# + ui_log_timedelta
      
      if next_timestamp > current_timestamp:
        fixation_points[ui_log.iloc[j][special_colnames["Screenshot"]]] = { 'fixation_points': {} }
        key = None
        
        for i in range(last_gaze_log_row, len(gaze_log)-1):
          gaze_timestamp = get_timestamp(starting_point, startDateTime_gaze_tz, gaze_log.iloc[i]["Timestamp"], "ms")# + gaze_log_timedelta
          
          # Gaze Event between current ui log event and next ui log event
          if gaze_timestamp < next_timestamp:
            fixation_points, key, last_fixation_index, last_fixation_index_row, last_ui_log_index_row, last_gaze_log_row = update_fixation_points(j, i, key, fixation_points, gaze_log, ui_log, last_fixation_index, last_gaze_log_row, last_fixation_index_row, last_ui_log_index_row, starting_point, initial_timestamp, current_timestamp, startDateTime_ui_log, startDateTime_gaze_tz, special_colnames)
            
          # Gaze Event before current ui log event
          # elif current_timestamp > gaze_timestamp:
          #   raise Exception("current_timestamp > gaze_timestamp")
          # Gaze Event after current ui log event and next ui log event
          else:
            last_gaze_log_row = i
            break
        # If all gaze log events have been covered: break
        if i == len(gaze_log)-2:
          break
      elif next_timestamp < current_timestamp:
        logging.exception("behaviourmonitoring/monitoring/gaze_log_mapping line:152. UI and Gaze Logs Timestamps are not well synchronized, next_timestamp (row " + str(j+1) + ") < current_timestamp (row " + str(j) + "): UI Log Current Screenshot " + ui_log.iloc[j][special_colnames["Screenshot"]])
        raise Exception("UI and Gaze Logs Timestamps are not well synchronized")
      else:
        logging.info("behaviourmonitoring/monitoring/gaze_log_mapping line:155. UI Logs events with the same timestamps: next_timestamp (row " + str(j+1) + ") == current_timestamp (row " + str(j) + ")")
  
  last_ui_log_timestamp = get_timestamp(starting_point, startDateTime_ui_log, ui_log.iloc[len(ui_log)-1][special_colnames["Timestamp"]], '%H:%M:%S')        
  last_gaze_timestamp = get_timestamp(starting_point, startDateTime_gaze_tz, gaze_log.iloc[len(gaze_log)-1]["Timestamp"], "ms")# + gaze_log_timedelta
  
  # Store gaze logs that takes place after last UI log event
  if last_gaze_timestamp > last_ui_log_timestamp:
    logging.info("behaviourmonitoring/monitoring/gaze_log_mapping line:161. Gaze log events after UI Log last event")
    for i in range(last_gaze_log_row, len(gaze_log)-1):
      key, init = gaze_log_get_key(i, gaze_log)
      if key and (key in fixation_points["subsequent"]):
        fixation_points["subsequent"][key]["#events"] += 1
      else:  
        fixation_points["subsequent"][key] = init
      
  return fixation_points

def monitoring(log_path, root_path, special_colnames, monitoring_type, monitoring_configurations):
    
    if os.path.exists(log_path):
      logging.info("apps/behaviourmonitoring/log_mapping/gaze_monitoring.py Log already exists, it's not needed to execute format conversor")
      print("Log already exists, it's not needed to execute format conversor")
    elif "format" in monitoring_configurations:
      log_filename = "log"
      log_path = format_mht_file(root_path + monitoring_configurations["mht_log_filename"], monitoring_configurations["format"], root_path, log_filename, monitoring_configurations["org:resource"])
  
  
    ui_log = pd.read_csv(log_path, sep=",")
    sep = monitoring_configurations["separator"]
    eyetracking_log_filename = monitoring_configurations["eyetracking_log_filename"]
    
    if eyetracking_log_filename and os.path.exists(root_path + eyetracking_log_filename):
        gazeanalysis_log = pd.read_csv(root_path + eyetracking_log_filename, sep=sep)
    else:
        logging.exception("behaviourmonitoring/monitoring/monitoring line:180. Eyetracking log cannot be read: " + root_path + eyetracking_log_filename)
        raise Exception("Eyetracking log cannot be read: " + root_path + eyetracking_log_filename)

    if monitoring_type == "imotions":
        gaze_log, metadata = decode_imotions_monitoring(gazeanalysis_log)
        startDateTime_gaze_tz = decode_imotions_native_slideevents(root_path, monitoring_configurations["native_slide_events"], sep)
        startDateTime_ui_log = get_mht_log_start_datetime(root_path + monitoring_configurations["mht_log_filename"])

        fixation_p = gaze_log_mapping(ui_log, gaze_log, special_colnames, startDateTime_ui_log, startDateTime_gaze_tz, monitoring_configurations)
        
        # Serializing json
        json_object = json.dumps(fixation_p, indent=4)
        with open(root_path + "fixation.json", "w") as outfile:
            outfile.write(json_object)
        logging.info("behaviourmonitoring/monitoring/monitoring. fixation.json saved!")
        
    else:
        logging.exception("behaviourmonitoring/monitoring/monitoring line:195. Gaze analysis selected is not available in the system")
        raise Exception("You select a gaze analysis that is not available in the system")
        
    return root_path + "fixation.json"