import pandas as pd
import json
from .settings import MODELS_CLASSES_FILEPATH

def read_ui_log_as_dataframe(log_path):
  return pd.read_csv(log_path, sep=",")#, index_col=0)

def get_model_classes(execution):
    f = open(MODELS_CLASSES_FILEPATH)
    json_func = json.load(f)
    model_name = execution.ui_elements_detection.type
    return json_func[model_name]
  
def get_execution_path(param_img_root):
    return param_img_root[:-1] + '_results/'
     