import pandas as pd
import json
from .settings import MODELS_CLASSES_FILEPATH

def detect_separator(file_path):
    # Leer las primeras líneas del archivo
    with open(file_path, 'r') as file:
        first_line = file.readline()
   
    # Determinar el separador basado en la primera línea
    if ',' in first_line and ';' in first_line:
        raise ValueError("Ambos separadores ',' y ';' están presentes en la primera línea. No se puede determinar el separador automáticamente.")
    elif ',' in first_line:
        separator = ','
    elif ';' in first_line:
        separator = ';'
    else:
        raise ValueError("No se pudo determinar el separador. Asegúrese de que el archivo tenga un separador ',' o ';'.")
   
    return separator
 
def read_ui_log_as_dataframe(log_path):
  # Reconoce automaticamente si los separadores son coma o punto y coma, en el dataframe
  separator = detect_separator(log_path)
  return pd.read_csv(log_path, sep=separator)#, index_col=0)

def get_model_classes(execution):
    f = open(MODELS_CLASSES_FILEPATH)
    json_func = json.load(f)
    model_name = execution.ui_elements_detection.type
    return json_func[model_name]
     