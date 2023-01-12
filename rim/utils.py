import json
import os
from rim.settings import FE_EXTRACTORS_FILEPATH
from featureextraction.feature_extraction_techniques import *

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