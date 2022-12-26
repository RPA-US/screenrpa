import json
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