import os
from core.settings import CDLR
import urllib.parse
import json
import pytz
from dateutil.parser import parse

def decode_imotions_monitoring(gazeanalysis_log):
    # Find row index where "#DATA" is located
    data_index = gazeanalysis_log.index[gazeanalysis_log.iloc[:, 0] == '#DATA'][0]

    # Split dataframe into two separate dataframes
    metadata_df = gazeanalysis_log.iloc[:data_index, :]
    data_df = gazeanalysis_log.iloc[data_index+1:, :]


    # Set the headers of data_df
    headers = data_df.iloc[0]
    data_df = data_df[1:].rename(columns=headers).reset_index(drop=True)
    
    return data_df, metadata_df #Aqui devuelve dos dataframes: Uno que es el data_df que es el que contiene información de las fijaciones 
#y otro que es el metadata_df que contiene información de la prueba que no nos interesa.

def decode_imotions_native_slideevents(native_slideevents_path, native_slideevents_filename):
    # Extracción de la fecha y hora de inicio: Extrae la fecha y hora de inicio  (SlideShowStartDateTime) 
    # de los metadatos y la convierte en un objeto datetime.

    # Extracción y ajuste de la zona horaria: Extrae la zona horaria de los metadatos, la formatea 
    # y la usa para ajustar la fecha y hora de inicio  a la zona horaria correcta.

    # Retorno de la fecha y hora de inicio ajustada: Devuelve la fecha y hora de inicio ajustada a la zona horaria correcta
    # Leer el archivo completo
    with open(os.path.join(native_slideevents_path,native_slideevents_filename), 'r') as file:
        data = file.readlines()
    # Encontrar los índices donde están las etiquetas #METADATA y #DATA
    metadata_index = data.index('#METADATA\n')
    encodedStr = data[metadata_index-1:metadata_index][0]
    native_properties= urllib.parse.unquote(encodedStr)
    native_properties= json.loads(native_properties)
    with open(os.path.join(native_slideevents_path,"native_properties.json"), 'w') as f:
            json.dump(native_properties, f, indent=4)
    res = parse(native_properties["SlideShowStartDateTime"])
    timezone_unformatted = native_properties["TimeZone"].split(";")[0].replace("+", " ") 
    f = open(CDLR)
    cdlr = json.load(f)
    timezone_name = cdlr[timezone_unformatted]
    timezone = pytz.timezone(timezone_name)
    res = res.astimezone(timezone)
    
    return res


#Tengo que convertir el formado del UTC del json que obtengo de de fixations_updated_centroids a un formato astimezone.
#FOR WEBGAZER AND TOBII
def decode_timezone(native_slideevents_path, native_slide_events):
    with open(os.path.join(native_slideevents_path , native_slide_events), 'r') as file:
        data = json.load(file)  
    res = parse(data["SlideShowStartDateTime"])
    timezone = pytz.timezone(data["TimeZone"])
    res = res.astimezone(timezone)
    res = res.replace(tzinfo=None)
    return res
