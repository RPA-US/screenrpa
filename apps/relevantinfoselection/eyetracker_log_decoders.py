from core.settings import CDLR
import urllib.parse
import json
import pytz
from dateutil.parser import parse

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
    res = parse(native_properties["SlideShowStartDateTime"])
    timezone_unformatted = native_properties["TimeZone"].split(";")[0].replace("+", " ")
    f = open(CDLR)
    cdlr = json.load(f)
    timezone_name = cdlr[timezone_unformatted]
    timezone = pytz.timezone(timezone_name)
    res = res.astimezone(timezone)
    
    return res