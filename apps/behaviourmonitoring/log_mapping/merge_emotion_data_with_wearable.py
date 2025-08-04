import pandas as pd
import logging
from datetime import datetime

def merge_emotions_with_wearable(emotions_csv_path, wearable_csv_path, output_path):
    """
    Une los datos de emociones con los datos de wearables por timestamp.
    
    Args:
        emotions_csv_path (str): Ruta al CSV de emociones
        wearable_csv_path (str): Ruta al CSV de wearables  
        output_path (str): Ruta donde guardar el CSV combinado
    """
    try:
        # Leer los archivos CSV
        emotions_df = pd.read_csv(emotions_csv_path)
        wearable_df = pd.read_csv(wearable_csv_path)
        
        logging.info(f"Cargando datos de emociones: {len(emotions_df)} registros")
        logging.info(f"Cargando datos de wearables: {len(wearable_df)} registros")
        
        # Convertir timestamps a datetime si no lo están ya
        if 'fecha_hora' in emotions_df.columns:
            emotions_df['timestamp'] = pd.to_datetime(emotions_df['fecha_hora'])
        elif 'timestamp' in emotions_df.columns:
            emotions_df['timestamp'] = pd.to_datetime(emotions_df['timestamp'])
        else:
            logging.error("No se encontró columna de timestamp en datos de emociones")
            return False
            
        if 'timestamp' in wearable_df.columns:
            wearable_df['timestamp'] = pd.to_datetime(wearable_df['timestamp'])
        else:
            logging.error("No se encontró columna de timestamp en datos de wearables")
            return False
        
        # Ordenar por timestamp
        emotions_df = emotions_df.sort_values('timestamp')
        wearable_df = wearable_df.sort_values('timestamp')
        
        # Realizar merge por timestamp (outer join para mantener todos los datos)
        merged_df = pd.merge_asof(
            emotions_df.sort_values('timestamp'),
            wearable_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=30),  # Tolerancia de 30 segundos
            suffixes=('_emotion', '_wearable')
        )
        
        # Guardar el archivo combinado
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Archivo combinado emociones-wearable guardado: {output_path} con {len(merged_df)} registros")
        
        return True
        
    except Exception as e:
        logging.error(f"Error al combinar datos de emociones y wearables: {str(e)}")
        return False