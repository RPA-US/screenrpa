import pandas as pd
import logging
import os
from datetime import datetime

def merge_emotions_with_process_discovery(emotions_csv_path, pd_log_csv_path, output_path):
    """
    Une los datos de emociones con el CSV de process discovery (pd_log.csv) por timestamp.
    Conserva la fecha (año/mes/día) de emociones.csv y la hora/minutos/segundos de pd_log.csv.
    
    Args:
        emotions_csv_path (str): Ruta al CSV de emociones
        pd_log_csv_path (str): Ruta al CSV de process discovery (pd_log.csv)  
        output_path (str): Ruta donde guardar el CSV combinado
    """
    try:
        print(f"Mergeando emociones con process discovery...")
        
        # Verificar que los archivos existen
        if not os.path.exists(emotions_csv_path):
            print(f"ERROR: Archivo de emociones no existe: {emotions_csv_path}")
            return False
            
        if not os.path.exists(pd_log_csv_path):
            print(f"ERROR: Archivo de process discovery no existe: {pd_log_csv_path}")
            return False
        
        # Leer los archivos CSV
        emotions_df = pd.read_csv(emotions_csv_path)
        pd_log_df = pd.read_csv(pd_log_csv_path)
        
        print(f"Emociones: {len(emotions_df)} registros, PD: {len(pd_log_df)} registros")
        
        # Determinar columnas de timestamp
        emotions_timestamp_col = 'fecha_hora' if 'fecha_hora' in emotions_df.columns else 'timestamp'
        pd_timestamp_col = 'time:timestamp' if 'time:timestamp' in pd_log_df.columns else 'timestamp'
        
        if emotions_timestamp_col not in emotions_df.columns:
            print(f"ERROR: No hay columna de timestamp en emociones. Columnas: {list(emotions_df.columns)}")
            return False
            
        if pd_timestamp_col not in pd_log_df.columns:
            print(f"ERROR: No hay columna de timestamp en PD. Columnas: {list(pd_log_df.columns)}")
            return False
        
        # Convertir timestamps a datetime
        emotions_df['timestamp_normalized'] = pd.to_datetime(emotions_df[emotions_timestamp_col], utc=False).dt.tz_localize(None)
        pd_log_df['timestamp_normalized'] = pd.to_datetime(pd_log_df[pd_timestamp_col], utc=False).dt.tz_localize(None)
        
        # Extraer la fecha de emociones y horas de pd_log
        emotions_date = emotions_df['timestamp_normalized'].dt.date.iloc[0]
        print(f"Usando fecha de emociones: {emotions_date}")
        
        # Reemplazar fecha en pd_log manteniendo hora original
        pd_log_df['timestamp_normalized'] = pd_log_df['timestamp_normalized'].apply(
            lambda x: datetime.combine(emotions_date, x.time())
        )
        
        # Actualizar la columna original de timestamp en pd_log
        pd_log_df[pd_timestamp_col] = pd_log_df['timestamp_normalized'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Identificar columnas de emociones
        emotion_columns = ['emocion', 'sentimiento']
        available_emotion_columns = [col for col in emotion_columns if col in emotions_df.columns]
        
        if not available_emotion_columns:
            available_emotion_columns = [col for col in emotions_df.columns 
                                        if col not in [emotions_timestamp_col, 'timestamp_normalized']]
            
        merge_columns = ['timestamp_normalized'] + available_emotion_columns
        
        # Ordenar por timestamp
        emotions_df = emotions_df.sort_values('timestamp_normalized')
        pd_log_df = pd_log_df.sort_values('timestamp_normalized')
        
        # Merge usando las columnas normalizadas
        merged_df = pd.merge_asof(
            pd_log_df.sort_values('timestamp_normalized'),
            emotions_df[merge_columns].sort_values('timestamp_normalized'),
            on='timestamp_normalized',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=30),
            suffixes=('', '_emotion')
        )
        
        # Eliminar la columna temporal
        merged_df.drop(columns=['timestamp_normalized'], inplace=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        
        emotion_matches = len(merged_df[merged_df[available_emotion_columns[0]].notna()]) if available_emotion_columns else 0
        
        print(f"SUCCESS: Guardado {len(merged_df)} registros, {emotion_matches} con emociones en {output_path}")
        logging.info(f"Archivo combinado process discovery-emociones guardado: {output_path} con {len(merged_df)} registros")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        logging.error(f"Error al combinar datos de process discovery y emociones: {str(e)}")
        return False