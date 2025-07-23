import pandas as pd
import os

def merge_wearable_with_ui(ui_log_path, wearable_csv_path, output_path):
    """
    Une el CSV de wearable con el UI log usando la hora y minuto (HH:MM) como clave.
    Crea un nuevo archivo con todas las columnas del UI log + las del wearable.
    El UI log original NO se modifica.
    """
    # Cargar los datos
    ui_df = pd.read_csv(ui_log_path)
    wearable_df = pd.read_csv(wearable_csv_path)

    # UI log: puede ser 'time:timestamp' o 'timestamp'
    if 'timestamp' not in ui_df.columns and 'time:timestamp' in ui_df.columns:
        ui_df['timestamp'] = ui_df['time:timestamp']

    # Crear columna solo con hora y minuto en ambos DataFrames
    ui_df['timestamp_min'] = pd.to_datetime(ui_df['timestamp'], errors='coerce').dt.strftime('%H:%M')
    wearable_df['timestamp_min'] = pd.to_datetime(wearable_df['timestamp'], errors='coerce').dt.strftime('%H:%M')

    # Merge por minuto
    merged_df = pd.merge(
        ui_df,
        wearable_df.drop(columns=['timestamp']),
        on='timestamp_min',
        how='left',
        suffixes=('_ui', '_wearable')
    )
    merged_df.drop(columns=['timestamp_min'], inplace=True)

    # Guardar el resultado en un nuevo archivo
    merged_df.to_csv(output_path, index=False)
    return output_path