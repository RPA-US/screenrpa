import pandas as pd
import os
import re

def merge_wearable_with_ui(ui_log_path, wearable_csv_path, output_path):
    """
    Une el CSV de wearable con el UI log usando la hora y minuto (HH:MM) como clave.
    Crea un nuevo archivo con todas las columnas del UI log + las del wearable.
    El UI log original NO se modifica.
    """
    # Cargar los datoss
    ui_df = pd.read_csv(ui_log_path)
    wearable_df = pd.read_csv(wearable_csv_path)

    # UI log: puede ser 'time:timestamp' o 'timestamp'
    if 'timestamp' not in ui_df.columns and 'time:timestamp' in ui_df.columns:
        ui_df['timestamp'] = ui_df['time:timestamp']

    # Extraer la fecha del nombre del archivo de captura de pantalla
    date_from_screenshot = None
    if 'screenshot' in ui_df.columns:
        for screenshot in ui_df['screenshot'].dropna():
            if isinstance(screenshot, str):
                # Buscar un patrón de fecha en el nombre del archivo (YYYY-MM-DD)
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', screenshot)
                if date_match:
                    date_from_screenshot = date_match.group(1)
                    break

    # Crear columna solo con hora y minuto en ambos DataFrames para el merge
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

    # Si encontramos la fecha en las capturas, añadir la fecha completa a las columnas de timestamp
    if date_from_screenshot:
        # Crear columna timestamp_full con la fecha completa + hora
        merged_df['timestamp_full'] = merged_df['timestamp'].apply(
            lambda x: f"{date_from_screenshot} {x}" if isinstance(x, str) and len(x) <= 8 else x
        )
        
        # Guardar como columna adicional y no reemplazar la original
        merged_df['csv_date'] = date_from_screenshot

    # Guardar el resultado en un nuevo archivo
    merged_df.to_csv(output_path, index=False)
    return output_path