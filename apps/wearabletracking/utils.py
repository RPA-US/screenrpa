from datetime import datetime

import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from django.core.files.base import ContentFile
from apps.wearabletracking.models import BiometricAnalysisReport
import matplotlib
matplotlib.use('Agg')


def validar_fechas(fecha_inicio, fecha_fin):
    try:
        current_date = datetime.strptime(fecha_inicio, "%Y-%m-%d").date()
        end_date = datetime.strptime(fecha_fin, "%Y-%m-%d").date()
        if current_date > end_date:
            return "La fecha de inicio debe ser igual o anterior a la fecha de fin."
    except Exception:
        return "Las fechas introducidas no son válidas."
    return None

def obtener_ultima_sync(devices):
    fechas_sync = []
    for device in devices:
        sync_str = device.get('lastSyncTime')
        if sync_str:
            try:
                fechas_sync.append(datetime.strptime(sync_str[:10], "%Y-%m-%d").date())
            except Exception:
                pass
    return max(fechas_sync) if fechas_sync else None

def fechas_fuera_de_sync(fecha_fin, ultima_sync):
    try:
        end_date = datetime.strptime(fecha_fin, "%Y-%m-%d").date()
        if ultima_sync and end_date > ultima_sync:
            return f"La fecha de fin no puede ser posterior a la última sincronización ({ultima_sync})."
    except Exception:
        return "Error al comprobar la fecha de sincronización."
    return None


def procesar_analisis_biometrico(execution):
    """
    Procesa los datos biométricos según la configuración seleccionada.
    Busca el archivo merged_ui_wearable.csv en las ubicaciones correctas.
    """
    if not execution.biometric_config:
        raise Exception("No hay configuración biométrica activa para esta ejecución")
    
    # Obtener configuración
    config = execution.biometric_config
    print(f"Configuración biométrica: {config.title}")
    
    # Posibles ubicaciones del archivo merged_ui_wearable.csv
    merged_file_name = 'merged_ui_wearable.csv'
    possible_locations = [
        # 1. En la carpeta raíz del caso de estudio
        os.path.join(execution.case_study.exp_folder_complete_path, merged_file_name),
    ]
    
    # 2. En cada escenario del caso de estudio
    for scenario in execution.scenarios_to_study:
        possible_locations.append(os.path.join(execution.case_study.exp_folder_complete_path, scenario, merged_file_name))
    
    # 3. En la carpeta de ejecución
    possible_locations.append(os.path.join(execution.exp_folder_complete_path, merged_file_name))
    
    # Buscar el archivo en todas las ubicaciones posibles
    merged_file_path = None
    for location in possible_locations:
        if os.path.exists(location):
            merged_file_path = location
            print(f"Archivo encontrado en: {merged_file_path}")
            break
    
    if not merged_file_path:
        print(f"No se encontró el archivo merged_ui_wearable.csv")
        print(f"Ubicaciones buscadas: {possible_locations}")
        raise Exception("No se encontró el archivo merged_ui_wearable.csv necesario para el análisis biométrico")
    
    # Cargamos los datos
    try:
        print(f"Cargando archivo: {merged_file_path}")
        df_combined = pd.read_csv(merged_file_path)
    except Exception as e:
        print(f"Error al cargar el archivo: {str(e)}")
        raise Exception(f"Error al cargar el archivo biométrico: {str(e)}")
    
    # Creamos el reporte
    report = BiometricAnalysisReport.objects.create(
        title=f"Análisis Biométrico - {config.title}",
        execution=execution,
        config=config,
        metrics=config.default_metrics,
        chart_type=config.default_chart_type,
        merged_file=merged_file_name
    )
    
    # Resto del procesamiento...
    stats_data = {}
    chart_data = {}
    
    for metric in config.default_metrics:
        if metric in df_combined.columns:
            values = df_combined[metric].dropna().tolist()
            if values:
                stats_data[metric] = {
                    'mean': round(sum(values) / len(values), 2),
                    'max': round(max(values), 2),
                    'min': round(min(values), 2)
                }
                
                if len(values) > 100:
                    step = len(values) // 100
                    chart_data[metric] = values[::step][:100]
                else:
                    chart_data[metric] = values
    
    # Preparamos etiquetas para el eje X
    chart_labels = []
    if 'timestamp' in df_combined.columns:
        timestamps = df_combined['timestamp'].tolist()
        if len(timestamps) > 100:
            step = len(timestamps) // 100
            timestamps = timestamps[::step][:100]
        try:
            chart_labels = [t.split(' ')[1][:5] if ' ' in t else t[:5] for t in timestamps]
        except:
            chart_labels = list(range(1, len(next(iter(chart_data.values()))) + 1))
    else:
        if chart_data:
            chart_labels = list(range(1, len(next(iter(chart_data.values()))) + 1))
    
    report.extra_data = {
        'stats': stats_data,
        'chart_data': chart_data,
        'chart_labels': chart_labels
    }
    report.save()
    
    # Copiamos el archivo a la carpeta de ejecución para mantener consistencia
    destination_path = os.path.join(execution.exp_folder_complete_path, merged_file_name)
    if merged_file_path != destination_path:
        try:
            import shutil
            shutil.copy2(merged_file_path, destination_path)
            print(f"Archivo copiado a: {destination_path}")
        except Exception as e:
            print(f"No se pudo copiar el archivo: {str(e)}")
    
    return report