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
    Procesa los datos biométricos según la configuración seleccionada,
    extrayendo datos y actividades del archivo merged_ui_wearable.csv.
    """
    if not execution.biometric_config:
        raise Exception("No hay configuración biométrica activa para esta ejecución")
    
    # Obtener configuración
    config = execution.biometric_config
    print(f"Configuración biométrica: {config.title}")
    
    # Buscar el archivo merged_ui_wearable.csv
    merged_file_name = 'merged_ui_wearable.csv'
    possible_locations = [
        os.path.join(execution.case_study.exp_folder_complete_path, merged_file_name),
        *[os.path.join(execution.case_study.exp_folder_complete_path, scenario, merged_file_name) 
          for scenario in execution.scenarios_to_study],
        os.path.join(execution.exp_folder_complete_path, merged_file_name)
    ]
    
    merged_file_path = next((loc for loc in possible_locations if os.path.exists(loc)), None)
    
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
        merged_file=os.path.basename(merged_file_path)
    )
    
    # Procesamiento mejorado
    stats_data = {}
    chart_data = {}
    events_data = {}  # Para eventos destacados (picos, mínimos, etc.)
    indicator_data = {}  # Para las tarjetas de indicadores
    
    # Procesar cada métrica seleccionada
    for metric in config.default_metrics:
        if metric in df_combined.columns:
            # Filtrar y convertir valores a números
            numeric_values = []
            for val in df_combined[metric].dropna():
                try:
                    # Si es un diccionario en formato de cadena (como '{'nightlyRelative': -0.8}')
                    if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                        continue  # Omitir estos valores
                    
                    # Convertir a número
                    numeric_val = float(val) if val != '' else None
                    if numeric_val is not None:
                        numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    # Si no se puede convertir, ignorar
                    continue
            
            if not numeric_values:
                continue
            
            # Usar solo los valores numéricos
            values = numeric_values
                
            # Estadísticas básicas
            stats_data[metric] = {
                'mean': round(sum(values) / len(values), 2),
                'max': round(max(values), 2),
                'min': round(min(values), 2),
                'current': round(values[-1], 2)
            }
            
            # Datos para gráficos (muestrear si son muchos)
            if len(values) > 100:
                step = len(values) // 100
                indices = list(range(0, len(values), step))[:100]
                chart_data[metric] = [values[i] for i in indices]
            else:
                indices = list(range(len(values)))
                chart_data[metric] = values
            
            # Detectar eventos destacados (picos, valores mínimos, cambios bruscos)
            events_data[metric] = []
            
            # Índices para eventos importantes
            if len(values) > 1:
                max_index = values.index(max(values))
                min_index = values.index(min(values))
                
                # Detectar cambios bruscos (diferencia con punto anterior)
                changes = []
                for i in range(1, len(values)):
                    change = abs(values[i] - values[i-1])
                    changes.append((i, change))
                
                # Ordenar por magnitud del cambio y tomar los más significativos
                changes.sort(key=lambda x: x[1], reverse=True)
                change_indices = [idx for idx, _ in changes[:2] if idx != max_index and idx != min_index]
                
                # Índices finales para eventos (máximo, mínimo y cambios significativos)
                event_indices = list(set([max_index, min_index] + change_indices))
                
                # Para cada índice, extraer la actividad correspondiente
                for idx in event_indices:
                    if idx < len(df_combined):
                        # Necesitamos mapear el índice en la lista de valores numéricos 
                        # a su posición correspondiente en el DataFrame original
                        # Esto es complicado porque hemos filtrado valores no numéricos
                        
                        # Enfoque simplificado: usar el índice directamente
                        # (esto asume que la mayoría de los valores son numéricos)
                        row_idx = min(idx, len(df_combined) - 1)
                        row = df_combined.iloc[row_idx]
                        
                        # Extraer información rica del UI log
                        activity_type = "Unknown"
                        activity_details = {}
                        
                        # Columnas principales de actividad
                        if 'category' in row and pd.notna(row['category']):
                            activity_type = row['category']
                            
                        if 'application' in row and pd.notna(row['application']):
                            activity_details['app'] = row['application']
                            
                        if 'concept:name' in row and pd.notna(row['concept:name']):
                            activity_details['action'] = row['concept:name']
                        
                        # Detalles adicionales importantes
                        ui_fields = [
                            ('typed_word', 'Input'), 
                            ('tag_innerText', 'Element Text'), 
                            ('tag_name', 'Element Type'),
                            ('tag_title', 'Element Title'),
                            ('coordX', 'Position X'), 
                            ('coordY', 'Position Y')
                        ]
                        
                        for field, label in ui_fields:
                            if field in row and pd.notna(row[field]) and row[field]:
                                activity_details[label] = row[field]
                        
                        # Formatear detalles de forma legible
                        details_text = []
                        for key, val in activity_details.items():
                            if val:  # Solo incluir si tiene valor
                                details_text.append(f"{key}: {val}")
                                
                        # Construir descripción final
                        activity_description = f"{activity_type}"
                        if details_text:
                            activity_description += f" ({'; '.join(details_text[:3])})"  # Limitar a 3 detalles
                        
                        # Determinar si es un valor anormal según la métrica
                        is_abnormal = False
                        if metric == 'fc':
                            is_abnormal = values[idx] > 90 or values[idx] < 40
                        elif metric == 'spo2':
                            is_abnormal = values[idx] < 95
                        elif metric == 'temperatura':
                            is_abnormal = values[idx] > 37.5 or values[idx] < 35.5
                        
                        # Timestamp para el evento
                        timestamp = str(row.get('timestamp', '')) or str(row.get('time:timestamp', ''))
                        
                        # Agregar el evento a la lista con información detallada
                        events_data[metric].append({
                            'index': indices.index(idx) if idx in indices else 0,
                            'original_index': idx,
                            'value': values[idx],
                            'timestamp': timestamp,
                            'activity': activity_description,
                            'is_abnormal': is_abnormal,
                            # Agregar campos adicionales para mejor visualización
                            'activity_type': activity_type,
                            'details': activity_details
                        })
    
    # Preparar etiquetas para el eje X
    chart_labels = []
    if 'timestamp' in df_combined.columns:
        timestamps = df_combined['timestamp'].tolist()
        if len(timestamps) > 100:
            step = len(timestamps) // 100
            timestamps = [timestamps[i] for i in range(0, len(timestamps), step)][:100]
        chart_labels = timestamps
    elif 'time:timestamp' in df_combined.columns:
        timestamps = df_combined['time:timestamp'].tolist()
        if len(timestamps) > 100:
            step = len(timestamps) // 100
            timestamps = [timestamps[i] for i in range(0, len(timestamps), step)][:100]
        chart_labels = timestamps
    else:
        if chart_data:
            chart_labels = list(range(1, len(next(iter(chart_data.values()))) + 1))
    
    # Preparar datos de indicadores para las tarjetas
    # Estos son los datos fijos que siempre queremos mostrar aunque no estén en el CSV
    key_metrics = {
        'fc': {'name': 'Heart rate', 'unit': 'bpm', 'icon': 'heartbeat', 'color': 'danger'},
        'spo2': {'name': 'SpO₂', 'unit': '%', 'icon': 'tint', 'color': 'primary'},
        'temperatura': {'name': 'Temperature', 'unit': '°C', 'icon': 'thermometer-half', 'color': 'purple'},
        'hrv': {'name': 'HRV', 'unit': 'ms', 'icon': 'chart-line', 'color': 'warning'}
    }
    
    # Para cada métrica clave, preparar datos para las tarjetas
    for metric_key, metric_config in key_metrics.items():
        if metric_key in df_combined.columns:
            # Filtrar y convertir valores a números
            numeric_values = []
            for val in df_combined[metric_key].dropna():
                try:
                    # Ignorar valores no numéricos
                    if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                        continue
                    
                    numeric_val = float(val) if val != '' else None
                    if numeric_val is not None:
                        numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    continue
            
            if numeric_values:
                last_value = numeric_values[-1]
                mean_value = sum(numeric_values) / len(numeric_values)
                
                # Determinar estado según valores normales para cada métrica
                status = "Normal"
                if metric_key == 'fc':
                    if last_value > 90:
                        status = "High"
                    elif last_value < 40:
                        status = "Low"
                elif metric_key == 'spo2':
                    if last_value < 95:
                        status = "Low"
                elif metric_key == 'temperatura':
                    if last_value > 37.5:
                        status = "High"
                    elif last_value < 35.5:
                        status = "Low"
                
                indicator_data[metric_key] = {
                    'name': metric_config['name'],
                    'value': round(last_value, 2),
                    'mean': round(mean_value, 2),
                    'unit': metric_config['unit'],
                    'status': status,
                    'icon': metric_config['icon'],
                    'color': metric_config['color']
                }
            else:
                # No hay valores numéricos válidos
                indicator_data[metric_key] = {
                    'name': metric_config['name'],
                    'value': 'N/A',
                    'mean': 'N/A',
                    'unit': metric_config['unit'],
                    'status': 'Unknown',
                    'icon': metric_config['icon'],
                    'color': metric_config['color']
                }
        else:
            # Si no está en el CSV, añadir un placeholder
            indicator_data[metric_key] = {
                'name': metric_config['name'],
                'value': 'N/A',
                'mean': 'N/A',
                'unit': metric_config['unit'],
                'status': 'Unknown',
                'icon': metric_config['icon'],
                'color': metric_config['color']
            }
    
    # Guardar todos los datos procesados
    report.extra_data = {
        'stats': stats_data,
        'chart_data': chart_data,
        'chart_labels': chart_labels,
        'events': events_data,
        'indicators': indicator_data
    }
    report.save()
    
    return report