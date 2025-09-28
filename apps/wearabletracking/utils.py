from datetime import datetime
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from django.core.files.base import ContentFile
from apps.wearabletracking.models import BiometricAnalysisReport
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
from django.core.files.base import ContentFile
from apps.wearabletracking.models import FitbitToken
from django.contrib.auth.models import User


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
    Procesa los datos biométricos para cada escenario de la ejecución.
    Genera un reporte independiente para cada escenario.
    """
    # Verificaciones iniciales
    if not execution.biometric_config:
        raise Exception("No hay configuración biométrica activa para esta ejecución")
        
    if not hasattr(execution, 'monitoring') or not execution.monitoring or not getattr(execution.monitoring, 'use_wearable_data', False):
        raise Exception("El procesamiento de logs no tiene habilitada la opción de datos wearable")
    
    # Obtener configuración
    config = execution.biometric_config
    print(f"Configuración biométrica: {config.title}")
    
    # Obtener datos del usuario de Fitbit si están disponibles
    user_data = {'age': 30, 'weight': 70, 'athlete': False}  # Valores por defecto
    
    # Intentar obtener token de Fitbit para este usuario
    try:
        
        # Usar el usuario asociado a la ejecución o configuración
        user_id = execution.user_id if hasattr(execution, 'user_id') else config.user_id
        if user_id:
            token = FitbitToken.objects.filter(user_id=user_id).first()
            if token:
                headers = {"Authorization": f"Bearer {token.access_token}"}
                
                # Obtener perfil de usuario desde la API de Fitbit
                profile_url = "https://api.fitbit.com/1/user/-/profile.json"
                profile_resp = requests.get(profile_url, headers=headers)
                if profile_resp.status_code == 200:
                    user_profile = profile_resp.json().get("user", {})
                    
                    # Actualizar datos del usuario con los reales
                    user_data = {
                        'age': user_profile.get("age", 30),
                        'weight': user_profile.get("weight", 70),
                        'height': user_profile.get("height", None),
                        'gender': user_profile.get("gender", None),
                        # Determinar si es atleta basado en pasos diarios promedio
                        'athlete': user_profile.get("averageDailySteps", 0) > 10000
                    }
                    
                    # Guardar el perfil del usuario para uso posterior
                    if not hasattr(execution.monitoring, 'extra_data'):
                        execution.monitoring.extra_data = {}
                    
                    execution.monitoring.extra_data['user_profile'] = user_profile
                    execution.monitoring.save()
                    
                    print(f"Datos de usuario obtenidos: Edad {user_data['age']}, Peso {user_data['weight']}kg")
    except Exception as e:
        print(f"Error al obtener datos del usuario de Fitbit: {str(e)}")
        # Continuar con los valores por defecto
    
    # Lista para almacenar todos los reportes generados
    reports = []
    
    # Procesar cada escenario por separado
    for scenario in execution.scenarios_to_study:
        try:
            print(f"Procesando escenario: {scenario}")
            
            # Ruta del archivo merged_ui_wearable.csv específica para este escenario
            merged_file_path = os.path.join(execution.case_study.exp_folder_complete_path, scenario, "merged_ui_wearable.csv")
            
            print(f"Buscando archivo en: {merged_file_path}")
            if not os.path.exists(merged_file_path):
                print(f"No se encontró el archivo merged_ui_wearable.csv en el escenario {scenario}")
                continue
            
            # Cargamos los datos
            try:
                print(f"Cargando archivo: {merged_file_path}")
                df_combined = pd.read_csv(merged_file_path)
            except Exception as e:
                print(f"Error al cargar el archivo para el escenario {scenario}: {str(e)}")
                continue
            
            # Crear un reporte específico para este escenario
            report = BiometricAnalysisReport.objects.create(
                title=f"Análisis Biométrico - {scenario}",
                execution=execution,
                config=config,
                metrics=config.default_metrics,
                chart_type=config.default_chart_type,
                merged_file=os.path.basename(merged_file_path),
                scenario=scenario  # Nuevo campo para identificar el escenario
            )
            
            # Extraer la fecha del CSV desde los nombres de archivos de captura de pantalla
            csv_date = None
            if 'screenshot' in df_combined.columns and not df_combined['screenshot'].empty:
                for screenshot in df_combined['screenshot'].dropna():
                    if isinstance(screenshot, str) and '_' in screenshot:
                        # Formato típico: 6_25268098_2025-01-24_13-24-19.png
                        parts = screenshot.split('_')
                        for part in parts:
                            # Buscar patrón de fecha YYYY-MM-DD
                            if len(part) == 10 and part.count('-') == 2:
                                try:
                                    # Verificar que sea una fecha válida
                                    datetime.strptime(part, "%Y-%m-%d")
                                    csv_date = part
                                    break
                                except ValueError:
                                    continue
                        if csv_date:
                            break
            
            # Si no encontramos la fecha en los screenshots, intentar con los timestamps
            if not csv_date:
                if 'timestamp' in df_combined.columns and not df_combined['timestamp'].empty:
                    timestamp = df_combined['timestamp'].iloc[0]
                    try:
                        # Extraer solo la fecha (YYYY-MM-DD) si tiene formato completo
                        if isinstance(timestamp, str):
                            if 'T' in timestamp:
                                csv_date = timestamp.split('T')[0]
                            elif ' ' in timestamp:
                                csv_date = timestamp.split(' ')[0]
                    except (AttributeError, IndexError):
                        pass
                elif 'time:timestamp' in df_combined.columns and not df_combined['time:timestamp'].empty:
                    timestamp = df_combined['time:timestamp'].iloc[0]
                    try:
                        # Extraer solo la fecha (YYYY-MM-DD) si tiene formato completo
                        if isinstance(timestamp, str):
                            if 'T' in timestamp:
                                csv_date = timestamp.split('T')[0]
                            elif ' ' in timestamp:
                                csv_date = timestamp.split(' ')[0]
                    except (AttributeError, IndexError):
                        pass
            
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
                            # Caso especial para temperatura (formato {'nightlyRelative': -0.8})
                            if metric == 'temperatura' and isinstance(val, str) and val.startswith('{'):
                                import ast
                                try:
                                    dict_val = ast.literal_eval(val)
                                    if 'nightlyRelative' in dict_val:
                                        numeric_val = dict_val['nightlyRelative']
                                        numeric_values.append(numeric_val)
                                        continue
                                except (ValueError, SyntaxError):
                                    # Si hay error al procesar, intentaremos como número normal
                                    pass
                            
                            # Para otros tipos de datos, procesar normalmente 
                            if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                                continue  # Omitir estos valores que no podemos procesar
                            
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
                    
                    # Primero identificar eventos importantes
                    important_indices = []
                    if len(values) > 1:
                        max_index = values.index(max(values))
                        min_index = values.index(min(values))
                        important_indices.extend([max_index, min_index])
                        
                        # Detectar cambios bruscos (diferencia con punto anterior)
                        changes = []
                        for i in range(1, len(values)):
                            change = abs(values[i] - values[i-1])
                            changes.append((i, change))
                        
                        # Ordenar por magnitud del cambio y tomar los más significativos
                        changes.sort(key=lambda x: x[1], reverse=True)
                        change_indices = [idx for idx, _ in changes[:28] if idx != max_index and idx != min_index]
                        important_indices.extend(change_indices)
                        
                        # Eliminar duplicados y ordenar
                        important_indices = sorted(list(set(important_indices)))
                    
                    # SOLUCIÓN: Guardar TODOS los datos sin muestreo
                    chart_data[metric] = values
                    indices = list(range(len(values)))
                    
                    # Detectar eventos destacados (picos, valores mínimos, cambios bruscos)
                    events_data[metric] = []
                    
                    # Para cada índice importante, extraer la actividad correspondiente
                    for idx in important_indices:
                        if idx < len(df_combined):
                            # Enfoque simplificado: usar el índice directamente
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
                            
                            context = {
                                'activity_type': activity_type,
                                'pasos': row.get('pasos', 0) if 'pasos' in row else None,
                                'fc': row.get('fc') if 'fc' in row else None
                            }
                            
                            # Añadir valores anteriores para detectar patrones
                            if idx > 30:  # Si hay suficientes datos previos
                                previous_values = values[max(0, idx-120):idx]  # Hasta 2 horas antes
                                context['previous_values'] = previous_values
                            
                            # Usar función evaluar_metrica_biometrica
                            evaluation = evaluar_metrica_biometrica(metric, values[idx], user_data, context)
                            is_abnormal = evaluation['is_abnormal']
                            abnormal_reason = evaluation['reason']
                            
                            # Timestamp para el evento
                            timestamp = str(row.get('timestamp', '')) or str(row.get('time:timestamp', ''))
                            
                            # SOLUCIÓN: Guardar TODOS los eventos importantes con su índice original
                            events_data[metric].append({
                                'index': idx,  # Índice directo en el array completo
                                'original_index': idx,  # Mantener para compatibilidad
                                'value': values[idx],
                                'timestamp': timestamp,
                                'activity': activity_description,
                                'is_abnormal': is_abnormal,
                                'abnormal_reason': abnormal_reason,
                                'activity_type': activity_type,
                                'details': activity_details
                            })
            
            # Preparar etiquetas para el eje X - Usar TODAS las etiquetas originales
            chart_labels = []
            if 'timestamp' in df_combined.columns:
                chart_labels = df_combined['timestamp'].tolist()
            elif 'time:timestamp' in df_combined.columns:
                chart_labels = df_combined['time:timestamp'].tolist()
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
                            # Caso especial para temperatura
                            if metric_key == 'temperatura' and isinstance(val, str) and val.startswith('{'):
                                import ast
                                try:
                                    dict_val = ast.literal_eval(val)
                                    if 'nightlyRelative' in dict_val:
                                        numeric_values.append(dict_val['nightlyRelative'])
                                        continue
                                except (ValueError, SyntaxError):
                                    pass
                            
                            # Ignorar valores no numéricos en formato especial
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
                        context = {'activity_type': 'Unknown'}
                        # Para FC, usar la media como valor principal y evaluar la media
                        if metric_key == 'fc':
                            indicator_value = mean_value
                            evaluation = evaluar_metrica_biometrica(metric_key, mean_value, user_data, context)
                            if not evaluation['is_abnormal']:
                                status = "Normal"
                            else:
                                # Si es anormal, usar criterios médicos para determinar si es alto o bajo
                                if mean_value > 100:
                                    status = "High"
                                else:
                                    status = "Low"
                        else:
                            indicator_value = last_value
                            evaluation = evaluar_metrica_biometrica(metric_key, last_value, user_data, context)
                            status = "Normal" if not evaluation['is_abnormal'] else "High" if last_value > stats_data.get(metric_key, {}).get('mean', last_value) else "Low"

                        indicator_data[metric_key] = {
                            'name': metric_config['name'],
                            'value': round(indicator_value, 2),
                            'mean': round(mean_value, 2),
                            'unit': metric_config['unit'],
                            'status': status,
                            'icon': metric_config['icon'],
                            'color': metric_config['color'],
                            'reason': evaluation['reason']
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

            # Añadir datos de usuario al reporte para referencia
            user_data_safe = {
                'age': user_data.get('age', 30),
                'weight': user_data.get('weight', 70),
                'athlete': user_data.get('athlete', False)
            }
            
            # Guardar todos los datos procesados
            report.extra_data = {
                'stats': stats_data,
                'chart_data': chart_data,
                'chart_labels': chart_labels,
                'events': events_data,
                'indicators': indicator_data,
                'csv_date': csv_date,
                'user_data': user_data_safe
            }
            report.save()
            
            # Añadir el reporte a la lista de reportes generados
            reports.append(report)
            print(f"Reporte generado para escenario {scenario} con ID: {report.id}")
            
        except Exception as e:
            print(f"Error procesando escenario {scenario}: {str(e)}")
            # Continuar con el siguiente escenario
            continue
    
    # Verificar que se haya generado al menos un reporte
    if not reports:
        raise Exception("No se pudo procesar ningún escenario para análisis biométrico")
    
    # Devolver la lista de reportes generados
    return reports

def evaluar_metrica_biometrica(metric, value, user_data=None, context=None):
    """
    Evalúa si un valor biométrico es normal o anormal según criterios científicos.
    
    Args:
        metric (str): Nombre de la métrica ('fc', 'pasos', 'spo2', etc.)
        value (float): Valor de la métrica
        user_data (dict): Datos del usuario (edad, peso, etc.)
        context (dict): Contexto adicional (actividad actual, valores previos, etc.)
        
    Returns:
        dict: Diccionario con 'is_abnormal' (bool) y 'reason' (str) explicando la razón
    """
    # Valores por defecto si no se proporcionan
    if user_data is None:
        user_data = {'age': 30, 'weight': 70, 'athlete': False}
    if context is None:
        context = {'activity_type': 'Unknown', 'previous_values': []}
        
    is_abnormal = False
    reason = "Normal"
    
    # Frecuencia Cardíaca
    if metric == 'fc':
        edad = user_data.get('age', 30)
        fc_max_teorica = 208 - (0.7 * edad)
        es_atleta = user_data.get('athlete', False)
        
        # Determinar tipo de actividad
        if 'activity_type' in context and ('Exercise' in str(context['activity_type']) or 
                                           'Running' in str(context['activity_type']) or 
                                           'Workout' in str(context['activity_type'])):
            # Durante ejercicio
            if value > fc_max_teorica:
                is_abnormal = True
                reason = f"FC por encima del máximo teórico ({fc_max_teorica:.0f} lpm)"
        else:
            # En reposo
            if es_atleta:
                # Criterios para atletas
                if value < 40:
                    is_abnormal = False  # Normal para atletas
                    reason = "FC en reposo normal para atletas"
                elif value > 100:
                    is_abnormal = True
                    reason = "Taquicardia (>100 lpm en reposo)"
            else:
                # Criterios para no atletas
                if value < 60:
                    is_abnormal = True
                    reason = "Bradicardia (<60 lpm en reposo)"
                elif value > 100:
                    is_abnormal = True
                    reason = "Taquicardia (>100 lpm en reposo)"
    
    # Saturación de Oxígeno
    elif metric == 'spo2':
        edad = user_data.get('age', 30)
        
        if edad > 65:  # Criterios para adultos mayores
            if value < 92:
                is_abnormal = True
                if value < 88:
                    reason = "SpO₂ peligrosamente baja (<88%)"
                else:
                    reason = "SpO₂ baja (<92%)"
        else:  # Criterios para adultos generales
            if value < 95:
                is_abnormal = True
                if value < 93:
                    reason = "SpO₂ baja (<93%)"
                else:
                    reason = "SpO₂ ligeramente reducida (93-94%)"
    
    # Pasos por minuto
    elif metric == 'pasos':
        if 'activity_type' in context:
            activity = str(context['activity_type']).lower()
            
            # Detectar ejercicio moderado o vigoroso por descripción
            is_exercise = any(term in activity for term in ['exercise', 'workout', 'running', 'jogging', 'training'])
            
            if is_exercise:
                if 'moderate' in activity and value < 100:
                    is_abnormal = True
                    reason = "Intensidad insuficiente para ejercicio moderado (<100 pasos/min)"
                elif ('vigorous' in activity or 'intense' in activity) and value < 130:
                    is_abnormal = True
                    reason = "Intensidad insuficiente para ejercicio vigoroso (<130 pasos/min)"
            else:
                # Verificar periodos sedentarios prolongados
                if value == 0 and 'previous_values' in context:
                    consecutive_zeros = sum(1 for v in context['previous_values'] if v == 0)
                    if consecutive_zeros >= 30:  # >30 minutos consecutivos
                        is_abnormal = True
                        reason = f"Periodo sedentario prolongado ({consecutive_zeros} min sin actividad)"
    
    # Variabilidad de frecuencia cardíaca (SDNN)
    elif metric == 'sdnn' or metric == 'hrv':
        if value < 50:
            is_abnormal = True
            reason = "HRV baja (<50ms), posible indicador de estrés elevado"
    
    # Índice de Carga Cardiovascular
    elif metric == 'cvl':
        if value > 40:
            is_abnormal = True
            reason = "Carga cardiovascular elevada (>40%)"
            
            # Si además está en reposo, es más preocupante
            if 'activity_type' in context and not any(term in str(context['activity_type']).lower() 
                                                   for term in ['exercise', 'workout', 'running']):
                reason += " durante actividad sedentaria"
    
    # RATIO FC/PASOS - ACTUALIZADO PARA CONTEXTO DE OFICINA
    elif metric == 'ratio_fc_pasos':
        # Obtener pasos actuales del contexto
        pasos_actual = 0
        if context and 'pasos' in context:
            pasos_actual = context.get('pasos', 0)
            if pasos_actual is None:
                pasos_actual = 0
        
        # Obtener FC actual si está disponible
        fc_actual = None
        if context and 'fc' in context:
            fc_actual = context.get('fc')
        
        # Interpretación contextualizada según nivel de actividad
        if pasos_actual <= 5:  # Sedentario completo
            if value > 9.0:  # Umbral para estrés mental/cognitivo en estado sedentario
                is_abnormal = True
                reason = f"Ratio elevado ({value:.1f}) en estado sedentario, posible estrés mental"
                if fc_actual and fc_actual > 90:
                    reason += f" (FC={fc_actual} lpm)"
            elif value > 8.0:  # Límite superior para estado sedentario
                is_abnormal = True
                reason = f"Ratio ligeramente elevado ({value:.1f}) para estado sedentario"
            else:
                is_abnormal = False
                reason = f"Ratio normal ({value:.1f}) para trabajo sedentario de oficina"
                
        elif pasos_actual <= 20:  # Movimiento ligero
            if value > 5.0:
                is_abnormal = True
                reason = f"Ratio elevado ({value:.1f}) para movimiento ligero"
            else:
                is_abnormal = False
                reason = f"Ratio normal ({value:.1f}) para movimiento ligero en oficina"
                
        else:  # Actividad (21+ pasos)
            if value > 3.0:
                is_abnormal = True
                reason = f"Ratio elevado ({value:.1f}) durante actividad, posible ineficiencia cardíaca"
            else:
                is_abnormal = False
                reason = f"Ratio eficiente ({value:.1f}) durante actividad"
        
        # Detección de incrementos súbitos
        if 'previous_values' in context and context['previous_values']:
            prev_values = context['previous_values']
            if len(prev_values) >= 5:  # Al menos 5 minutos previos
                # Calculamos la media de los últimos 5 minutos
                recent_avg = sum(prev_values[-5:]) / 5
                
                # Si hay un incremento súbito de más del 50%
                if value > recent_avg * 1.5 and pasos_actual <= 5:
                    is_abnormal = True
                    reason = f"Incremento súbito del ratio: {value:.1f} vs. promedio reciente {recent_avg:.1f}, posible respuesta de estrés"
    
    # Temperatura
    elif metric == 'temperatura':
        # Para variaciones de temperatura relativa (desviación del baseline personal)
        if value > 1.0 or value < -1.0:
            is_abnormal = True
            if value > 1.0:
                reason = f"Temperatura elevada (+{value:.1f}°C sobre baseline)"
            else:
                reason = f"Temperatura reducida ({value:.1f}°C bajo baseline)"
    
    # Calorías
    elif metric == 'calorias':
        peso = user_data.get('weight', 70)  # kg
        calorias_reposo = 1.0 * (peso/70)  # Valor base para reposo
        
        if 'pasos' in context:
            pasos_actual = context.get('pasos', 0)
            
            # Evaluar según nivel de actividad estimado por pasos
            if pasos_actual < 20:  # Reposo/sedentario
                if value > calorias_reposo * 2:
                    is_abnormal = True
                    reason = "Gasto calórico elevado para estado sedentario"
            elif pasos_actual >= 130:  # Actividad vigorosa
                if value < calorias_reposo * 5:
                    is_abnormal = True
                    reason = "Gasto calórico insuficiente para nivel de actividad intensa"
    
    # Para métricas no específicamente implementadas
    else:
        is_abnormal = False
        reason = "Métrica dentro de rango normal"
        
    return {
        'is_abnormal': is_abnormal,
        'reason': reason
    }

def generate_biometric_report_pdf(report):
    """
    Genera un PDF con formato mejorado, elegante y profesional para el reporte biométrico
    """
    buffer = BytesIO()
    
    # Fecha y metadata del reporte
    date_str = report.extra_data.get('csv_date', report.created_at.strftime('%Y-%m-%d'))
    
    # Crear documento con márgenes adecuados
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=40,
        title=report.title
    )
    
    # Obtener estilos y crear estilos personalizados
    styles = getSampleStyleSheet()
    
    # Personalizar estilo del título principal
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#324b8b'),
        spaceAfter=10,
        alignment=1  # Centrado
    )
    
    # Personalizar otros estilos
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#324b8b'),
        spaceBefore=15,
        spaceAfter=8,
        borderWidth=0,
        borderColor=colors.HexColor('#324b8b'),
        borderPadding=5,
        borderRadius=2,
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=2,
        spaceAfter=5
    )
    
    # Estilo para subtítulos
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#5e72e4'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    # Estilo para notas informativas
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Italic'],
        fontSize=9,
        textColor=colors.darkgrey,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=5,
        spaceAfter=10
    )
    
    # Lista para elementos del PDF
    elements = []
    
    # Encabezado más elegante con línea debajo
    elements.append(Paragraph(report.title, title_style))
    elements.append(Table([['']], colWidths=[450], rowHeights=[1], 
                          style=[('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#5e72e4'))]))
    elements.append(Spacer(1, 15))
    
    # Información general del reporte en formato de tabla elegante
    metadata_data = [
        ['Fecha', date_str],
        ['Configuración', report.config.title],
        ['Tipo de gráfico', report.chart_type.capitalize()],
    ]
    
    metadata_table = Table(metadata_data, colWidths=[120, 350])
    metadata_table.setStyle(TableStyle([
        # Bordes sutiles
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.lightgrey),
        ('LINEABOVE', (0, 0), (-1, 0), 0.5, colors.lightgrey),
        # Alineación y espaciado
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        # Estilo de texto
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#5e72e4')),
    ]))
    elements.append(metadata_table)
    elements.append(Spacer(1, 20))
    
    # Indicadores principales en una tabla bien formateada
    indicators = report.extra_data.get('indicators', {})
    if indicators:
        elements.append(Paragraph("Indicadores de Salud", heading2_style))
        elements.append(Spacer(1, 5))
        
        # Preparar datos para la tabla de indicadores
        indicator_data = [['Indicador', 'Valor', 'Estado', 'Evaluación']]
        has_indicators = False
        
        for key, ind in indicators.items():
            if ind['value'] != 'N/A':
                has_indicators = True
                # Formateo especial para temperatura
                if key == 'temperatura' and ind['value'] != 'N/A':
                    value_display = f"{'+' if float(ind['value']) >= 0 else ''}{ind['value']}{ind['unit']}"
                else:
                    value_display = f"{ind['value']}{ind['unit']}"
                
                # Colorear el estado
                status = ind['status']
                # Incluir la razón de evaluación
                reason = ind.get('reason', 'No disponible')
                indicator_data.append([ind['name'], value_display, status, reason])
        
        if has_indicators:
            # Crear tabla de indicadores con mejor formato
            indicator_table = Table(indicator_data, colWidths=[100, 80, 70, 200])
            indicator_table.setStyle(TableStyle([
                # Encabezado
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5e72e4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                # Cuerpo de la tabla
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (2, -1), 'CENTER'),
                ('ALIGN', (3, 1), (3, -1), 'LEFT'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                # Bordes y divisiones
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                ('LINEABOVE', (0, 1), (-1, 1), 1, colors.lightgrey),
                # Espaciado interno de celdas
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            # Aplicar colores a los estados 
            for i in range(1, len(indicator_data)):
                status = indicator_data[i][2]
                if status == 'Normal':
                    indicator_table._cellStyles[i][2].textColor = colors.green
                elif status == 'High':
                    indicator_table._cellStyles[i][2].textColor = colors.orange
                elif status == 'Low':
                    indicator_table._cellStyles[i][2].textColor = colors.red
            
            elements.append(indicator_table)
        else:
            elements.append(Paragraph("No hay datos disponibles para los indicadores", info_style))
        
        elements.append(Spacer(1, 20))
    
    # Para cada métrica, crear una sección separada con título, gráfica y datos
    chart_data = report.extra_data.get('chart_data', {})
    chart_labels = report.extra_data.get('chart_labels', [])
    stats = report.extra_data.get('stats', {})
    events = report.extra_data.get('events', {})
    
    # Nombres legibles para métricas
    metric_names = {
        'fc': 'Frecuencia Cardíaca (bpm)',
        'pasos': 'Pasos',
        'calorias': 'Calorías',
        'zona_activa': 'Zona Activa',
        'sedentario': 'Tiempo Sedentario',
        'ratio_fc_pasos': 'Ratio FC/Pasos',
        'cvl': 'CVL',
        'sdnn': 'SDNN (ms)',
        'spo2': 'SpO₂ (%)',
        'temperatura': 'Variación de Temperatura (°C)',
        'hrv': 'HRV (ms)',
    }
    
    # Colores para las gráficas
    colors_dict = {
        'fc': '#FF6384',           # Rojo para frecuencia cardíaca
        'pasos': '#36A2EB',        # Azul para pasos
        'calorias': '#FFCE56',     # Amarillo para calorías
        'zona_activa': '#4BC0C0',  # Verde azulado para zona activa
        'sedentario': '#9966FF',   # Púrpura para sedentario
        'ratio_fc_pasos': '#FF9F40', # Naranja para ratio FC/pasos
        'cvl': '#C9CBCF',          # Gris para CVL
        'sdnn': '#7FC97F',         # Verde para SDNN
        'spo2': '#1d8cf8',         # Azul claro para SpO2
        'temperatura': '#a38df8',  # Púrpura para temperatura
        'hrv': '#f58231'           # Naranja para HRV
    }
    
    for metric in report.metrics:
        # Contenedor con borde para cada métrica
        metric_elements = []
        
        # Título de la métrica
        metric_elements.append(Paragraph(metric_names.get(metric, metric.capitalize()), heading2_style))
        metric_elements.append(Spacer(1, 5))
        
        # Verificar si hay datos para esta métrica
        has_data = metric in chart_data and chart_data[metric] and len(chart_data[metric]) > 0
        
        if has_data:
            # Estadísticas en formato de tabla elegante
            if metric in stats:
                s = stats[metric]
                
                # Tabla de estadísticas con mejor formato
                stat_data = [['Media', 'Máximo', 'Mínimo', 'Último Valor']]
                stat_data.append([
                    str(s['mean']), 
                    str(s['max']), 
                    str(s['min']), 
                    str(s['current'])
                ])
                
                stat_table = Table(stat_data, colWidths=[100, 100, 100, 100])
                stat_table.setStyle(TableStyle([
                    # Encabezado
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6e9f0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#324b8b')),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    # Cuerpo
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                    # Bordes
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                    # Espaciado
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                metric_elements.append(stat_table)
                metric_elements.append(Spacer(1, 15))
            
            # Gráfica
            if metric in chart_data and chart_labels:
                # Crear gráfica con matplotlib
                plt.figure(figsize=(7, 3.5))
                
                # Obtener color para la métrica
                color = colors_dict.get(metric, '#5e72e4')
                
                # Reducir la cantidad de puntos si hay demasiados
                values = chart_data[metric]
                labels = chart_labels
                
                if len(values) > 50:
                    step = len(values) // 50
                    values = [values[i] for i in range(0, len(values), step)]
                    labels = [labels[i] for i in range(0, len(labels), step) if i < len(labels)]
                
                # Tipo de gráfico según configuración
                if report.chart_type == 'bar':
                    plt.bar(range(len(values)), values, color=color, alpha=0.6)
                elif report.chart_type == 'area':
                    plt.fill_between(range(len(values)), values, alpha=0.3, color=color)
                    plt.plot(range(len(values)), values, color=color, linewidth=2)
                else:  # 'line' (default)
                    plt.plot(range(len(values)), values, color=color, linewidth=2)
                
                # Mejorar el estilo general del gráfico
                plt.grid(True, linestyle='--', alpha=0.7, color='#e6e9f0')
                plt.gcf().set_facecolor('#fcfcfc')
                
                # Configurar etiquetas del eje X
                if len(labels) > 10:
                    plt.xticks(
                        range(0, len(values), len(values) // 10),
                        [labels[i] for i in range(0, len(labels), len(labels) // 10) if i < len(labels)],
                        rotation=45
                    )
                else:
                    plt.xticks(range(len(values)), labels, rotation=45)
                
                # Añadir sombra para efecto 3D sutil
                plt.gca().spines['bottom'].set_linewidth(1.5)
                plt.gca().spines['left'].set_linewidth(1.5)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                
                # Título y ajustes
                plt.title(metric_names.get(metric, metric.capitalize()), fontsize=12, color=color, fontweight='bold')
                plt.tight_layout()
                
                # Guardar en buffer
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
                plt.close()
                img_buffer.seek(0)
                
                # Añadir imagen al PDF con un marco
                img = Image(img_buffer, width=450, height=225)
                metric_elements.append(img)
                metric_elements.append(Spacer(1, 15))
            
            # Tabla de eventos - MODIFICADO para incluir todos los eventos
            if metric in events and events[metric]:
                metric_elements.append(Paragraph("Eventos Significativos", subtitle_style))
                metric_elements.append(Spacer(1, 5))
                
                # Datos para la tabla de eventos
                event_data = [['Hora', 'Valor', 'Estado', 'Actividad', 'Detalles']]
                
                # Incluir todos los eventos, sin limitación
                for event in events[metric]:  # Sin límite [:5]
                    # Formatear valor según el tipo de métrica
                    if metric == 'temperatura' and event.get('value') is not None:
                        sign = '+' if event['value'] >= 0 else ''
                        value_str = f"{sign}{event['value']:.1f}°C"
                    else:
                        value_str = str(event.get('value', 'N/A'))
                    
                    # Formatear detalles
                    details = event.get('details', {})
                    details_str = ', '.join([f"{k}: {v}" for k, v in list(details.items())[:2] if v])
                    
                    # Incluir el estado y la razón
                    is_abnormal = event.get('is_abnormal', False)
                    reason = event.get('abnormal_reason', 'Normal')
                    
                    # Estado formateado para el PDF
                    if is_abnormal:
                        status_str = "⚠️ Anormal"
                    else:
                        status_str = "✓ Normal"
                    
                    event_data.append([
                        event.get('timestamp', 'N/A')[-8:] if event.get('timestamp', 'N/A') else 'N/A',  # Solo la hora
                        value_str,
                        status_str,
                        event.get('activity_type', 'Unknown')[:15],  # Limitar longitud
                        f"{reason[:20]}{'...' if len(reason) > 20 else ''}"  # Incluir razón
                    ])
                
                # Crear tabla de eventos con mejor formato
                event_table = Table(event_data, colWidths=[40, 60, 60, 100, 190])
                event_table.setStyle(TableStyle([
                    # Encabezado
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6e9f0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#324b8b')),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    # Cuerpo
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('ALIGN', (0, 1), (2, -1), 'CENTER'),  # Hora, valor y estado centrados
                    ('ALIGN', (3, 1), (-1, -1), 'LEFT'),   # Actividad y detalles a la izquierda
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    # Bordes
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                    # Espaciado
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    # Tamaño de fuente
                    ('FONTSIZE', (0, 1), (-1, -1), 8),  # Texto más pequeño para los datos
                ]))
                
                # Aplicar colores alternos a las filas
                for i in range(1, len(event_data)):
                    if i % 2 == 0:
                        for j in range(len(event_data[i])):
                            event_table._cellStyles[i][j].backColor = colors.HexColor('#f9f9f9')
                
                # Aplicar colores según estado
                for i in range(1, len(event_data)):
                    # Color basado en normal/anormal
                    status = event_data[i][2]
                    if "Anormal" in status:
                        event_table._cellStyles[i][2].textColor = colors.orange
                    else:
                        event_table._cellStyles[i][2].textColor = colors.green
                
                metric_elements.append(event_table)
            else:
                metric_elements.append(Paragraph("No hay eventos significativos registrados para esta métrica.", info_style))
        else:
            # Mensaje cuando no hay datos
            metric_elements.append(Paragraph("No hay datos disponibles para esta métrica en el período analizado.", info_style))
        
        # Añadir todos los elementos de la métrica con un separador
        for element in metric_elements:
            elements.append(element)
        
        # Separador entre métricas
        elements.append(Spacer(1, 20))
        elements.append(Table([['']], colWidths=[450], rowHeights=[1], 
                             style=[('LINEBELOW', (0, 0), (-1, -1), 1, colors.lightgrey)]))
        elements.append(Spacer(1, 20))
    
    # NUEVO: Añadir sección de criterios científicos al PDF
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Criterios de Evaluación Biométrica", heading2_style))
    elements.append(Spacer(1, 5))

    criteria_text = """
    Los valores biométricos se evalúan según criterios científicos basados en la edad, condición física y contexto de actividad del usuario:

    • <b>Frecuencia Cardíaca:</b> Normal en reposo (60-100 lpm), atletas (40-60 lpm)
    • <b>SpO₂:</b> Normal (≥95%), adultos mayores (≥92%)
    • <b>Temperatura:</b> Variaciones normales de hasta ±1.0°C
    • <b>HRV:</b> Normal (≥50ms)
    • <b>Ratio FC/Pasos:</b>
       - Sedentario (0-5 pasos): Normal entre 3.0-8.0
       - Movimiento ligero (6-20 pasos): Normal entre 2.0-5.0
       - Actividad (21+ pasos): Normal menor a 3.0
    • <b>Pasos:</b> Sedentarismo (<30 min consecutivos sin movimiento)
    """

    elements.append(Paragraph(criteria_text, normal_style))
    elements.append(Spacer(1, 10))
    
    # Pie de página
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Este reporte contiene información biométrica recopilada durante la ejecución del caso de estudio. La interpretación de los datos debe ser realizada por profesionales cualificados.", info_style))
    
    # Construir el PDF 
    doc.build(elements)
    
    # Obtener el contenido del PDF
    pdf = buffer.getvalue()
    buffer.close()
    
    # Guardar en el modelo
    report.report_file.save(f"biometric_report_{report.id}.pdf", ContentFile(pdf))
    
    return pdf