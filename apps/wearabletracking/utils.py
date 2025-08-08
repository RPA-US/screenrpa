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
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
from django.core.files.base import ContentFile


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
                                    # Para temperatura relativa, valores fuera de ±1.0°C son anormales
                                    is_abnormal = values[idx] > 1.0 or values[idx] < -1.0
                                
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
                            # Para temperatura relativa, valores fuera de ±0.8°C son anormales
                            if last_value > 0.8:
                                status = "High"
                            elif last_value < -0.8:
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
            
            # Guardar todos los datos procesados, incluyendo la fecha del CSV
            report.extra_data = {
                'stats': stats_data,
                'chart_data': chart_data,
                'chart_labels': chart_labels,
                'events': events_data,
                'indicators': indicator_data,
                'csv_date': csv_date
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
        indicator_data = [['Indicador', 'Valor', 'Estado']]
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
                indicator_data.append([ind['name'], value_display, status])
        
        if has_indicators:
            # Crear tabla de indicadores con mejor formato
            indicator_table = Table(indicator_data, colWidths=[200, 150, 100])
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
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
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
            
            # Aplicar colores a los estados después, sin lambda
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
        'fc': '#f5365c',
        'pasos': '#5e72e4',
        'calorias': '#fb6340',
        'zona_activa': '#2dce89',
        'sedentario': '#11cdef',
        'ratio_fc_pasos': '#8965e0',
        'cvl': '#ffd600',
        'sdnn': '#8898aa',
        'spo2': '#1d8cf8',
        'temperatura': '#a38df8',
        'hrv': '#f58231'
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
            
            # Tabla de eventos
            if metric in events and events[metric]:
                metric_elements.append(Paragraph("Eventos Significativos", subtitle_style))
                metric_elements.append(Spacer(1, 5))
                
                # Datos para la tabla de eventos
                event_data = [['Hora', 'Valor', 'Actividad', 'Detalles']]
                
                for event in events[metric][:5]:  # Limitar a 5 eventos para que se vea mejor
                    # Formatear valor según el tipo de métrica
                    if metric == 'temperatura' and event.get('value') is not None:
                        sign = '+' if event['value'] >= 0 else ''
                        value_str = f"{sign}{event['value']:.1f}°C"
                    else:
                        value_str = str(event.get('value', 'N/A'))
                    
                    # Formatear detalles
                    details = event.get('details', {})
                    details_str = ', '.join([f"{k}: {v}" for k, v in list(details.items())[:2] if v])
                    
                    event_data.append([
                        event.get('timestamp', 'N/A')[-8:] if event.get('timestamp', 'N/A') else 'N/A',  # Solo la hora
                        value_str,
                        event.get('activity_type', 'Unknown')[:15],  # Limitar longitud
                        details_str[:30] + ('...' if len(details_str) > 30 else '')  # Limitar longitud
                    ])
                
                # Crear tabla de eventos con mejor formato
                event_table = Table(event_data, colWidths=[70, 70, 120, 200])
                event_table.setStyle(TableStyle([
                    # Encabezado
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6e9f0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#324b8b')),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    # Cuerpo
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('ALIGN', (0, 1), (1, -1), 'CENTER'),  # Hora y valor centrados
                    ('ALIGN', (2, 1), (-1, -1), 'LEFT'),   # Actividad y detalles a la izquierda
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    # Bordes
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                    # Espaciado
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    # Tamaño de fuente
                    ('FONTSIZE', (0, 1), (-1, -1), 8),  # Texto más pequeño para los datos
                ]))
                
                # Aplicar colores alternos a las filas manualmente (sin lambda)
                for i in range(1, len(event_data)):
                    if i % 2 == 0:
                        for j in range(len(event_data[i])):
                            event_table._cellStyles[i][j].backColor = colors.HexColor('#f9f9f9')
                
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