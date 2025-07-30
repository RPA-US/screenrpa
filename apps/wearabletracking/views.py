import io
import json
import os
import zipfile
import requests
from django.conf import settings

from apps.wearabletracking.utils import obtener_ultima_sync, validar_fechas, fechas_fuera_de_sync
from apps.analyzer.models import CaseStudy, Execution
from .models import FitbitToken, BiometricAnalysisConfig, BiometricAnalysisReport
from datetime import datetime, timedelta
import csv
from django.http import HttpResponse, FileResponse
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404

from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.urls import reverse

@login_required
def wearable_home(request):
    token = FitbitToken.objects.filter(user=request.user).first()
    if token and (token.created_at + timedelta(seconds=token.expires_in)) > timezone.now():
        # Obtener dispositivos
        device_url = "https://api.fitbit.com/1/user/-/devices.json"
        headers = {"Authorization": f"Bearer {token.access_token}"}
        device_resp = requests.get(device_url, headers=headers)
        devices = device_resp.json() if device_resp.status_code == 200 else []

        return render(request, 'wearabletracking/callback.html', {
            'token': {
                'access_token': token.access_token,
                'user_id': token.user_id,
            },
            'devices': devices,
            'fitbit_user': token.user_id,
        })
    else:
        return render(request, 'wearabletracking/wearable_home.html')
    
def authorize(request):
    scope = "activity heartrate location nutrition profile settings sleep social weight oxygen_saturation stress respiratory_rate temperature"
    auth_url = (
        f"https://www.fitbit.com/oauth2/authorize"
        f"?response_type=code"
        f"&client_id={settings.FITBIT_CLIENT_ID}"
        f"&redirect_uri={settings.FITBIT_REDIRECT_URI}"
        f"&scope={scope.replace(' ', '%20')}"
        f"&expires_in=604800"
    )
    return redirect(auth_url)

@login_required
def callback(request):
    code = request.GET.get('code')
    if not code:
        return render(request, 'wearabletracking/callback.html', {'error': 'No se recibió el código.'})

    token_url = "https://api.fitbit.com/oauth2/token"
    headers = {
        "Authorization": requests.auth._basic_auth_str(settings.FITBIT_CLIENT_ID, settings.FITBIT_CLIENT_SECRET),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "client_id": settings.FITBIT_CLIENT_ID,
        "grant_type": "authorization_code",
        "redirect_uri": settings.FITBIT_REDIRECT_URI,
        "code": code
    }

    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code == 200:
        token_data = response.json()

        # Elimina el token antiguo solo del usuario actual
        FitbitToken.objects.filter(user=request.user).delete()

        # Guardar token en la base de datos para el usuario actual
        token = FitbitToken.objects.create(
            user=request.user,
            access_token=token_data['access_token'],
            refresh_token=token_data['refresh_token'],
            expires_in=token_data['expires_in'],
            token_type=token_data['token_type'],
            scope=token_data['scope'],
            fitbit_user_id=token_data['user_id']
        )

        # Obtener dispositivos del usuario
        device_url = "https://api.fitbit.com/1/user/-/devices.json"
        device_headers = {"Authorization": f"Bearer {token.access_token}"}
        device_resp = requests.get(device_url, headers=device_headers)
        devices = device_resp.json() if device_resp.status_code == 200 else []

        return render(request, 'wearabletracking/callback.html', {
            'token': token_data,
            'devices': devices,
            'fitbit_user': token_data['user_id'],
        })
    else:
        return render(request, 'wearabletracking/callback.html', {'error': response.json()})
     
@login_required
def fitbit_logout(request):
    FitbitToken.objects.filter(user=request.user).delete()
    return redirect(reverse('wearabletracking:wearable_home'))


def calcular_metricas_derivadas(pasos, fc, edad, fc_reposo):
    sedentario = [0] * len(pasos)
    for i in range(30, len(pasos)):
        if sum(pasos[i-30:i]) == 0:
            sedentario[i] = 1
    ratio = [fc[i]/(pasos[i]+1) if fc[i] is not None else None for i in range(len(fc))]
    fc_max = 208 - 0.7 * edad
    cvl = [round(((fc[i] - fc_reposo)/(fc_max - fc_reposo))*100, 2) if fc[i] else None for i in range(len(fc))]
    # SDNN en ventanas móviles de 5 minutos
    ventana = 5
    sdnn = []
    for i in range(len(fc)):
        if i < ventana - 1 or any(fc[j] is None for j in range(i-ventana+1, i+1)):
            sdnn.append(None)
        else:
            ventana_fc = fc[i-ventana+1:i+1]
            media = sum(ventana_fc) / ventana
            varianza = sum((x - media) ** 2 for x in ventana_fc) / (ventana - 1)
            sdnn_val = varianza ** 0.5
            sdnn.append(round(sdnn_val, 2))
    return sedentario, ratio, cvl, sdnn

def exportar_datos_fitbit(request):
    if request.method == 'POST':
        fecha_inicio = request.POST.get('fecha_inicio')
        fecha_fin = request.POST.get('fecha_fin')

        # Obtener token y dispositivos antes de cualquier validación
        token = FitbitToken.objects.filter(user=request.user).first()
        devices = []
        if token:
            headers = {"Authorization": f"Bearer {token.access_token}"}
            device_url = "https://api.fitbit.com/1/user/-/devices.json"
            device_resp = requests.get(device_url, headers=headers)
            devices = device_resp.json() if device_resp.status_code == 200 else []
        else:
            headers = {}

        # Validación de fechas
        error_msg = validar_fechas(fecha_inicio, fecha_fin)
        if error_msg:
            return render(request, 'wearabletracking/callback.html', {
                'devices': devices,
                'fitbit_user': token.user_id if token else None,
                'error': error_msg,
                'fecha_inicio': fecha_inicio,
                'fecha_fin': fecha_fin,
            })

        # Validación de sincronización
        ultima_sync = obtener_ultima_sync(devices)
        fuera_de_sync = fechas_fuera_de_sync(fecha_fin, ultima_sync)
        if fuera_de_sync:
            return render(request, 'wearabletracking/callback.html', {
                'devices': devices,
                'fitbit_user': token.user_id if token else None,
                'error': fuera_de_sync,
                'fecha_inicio': fecha_inicio,
                'fecha_fin': fecha_fin,
            })

        edad = 30  # Ajustar según usuario real
        fc_reposo = 65  # Ajustar según usuario real

        current_date = datetime.strptime(fecha_inicio, "%Y-%m-%d").date()
        end_date = datetime.strptime(fecha_fin, "%Y-%m-%d").date()

        # Detectar días sin datos relevantes
        dias_sin_datos = []
        fechas_a_exportar = []
        temp_current_date = current_date
        def fetch_data(metric, fecha_str):
            url = f"https://api.fitbit.com/1/user/-/activities/{metric}/date/{fecha_str}/1d/1min.json"
            r = requests.get(url, headers=headers)
            key = f"activities-{metric}-intraday"
            if r.status_code == 200:
                return {d['time']: d['value'] for d in r.json().get(key, {}).get('dataset', [])}
            return {}

        while temp_current_date <= end_date:
            fecha_str = temp_current_date.strftime("%Y-%m-%d")
            fc_data = fetch_data('heart', fecha_str)
            pasos_data = fetch_data('steps', fecha_str)
            calorias_data = fetch_data('calories', fecha_str)
            if not fc_data and not pasos_data and not calorias_data:
                dias_sin_datos.append(fecha_str)
            fechas_a_exportar.append(fecha_str)
            temp_current_date += timedelta(days=1)

        # Advertencia si hay días sin datos y no se ha confirmado la descarga
        if dias_sin_datos and not request.POST.get('confirmar_descarga'):
            advertencia = (
                f"Advertencia: No hay datos para los días {', '.join(dias_sin_datos)}. "
                "¿Desea descargar el archivo igualmente?"
            )
            return render(request, 'wearabletracking/callback.html', {
                'devices': devices,
                'fitbit_user': token.user_id if token else None,
                'error': advertencia,
                'dias_sin_datos': dias_sin_datos,
                'fecha_inicio': fecha_inicio,
                'fecha_fin': fecha_fin,
            })

        # Crear un buffer para el zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            while current_date <= end_date:
                fecha_str = current_date.strftime("%Y-%m-%d")

                # Métricas diarias
                spo2_url = f"https://api.fitbit.com/1/user/-/spo2/date/{fecha_str}.json"
                spo2_resp = requests.get(spo2_url, headers=headers)
                spo2_avg = spo2_resp.json().get("value", {}).get("avg", "") if spo2_resp.status_code == 200 else ""

                temp_url = f"https://api.fitbit.com/1/user/-/temp/skin/date/{fecha_str}.json"
                temp_resp = requests.get(temp_url, headers=headers)
                temp_data = temp_resp.json().get("tempSkin", [{}])
                temp_var = temp_data[0].get("value", "") if temp_data else ""

                hrv_url = f"https://api.fitbit.com/1/user/-/hrv/date/{fecha_str}/all.json"
                hrv_resp = requests.get(hrv_url, headers=headers)
                hrv_rmssd = ""
                if hrv_resp.status_code == 200:
                    hrv_data = hrv_resp.json().get("hrv", [])
                    rmssd_values = []
                    for day_data in hrv_data:
                        for minute in day_data.get("minutes", []):
                            val = minute.get("value", {}).get("rmssd")
                            if val is not None:
                                rmssd_values.append(val)
                    if rmssd_values:
                        hrv_rmssd = sum(rmssd_values) / len(rmssd_values)

                # Frecuencia cardíaca, pasos, calorías
                fc_data = fetch_data('heart', fecha_str)
                pasos_data = fetch_data('steps', fecha_str)
                calorias_data = fetch_data('calories', fecha_str)

                # Zonas activas: manejo especial
                azm_url = f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{fecha_str}/1d/1min.json"
                azm_resp = requests.get(azm_url, headers=headers)
                azm_data = {}
                if azm_resp.status_code == 200:
                    azm_json = azm_resp.json().get("activities-active-zone-minutes-intraday", [])
                    if azm_json and "minutes" in azm_json[0]:
                        for entry in azm_json[0]["minutes"]:
                            minute = entry["minute"][-8:]  # HH:MM:SS
                            azm_data[minute] = entry["value"].get("activeZoneMinutes", 0)

                minutos = sorted(set(fc_data.keys()) | set(pasos_data.keys()) | set(azm_data.keys()))
                fc_list = [fc_data.get(m, None) for m in minutos]
                pasos_list = [pasos_data.get(m, 0) for m in minutos]

                sedentario, ratio, cvl, sdnn = calcular_metricas_derivadas(
                    pasos_list, fc_list, edad, fc_reposo
                )

                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow([
                    'timestamp', 'fc', 'pasos', 'calorias', 'zona_activa',
                    'sedentario', 'ratio_fc_pasos', 'cvl',
                    'spo2', 'temperatura', 'hrv', 'sdnn'
                ])

                for idx, minuto in enumerate(minutos):
                    writer.writerow([
                        f"{fecha_str}T{minuto}",
                        fc_data.get(minuto, ''),
                        pasos_data.get(minuto, ''),
                        calorias_data.get(minuto, ''),
                        azm_data.get(minuto, 0),
                        sedentario[idx],
                        ratio[idx],
                        cvl[idx],
                        spo2_avg,
                        temp_var,
                        hrv_rmssd,
                        sdnn[idx]
                    ])

                zip_file.writestr(f"fitbit_{fecha_str}.csv", csv_buffer.getvalue())
                current_date += timedelta(days=1)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="fitbit_csvs_por_dia.zip"'
        return response

    return render(request, 'wearabletracking/form_range.html')

def analytics(request):
    date_str = request.GET.get('date', datetime.today().strftime("%Y-%m-%d"))
    csv_path = os.path.join(settings.MEDIA_ROOT, f"fitbit_{date_str}.csv")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['time'] = df['timestamp'].dt.strftime('%H:%M')

    df['fc']             = df['fc'].fillna(0)
    df['pasos']          = df['pasos'].fillna(0)
    df['calorias']       = df['calorias'].fillna(0)
    df['zona_activa']    = df['zona_activa'].fillna(0)
    df['sedentario']     = df['sedentario'].fillna(0)
    df['ratio_fc_pasos'] = df['ratio_fc_pasos'].fillna(0)
    df['cvl']            = df['cvl'].fillna(0)
    df['sdnn']           = df['sdnn'].fillna(0)
    df['spo2']           = df['spo2'].fillna(method='ffill')  # Para que no esté vacío

    # Datos unificados por tiempo para las gráficas
    records = []
    for i in range(len(df)):
        records.append({
            "name": df['time'].iloc[i],
            "frecuencia": int(df['fc'].iloc[i]),
            "estres": float(df['cvl'].iloc[i]),
            "oxigeno": float(df['spo2'].iloc[i])
        })

    daily = {
        'spo2':      df['spo2'].iloc[0],
        'hrv_rmssd': df['hrv'].iloc[0] if 'hrv' in df.columns else None,
        'temp_skin': df['temperatura'].iloc[0] if 'temperatura' in df.columns else None,
    }

    token = FitbitToken.objects.filter(user=request.user).first()
    headers = {"Authorization": f"Bearer {token.access_token}"}
    resp = requests.get(
        f"https://api.fitbit.com/1/user/-/stressManagement/date/{date_str}.json",
        headers=headers
    )
    if resp.status_code == 200:
        arr = resp.json().get('stress', [])
        daily['stress_score'] = arr[0].get('stressScore') if arr else None
    else:
        daily['stress_score'] = None

    return render(request, 'fitbit_app/analytics.html', {
        'date': date_str,
        'data_json': json.dumps(records),
        'daily': daily,
    })

# Lista de configuraciones biométricas
def biometric_config_list(request, case_study_id):
    case_study = get_object_or_404(CaseStudy, pk=case_study_id)
    configs = BiometricAnalysisConfig.objects.filter(case_study=case_study)
    return render(request, 'wearabletracking/biometric_config_list.html', {
        'case_study': case_study,
        'object_list': configs,
        'case_study_id': case_study_id,
    })
# def biometric_config_list(request, case_study_id):
#     # Simulación de datos
#     configs = [
#         {
#             'id': 1,
#             'created_at': '2025-07-25',
#             'title': 'Config ejemplo 1',
#             'default_metrics': ['fc', 'pasos'],
#             'default_chart_type': 'line',
#             'freeze': False,
#             'active': True,
#         },
#         {
#             'id': 2,
#             'created_at': '2025-07-20',
#             'title': 'Config ejemplo 2',
#             'default_metrics': ['calorias'],
#             'default_chart_type': 'bar',
#             'freeze': True,
#             'active': False,
#         },
#     ]
#     return render(request, 'wearabletracking/biometric_config_list.html', {
#         'case_study': {'id': case_study_id},
#         'object_list': configs,
#         'case_study_id': case_study_id,
#     })

# Formulario para crear nueva configuración biométrica
@login_required
def biometric_config_create(request, case_study_id):
    case_study = get_object_or_404(CaseStudy, pk=case_study_id)
    if request.method == 'POST':
        # Recoge los datos del formulario visual
        metrics = request.POST.get('metrics', '')
        chart_type = request.POST.get('chart_type', 'line')
        title = request.POST.get('title', 'Configuración biométrica')
        description = request.POST.get('description', '')
        metrics_list = [m for m in metrics.split(',') if m]

        BiometricAnalysisConfig.objects.create(
            case_study=case_study,
            user=request.user,
            default_metrics=metrics_list,
            default_chart_type=chart_type,
            active=False,
            # Si tienes title y description en el modelo, añádelos aquí
            title=title,
            description=description,
        )
        return redirect('wearabletracking:biometric_config_list', case_study_id=case_study_id)
    return render(request, 'wearabletracking/biometric_config_form.html', {
        'case_study': case_study,
        'case_study_id': case_study_id,
    })
# def biometric_config_create(request, case_study_id):
#     # Simulación: no uses BiometricAnalysisConfigForm ni lógica de guardado
#     case_study = {'id': case_study_id, 'title': 'Estudio de ejemplo'}
#     return render(request, 'wearabletracking/biometric_config_form.html', {
#         'case_study': case_study,
#         'case_study_id': case_study_id,
#         # Puedes pasar datos simulados si quieres mostrar valores por defecto
#     })

# Detalle de configuración biométrica
def biometric_config_detail(request, config_id):
    config = get_object_or_404(BiometricAnalysisConfig, pk=config_id)
    return render(request, 'wearabletracking/biometric_config_detail.html', {
        'config': config,
        'case_study_id': config.case_study.id,
    })

# Activar configuración biométrica
def biometric_config_activate(request, config_id):
    config = get_object_or_404(BiometricAnalysisConfig, pk=config_id)
    # Cambia el estado: si está activa, desactívala; si no, actívala
    config.active = not config.active
    config.save()
    return redirect('wearabletracking:biometric_config_list', case_study_id=config.case_study.id)

@login_required
@login_required
def biometric_config_edit(request, config_id):
    config = get_object_or_404(BiometricAnalysisConfig, pk=config_id)
    if config.freeze:
        return redirect('wearabletracking:biometric_config_detail', config_id=config.id)
        
    if request.method == 'POST':
        metrics = request.POST.get('metrics', '')
        chart_type = request.POST.get('chart_type', 'line')
        title = request.POST.get('title', config.title)
        description = request.POST.get('description', config.description)
        
        metrics_list = [m for m in metrics.split(',') if m]
        
        config.title = title
        config.description = description
        config.default_metrics = metrics_list
        config.default_chart_type = chart_type
        config.save()
        
        return redirect('wearabletracking:biometric_config_list', case_study_id=config.case_study.id)
        
    return render(request, 'wearabletracking/biometric_config_form.html', {
        'case_study': config.case_study,
        'case_study_id': config.case_study.id,
        'config': config,
        'edit_mode': True,
    })

# Lista de reportes biométricos de una ejecución
def biometric_report_list(request, execution_id):
    execution = get_object_or_404(Execution, pk=execution_id)
    report = getattr(execution, 'biometric_report', None)
    return render(request, 'wearabletracking/biometric_report_list.html', {
        'execution': execution,
        'report': report,
    })

# Detalle de reporte biométrico
def biometric_report_detail(request, report_id):
    """Vista para mostrar los resultados de un análisis biométrico"""
    report = get_object_or_404(BiometricAnalysisReport, pk=report_id)
    
    # Extraer datos para el gráfico
    chart_data = {}
    chart_labels = []
    stats = {}
    datasets = []
    
    if hasattr(report, 'extra_data') and report.extra_data:
        stats = report.extra_data.get('stats', {})
        chart_data = report.extra_data.get('chart_data', {})
        chart_labels = report.extra_data.get('chart_labels', [])
        
        # Si no hay etiquetas pero hay datos, generamos etiquetas numéricas
        if not chart_labels and chart_data:
            first_metric = next(iter(chart_data))
            chart_labels = list(range(1, len(chart_data[first_metric]) + 1))
    
        # Preparar datasets para Chart.js
        for metric, values in chart_data.items():
            color = get_color_for_metric(metric)
            datasets.append({
                'label': dict(BiometricAnalysisConfig.METRIC_CHOICES).get(metric, metric),
                'data': values,
                'borderColor': color,
                'backgroundColor': color + '33',  # Añadir transparencia
                'fill': report.chart_type == 'radar',
                'tension': 0.1
            })
    
    context = {
        'report': report,
        'chart_labels': chart_labels,
        'datasets': datasets,
        'stats': stats
    }
    
    return render(request, 'wearabletracking/biometric_report_detail.html', context)

def get_color_for_metric(metric):
    """Retorna un color consistente para cada métrica"""
    colors = {
        'fc': '#FF6384',           # Rojo para frecuencia cardíaca
        'pasos': '#36A2EB',        # Azul para pasos
        'calorias': '#FFCE56',     # Amarillo para calorías
        'zona_activa': '#4BC0C0',  # Verde azulado para zona activa
        'sedentario': '#9966FF',   # Púrpura para sedentario
        'ratio_fc_pasos': '#FF9F40', # Naranja para ratio FC/pasos
        'cvl': '#C9CBCF',          # Gris para CVL
        'sdnn': '#7FC97F'          # Verde para SDNN
    }
    # Devuelve un color basado en la métrica o genera uno aleatorio pero consistente
    return colors.get(metric, '#' + hex(hash(metric) % 0xffffff)[2:].zfill(6))

# Descargar reporte biométrico (PDF)
def biometric_report_download(request, report_id):
    """Vista para descargar los datos del reporte biométrico"""
    report = get_object_or_404(BiometricAnalysisReport, pk=report_id)
    file_path = report.get_merged_file_path()
    
    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    
    # Si no hay archivo, devolver error
    return HttpResponse("No se encontró el archivo de datos", status=404)

@login_required
def biometric_config_delete(request, config_id):
    config = get_object_or_404(BiometricAnalysisConfig, pk=config_id)
    case_study_id = config.case_study.id
    if not config.freeze:  # Solo permite borrar si no está congelada
        config.delete()
    return redirect('wearabletracking:biometric_config_list', case_study_id=case_study_id)