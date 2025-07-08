import io
import json
import os
import zipfile
import requests
from django.shortcuts import render, redirect
from django.conf import settings
from .models import FitbitToken
from datetime import datetime, timedelta
import csv
from django.http import HttpResponse
import numpy as np
import io
import pandas as pd
from django.shortcuts import render

from django.utils import timezone

def wearable_home(request):
    token = FitbitToken.objects.first()
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

        # Elimina tokens antiguos si solo quieres uno por usuario
        FitbitToken.objects.all().delete()

        # Guardar token en la base de datos
        token = FitbitToken.objects.create(
            access_token=token_data['access_token'],
            refresh_token=token_data['refresh_token'],
            expires_in=token_data['expires_in'],
            token_type=token_data['token_type'],
            scope=token_data['scope'],
            user_id=token_data['user_id']
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
        
        token = FitbitToken.objects.first()
        headers = {"Authorization": f"Bearer {token.access_token}"}
        edad = 30  # Ajustar según usuario real
        fc_reposo = 65  # Ajustar según usuario real

        current_date = datetime.strptime(fecha_inicio, "%Y-%m-%d").date()
        end_date = datetime.strptime(fecha_fin, "%Y-%m-%d").date()

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
                temp_var = temp_resp.json().get("tempSkin", [{}])[0].get("value", "") if temp_resp.status_code == 200 else ""

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

                def fetch_data(metric):
                    url = f"https://api.fitbit.com/1/user/-/activities/{metric}/date/{fecha_str}/1d/1min.json"
                    r = requests.get(url, headers=headers)
                    key = f"activities-{metric}-intraday"
                    if r.status_code == 200:
                        return {d['time']: d['value'] for d in r.json().get(key, {}).get('dataset', [])}
                    return {}

                # Frecuencia cardíaca, pasos, calorías
                fc_data = fetch_data('heart')
                pasos_data = fetch_data('steps')
                calorias_data = fetch_data('calories')

                # Zonas activas: manejo especial
                azm_url = f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{fecha_str}/1d/1min.json"
                azm_resp = requests.get(azm_url, headers=headers)
                azm_data = {}
                if azm_resp.status_code == 200:
                    azm_json = azm_resp.json().get("activities-active-zone-minutes-intraday", [])
                    if azm_json and "minutes" in azm_json[0]:
                        for entry in azm_json[0]["minutes"]:
                            # Extrae solo la hora y minuto para alinear con otros datos
                            minute = entry["minute"][-8:]  # HH:MM:SS
                            azm_data[minute] = entry["value"].get("activeZoneMinutes", 0)

                # Unifica todos los minutos disponibles
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
                        azm_data.get(minuto, 0),  # Si no hay dato, pone 0
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

    token = FitbitToken.objects.first()
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
