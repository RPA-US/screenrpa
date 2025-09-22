from django.shortcuts import render
import threading, io, zipfile, json, cv2, base64
from datetime import datetime
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from deepface import DeepFace
import pandas as pd
import os
from django.views.decorators.csrf import csrf_exempt
import time

# Create your views here.

def emotion_home(request):
    return render(request, 'emotions/record.html')

# Estado global
_running = False
_cap = None
_worker_thread = None
_data_points = []
_properties = {}
_first_ts = None
_current_emotion = "No detectado"

def _emotion_worker():
    global _running, _cap, _data_points, _properties, _first_ts, _current_emotion
    while _running:
        ret, frame = _cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False
            )[0]
            dominant = result['dominant_emotion']
            confidence = result['emotion'][dominant]
            _current_emotion = dominant
        except Exception as e:
            print(f"Error en análisis de emoción: {e}")
            dominant, confidence = "No detectado", 0.0
            _current_emotion = "No detectado"

        # Clasificación positiva/negativa/neutral
        if dominant in ["happy", "surprise"]:
            sentimiento = "positivo"
        elif dominant in ["neutral"]:
            sentimiento = "neutral"
        elif dominant in ["angry", "fear", "sad", "disgust"]:
            sentimiento = "negativo"
        else:
            sentimiento = "desconocido"

        now_ts = datetime.now().timestamp()
        if _first_ts is None:
            _first_ts = now_ts
            _properties = {
                "RecordingStart": datetime.fromtimestamp(now_ts).isoformat(),
                "TimeZone": datetime.now().astimezone().tzname()
            }

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _data_points.append({
            'timestamp': timestamp,
            'emocion': dominant,
            'sentimiento': sentimiento,  # ✅ Nueva columna
        })

        time.sleep(0.1)

    if _cap:
        _cap.release()

@csrf_exempt
def start_record(request):
    """Inicia hilo de captura y análisis."""
    global _running, _cap, _worker_thread, _data_points, _properties, _first_ts

    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    if _running:
        return JsonResponse({'status': 'already running'})

    # Probar múltiples índices de cámara
    camera_indices = [0, 1, 2]  # Probar diferentes índices
    _cap = None
    
    for idx in camera_indices:
        print(f"Probando cámara con índice {idx}")
        test_cap = cv2.VideoCapture(idx)
        
        if test_cap.isOpened():
            # Probar leer un frame
            ret, frame = test_cap.read()
            if ret:
                print(f"Cámara encontrada en índice {idx}")
                _cap = test_cap
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    if not _cap or not _cap.isOpened():
        return JsonResponse({
            'status': 'camera error',
            'message': 'No se pudo acceder a ninguna cámara. Verifica que no esté en uso por otra aplicación.'
        }, status=500)

    # Configurar cámara
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _cap.set(cv2.CAP_PROP_FPS, 30)

    # Reset de estado
    _running = True
    _data_points = []
    _properties = {}
    _first_ts = None

    _worker_thread = threading.Thread(target=_emotion_worker, daemon=True)
    _worker_thread.start()
    return JsonResponse({'status': 'started'})

@csrf_exempt
def stop_record(request):
    """Detiene la grabación."""
    global _running, _cap
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
        
    if not _running:
        return JsonResponse({'status': 'not running'})
    
    _running = False
    
    # Esperar a que termine el hilo y liberar la cámara
    if _cap:
        time.sleep(0.5)  # Dar tiempo al hilo para terminar
        _cap.release()
        _cap = None
    
    return JsonResponse({'status': 'stopped'})

def get_current_emotion(request):
    """Devuelve la emoción actual detectada."""
    global _current_emotion, _running
    return JsonResponse({
        'emotion': _current_emotion,
        'running': _running
    })

def download_record(request):
    """Descarga los registros como CSV."""
    global _data_points

    if not _data_points:
        return JsonResponse({'error': 'No hay datos para descargar'}, status=400)

    # Crear DataFrame y CSV
    df = pd.DataFrame(_data_points)
    
    # Crear directorio si no existe
    os.makedirs('datos', exist_ok=True)
    
    # Generar CSV en memoria
    csv_content = df.to_csv(index=False)
    
    # Responder con el CSV
    response = HttpResponse(csv_content, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="registros_emociones.csv"'
    return response

def video_feed(request):
    """Stream de video con detección de emociones."""
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Genera frames para el stream de video."""
    global _running, _cap, _current_emotion
    
    if not _running or not _cap:
        return
        
    while _running:
        ret, frame = _cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        # Agregar texto de emoción al frame
        cv2.putText(frame, f"Emocion: {_current_emotion}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Codificar frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def camera_test(request):
    """Función de diagnóstico para probar cámaras."""
    camera_info = []
    
    for idx in range(10):  # Probar más índices
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    camera_info.append({
                        'index': idx,
                        'status': 'OK',
                        'width': int(width),
                        'height': int(height),
                        'fps': int(fps)
                    })
                else:
                    camera_info.append({
                        'index': idx,
                        'status': 'Opened but no frame',
                        'width': 0,
                        'height': 0,
                        'fps': 0
                    })
                cap.release()
            else:
                camera_info.append({
                    'index': idx,
                    'status': 'Cannot open',
                    'width': 0,
                    'height': 0,
                    'fps': 0
                })
        except Exception as e:
            camera_info.append({
                'index': idx,
                'status': f'Error: {str(e)}',
                'width': 0,
                'height': 0,
                'fps': 0
            })
    
    return JsonResponse({
        'cameras': camera_info,
        'opencv_version': cv2.__version__
    })