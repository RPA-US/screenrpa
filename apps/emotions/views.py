from django.shortcuts import render
import threading, io, zipfile, json, cv2
from datetime import datetime
from django.http import JsonResponse, HttpResponse
from deepface import DeepFace

# Create your views here.

def emotion_home(request):
    return render(request, 'emotions/record.html')

# Estado global (se reinicia cada vez que llamas a /start/)
_running = False
_cap = None
_worker_thread = None
_data_points = []
_properties = {}
_first_ts = None

def _emotion_worker():
    global _running, _cap, _data_points, _properties, _first_ts
    while _running:
        ret, frame = _cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False
            )[0]
            dominant = result['dominant_emotion']
            confidence = result['emotion'][dominant]
        except Exception:
            dominant, confidence = "No detectado", 0.0

        now_ts = datetime.now().timestamp()
        if _first_ts is None:
            _first_ts = now_ts
            _properties = {
                "RecordingStart": datetime.fromtimestamp(now_ts).isoformat(),
                "TimeZone": datetime.now().astimezone().tzname()
            }

        elapsed_ms = int((now_ts - _first_ts) * 1000)
        _data_points.append((dominant, f"{confidence:.4f}", elapsed_ms))

    # Liberar cámara al parar
    if _cap:
        _cap.release()

def start_record(request):
    """Inicia hilo de captura y análisis."""
    global _running, _cap, _worker_thread, _data_points, _properties, _first_ts

    if _running:
        return JsonResponse({'status': 'already running'})

    _cap = cv2.VideoCapture(0)
    if not _cap.isOpened():
        return JsonResponse({'status': 'camera error'}, status=500)

    # Reset de estado
    _running = True
    _data_points = []
    _properties = {}
    _first_ts = None

    _worker_thread = threading.Thread(target=_emotion_worker, daemon=True)
    _worker_thread.start()
    return JsonResponse({'status': 'started'})

def stop_record(request):
    """Detiene la grabación."""
    global _running
    if not _running:
        return JsonResponse({'status': 'not running'})
    _running = False
    return JsonResponse({'status': 'stopped'})

def download_record(request):
    """
    Empaqueta data en un ZIP (CSV + JSON) y lo devuelve para descarga.
    """
    global _data_points, _properties

    # Generar CSV en memoria
    csv_lines = ['Emotion,Confidence,ElapsedMs']
    for emo, conf, elapsed in _data_points:
        csv_lines.append(f"{emo},{conf},{elapsed}")
    csv_content = "\n".join(csv_lines)

    # JSON de propiedades
    json_props = json.dumps(_properties, ensure_ascii=False, indent=2)

    # Crear ZIP en un buffer
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('emotion_data.csv', csv_content)
        z.writestr('properties.json', json_props)
    buf.seek(0)

    # Responder con el ZIP
    resp = HttpResponse(buf.read(), content_type='application/zip')
    resp['Content-Disposition'] = 'attachment; filename="emotions.zip"'
    return resp