from datetime import datetime

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