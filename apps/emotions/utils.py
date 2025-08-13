import os
import pandas as pd
import numpy as np
import logging
from .models import EmotionAnalysisReport

REQUIRED_COLUMNS = ['timestamp', 'emocion']

def procesar_analisis_emociones(execution):
    reports = []

    # Comprobación de configuración
    if not execution.monitoring or not getattr(execution.monitoring, 'use_emotions_data', False):
        logging.warning("El análisis de emociones no está activado para esta ejecución")
        return reports

    emotions_filename = getattr(execution.monitoring, 'emotions_filename', 'registros_emociones.csv')
    if not emotions_filename:
        logging.error("No se especificó un nombre de archivo para datos de emociones")
        return reports

    for scenario in execution.scenarios_to_study:
        try:
            scenario_path = os.path.join(execution.exp_folder_complete_path, scenario)
            emotions_file_path = os.path.join(scenario_path, emotions_filename)

            if not os.path.exists(emotions_file_path):
                logging.warning(f"[{scenario}] No se encontró el archivo de emociones: {emotions_file_path}")
                continue

            report, _ = EmotionAnalysisReport.objects.get_or_create(
                execution=execution,
                scenario=scenario,
                defaults={'title': f"Análisis de emociones - {scenario}", 'emotions_file': emotions_filename}
            )

            if process_emotions_data(report):
                reports.append(report)
                logging.info(f"[{scenario}] Procesado OK")
            else:
                logging.error(f"[{scenario}] Error procesando datos de emociones")

        except Exception as e:
            logging.exception(f"[{scenario}] Excepción procesando emociones: {e}")

    return reports


def _read_emotions_csv(file_path: str) -> pd.DataFrame | None:
    """
    Lector robusto para CSV con columnas: timestamp (ISO8601) y emocion (str).
    Añade elapsed_ms relativo al primer timestamp.
    """
    if not os.path.exists(file_path):
        logging.error(f"No existe el archivo de emociones: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        logging.error(f"Error leyendo CSV ({file_path}): {e}")
        return None

    # Normaliza nombres
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        present = [c for c in REQUIRED_COLUMNS if c in df.columns]
        logging.error(f"Faltan columnas requeridas. Presentes: {present}, Faltantes: {missing}")
        return None

    # Limpieza básica
    df['emocion'] = df['emocion'].astype(str).str.strip()
    df = df[df['emocion'].notna() & (df['emocion'] != '')]

    # Parseo de timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    if df.empty:
        logging.warning(f"{os.path.basename(file_path)} sin datos válidos tras limpieza.")
        return None

    # elapsed_ms relativo al primer timestamp (para métricas temporales)
    t0 = df['timestamp'].min()
    df['elapsed_ms'] = (df['timestamp'] - t0).dt.total_seconds() * 1000.0

    return df


def process_emotions_data(report) -> bool:
    try:
        file_path = report.get_emotions_file_path()
        if not file_path:
            logging.error("Ruta al archivo de emociones es None/vacía")
            return False

        df = _read_emotions_csv(file_path)
        if df is None or df.empty:
            return False

        metrics: dict[str, object] = {}

        # Conteo y porcentajes
        counts = df['emocion'].value_counts()
        emotions_count = {str(k): int(v) for k, v in counts.to_dict().items()}
        metrics['emotions_count'] = emotions_count

        total = int(sum(emotions_count.values()))
        if total == 0:
            logging.warning("No hay emociones contabilizadas (total=0)")
            return False

        emotion_percentages = {k: round((v / total) * 100.0, 2) for k, v in emotions_count.items()}
        metrics['emotion_percentages'] = emotion_percentages

        dominant_emotion = max(emotions_count, key=emotions_count.get)
        metrics['dominant_emotion'] = dominant_emotion
        metrics['dominant_percentage'] = float(emotion_percentages[dominant_emotion])

        # Duración y segmentación temporal (si hay al menos 2 muestras)
        if df['elapsed_ms'].notna().any():
            duration_ms = float(df['elapsed_ms'].max())
            metrics['duration_ms'] = int(duration_ms)
            metrics['duration_seconds'] = round(duration_ms / 1000.0, 2)

            # Segmentos de 10 s (si hay más de un punto, tiene más sentido)
            df['segment'] = (df['elapsed_ms'] // 10_000).astype(int)
            segment_emotions = {
                int(s): {str(k): int(v) for k, v in g['emocion'].value_counts().to_dict().items()}
                for s, g in df.groupby('segment')
            }
            metrics['time_segments'] = segment_emotions
        else:
            metrics['time_segments'] = {}

        # Transiciones entre emociones consecutivas
        s = df['emocion'].astype(str)
        transitions = (s.shift(1) + '->' + s)[1:]
        transitions = transitions[~transitions.str.contains('^nan->', na=True)]
        trans_counts = transitions.value_counts().to_dict()
        metrics['emotion_transitions'] = {str(k): int(v) for k, v in trans_counts.items()}

        # Guarda métricas
        report.extra_data = metrics
        report.save(update_fields=['extra_data'])

        # Gráfico
        generate_emotion_chart(report, df)

        return True

    except Exception as e:
        logging.exception(f"Error procesando datos de emociones: {e}")
        return False


def generate_emotion_chart(report, df: pd.DataFrame) -> None:
    """
    Gráfico de pastel con distribución de emociones (backend Agg).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt

        base_dir = os.path.dirname(report.get_emotions_file_path())
        charts_dir = os.path.join(base_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        counts = df['emocion'].value_counts()
        if counts.empty:
            logging.warning("Sin datos para graficar emociones.")
            return

        fig = plt.figure(figsize=(10, 6))
        ax = counts.plot.pie(autopct='%1.1f%%', startangle=90)
        ax.set_ylabel('')
        ax.set_title('Distribución de Emociones')
        ax.axis('equal')

        chart_path = os.path.join(charts_dir, f'emotions_chart_{report.scenario}.png')
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close(fig)

        extra = report.extra_data or {}
        # ruta relativa respecto a la carpeta de la ejecución (dos niveles por encima del CSV)
        rel_path = os.path.relpath(
            chart_path,
            os.path.dirname(os.path.dirname(report.get_emotions_file_path()))
        )
        extra['chart_path'] = rel_path
        report.extra_data = extra
        report.save(update_fields=['extra_data'])

    except Exception as e:
        logging.exception(f"Error generando gráfico de emociones: {e}")
