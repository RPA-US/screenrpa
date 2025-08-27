import os
import pandas as pd
import numpy as np
import logging
from .models import EmotionAnalysisReport

REQUIRED_COLUMNS = ['timestamp', 'emocion']
WEARABLE_COLUMNS = [
    "fc", "pasos", "calorias", "zona_activa", "sedentario",
    "ratio_fc_pasos", "cvl", "spo2", "temperatura", "hrv", "sdnn"
]
NUMERIC_WEARABLE = [
    "fc", "pasos", "calorias", "zona_activa", "ratio_fc_pasos",
    "cvl", "spo2", "temperatura", "hrv", "sdnn"
]
MERGED_EMOTIONS_FILENAME = "merged_emotions_wearable.csv"

def procesar_analisis_emociones(execution):
    reports = []

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
            merged_file_path   = os.path.join(scenario_path, MERGED_EMOTIONS_FILENAME)

            # 1) Preferimos el fusionado si existe
            if os.path.exists(merged_file_path):
                selected_filename = MERGED_EMOTIONS_FILENAME
                has_merged = True
            elif os.path.exists(emotions_file_path):
                selected_filename = emotions_filename
                has_merged = False
            else:
                logging.warning(f"[{scenario}] No se encontró ningún CSV en {scenario_path}")
                continue

            report, _ = EmotionAnalysisReport.objects.get_or_create(
                execution=execution,
                scenario=scenario,
                defaults={
                    'title': f"Análisis de emociones - {scenario}",
                    'emotions_file': selected_filename,
                    'merged_file': MERGED_EMOTIONS_FILENAME,
                    'has_merged_data': has_merged,
                }
            )

            # 2) Sincronizamos campos si han cambiado
            to_update = []
            if report.emotions_file != selected_filename:
                report.emotions_file = selected_filename
                to_update.append('emotions_file')
            if report.merged_file != MERGED_EMOTIONS_FILENAME:
                report.merged_file = MERGED_EMOTIONS_FILENAME
                to_update.append('merged_file')
            if report.has_merged_data != has_merged:
                report.has_merged_data = has_merged
                to_update.append('has_merged_data')
            if to_update:
                report.save(update_fields=to_update)

            # 3) Procesamos
            if process_emotions_data(report):
                reports.append(report)
                logging.info(f"[{scenario}] Procesado OK usando '{selected_filename}' (has_merged={has_merged})")
            else:
                logging.error(f"[{scenario}] Error procesando datos ({selected_filename})")

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

        # --------- EMOCIONES BÁSICAS ---------
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

        # Duración y segmentación temporal
        if df['elapsed_ms'].notna().any():
            duration_ms = float(df['elapsed_ms'].max())
            metrics['duration_ms'] = int(duration_ms)
            metrics['duration_seconds'] = round(duration_ms / 1000.0, 2)

            df['segment'] = (df['elapsed_ms'] // 10_000).astype(int)
            segment_emotions = {
                int(s): {str(k): int(v) for k, v in g['emocion'].value_counts().to_dict().items()}
                for s, g in df.groupby('segment')
            }
            metrics['time_segments'] = segment_emotions
        else:
            metrics['time_segments'] = {}

        # Transiciones
        s = df['emocion'].astype(str)
        transitions = (s.shift(1) + '->' + s)[1:]
        transitions = transitions[~transitions.str.contains('^nan->', na=True)]
        trans_counts = transitions.value_counts().to_dict()
        metrics['emotion_transitions'] = {str(k): int(v) for k, v in trans_counts.items()}

        # --------- MULTIMODAL (WEARABLE × EMOCIÓN) ---------
        have_wearable = any(col in df.columns for col in WEARABLE_COLUMNS)
        if have_wearable:
            multimodal: dict[str, object] = {"_has_data": True}  # bandera para que sea truthy

            # A) Medias por emoción
            cols_present = [c for c in NUMERIC_WEARABLE if c in df.columns]
            if "sedentario" in df.columns:
                if not np.issubdtype(df["sedentario"].dtype, np.number):
                    df["sedentario_bin"] = pd.to_numeric(df["sedentario"], errors="coerce").fillna(0).astype(float)
                else:
                    df["sedentario_bin"] = df["sedentario"].astype(float)
                cols_present_with_sed = cols_present + ["sedentario_bin"]
            else:
                cols_present_with_sed = cols_present

            if cols_present_with_sed:
                avg_df = df.groupby("emocion")[cols_present_with_sed].mean(numeric_only=True).round(3)
                multimodal["avg_by_emotion"] = {
                    str(idx): {str(k): float(v) for k, v in row.dropna().to_dict().items()}
                    for idx, row in avg_df.iterrows()
                }

            # B) Boxplots
            boxplot = {}
            if "fc" in df.columns:
                boxplot["fc_by_emotion"] = {
                    str(emo): [float(x) for x in g["fc"].dropna().tolist()]
                    for emo, g in df.groupby("emocion")
                }
            if "hrv" in df.columns:
                boxplot["hrv_by_emotion"] = {
                    str(emo): [float(x) for x in g["hrv"].dropna().tolist()]
                    for emo, g in df.groupby("emocion")
                }
            if boxplot:
                multimodal["boxplot"] = boxplot

            # C) Scatter HRV vs FC
            if "hrv" in df.columns and "fc" in df.columns:
                pts = df[["hrv", "fc", "emocion"]].dropna().head(5000)
                pts["hrv"] = pd.to_numeric(pts["hrv"], errors="coerce")
                pts["fc"]  = pd.to_numeric(pts["fc"], errors="coerce")
                pts = pts.dropna()
                multimodal["scatter_hrv_fc"] = [
                    {"hrv": float(r.hrv), "fc": float(r.fc), "emocion": str(r.emocion)}
                    for r in pts.itertuples(index=False)
                ]

            # D) Actividad vs Emoción
            if ("zona_activa" in df.columns or "pasos" in df.columns) or ("sedentario" in df.columns):
                temp = df.copy()
                z = pd.to_numeric(temp["zona_activa"], errors="coerce").fillna(0) if "zona_activa" in temp.columns else 0
                p = pd.to_numeric(temp["pasos"], errors="coerce").fillna(0) if "pasos" in temp.columns else 0
                temp["is_active"] = ((z > 0) | (p > 0)).astype(int)
                if "sedentario" in temp.columns:
                    temp["sedentario_bin"] = pd.to_numeric(temp["sedentario"], errors="coerce").fillna(0).astype(int)
                    temp.loc[temp["sedentario_bin"] == 1, "is_active"] = 0

                dist = (temp.groupby(["emocion", "is_active"]).size()
                            .unstack(fill_value=0)
                            .rename(columns={0:"sedentario",1:"activo"}))
                dist_pct = (dist.div(dist.sum(axis=1), axis=0).round(3) * 100.0).fillna(0.0)
                multimodal["activity_by_emotion"] = {
                    str(idx): {"activo": float(row.get("activo", 0.0)),
                               "sedentario": float(row.get("sedentario", 0.0))}
                    for idx, row in dist_pct.iterrows()
                }

            # E) Correlaciones
            numeric_cols = [c for c in NUMERIC_WEARABLE if c in df.columns]
            if numeric_cols:
                emo_dummies = pd.get_dummies(df["emocion"], prefix="emo")
                joined = pd.concat([df[numeric_cols].apply(pd.to_numeric, errors="coerce"), emo_dummies], axis=1)
                corr = joined.corr(method="pearson").fillna(0.0).round(3)
                emo_cols = [c for c in corr.columns if c.startswith("emo_")]
                if set(numeric_cols).issubset(corr.index) and set(emo_cols).issubset(corr.columns):
                    corr_block = corr.loc[numeric_cols, emo_cols]
                    multimodal["correlations"] = {
                        row: {col: float(corr_block.loc[row, col]) for col in emo_cols}
                        for row in corr_block.index
                    }
                    multimodal["correlations_meta"] = {
                        "emotion_columns": emo_cols,
                        "metric_columns": numeric_cols
                    }

            metrics["multimodal"] = multimodal
            # marca en el reporte (si quieres mantener coherencia con el modelo)
            if not report.has_merged_data:
                report.has_merged_data = True

        # --------- GUARDAR Y GRÁFICO ---------
        # guardamos extra_data y, si cambió, has_merged_data
        if report.has_merged_data:
            report.extra_data = metrics
            report.save(update_fields=['extra_data', 'has_merged_data'])
        else:
            report.extra_data = metrics
            report.save(update_fields=['extra_data'])

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
