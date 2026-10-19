"""
AquaSense-Agent — Streamlit Demo
Deploy en Streamlit Cloud: https://share.streamlit.io
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from datetime import datetime

from dataset_manager import dataset_is_ready, download_dataset

ROOT = Path(__file__).resolve().parent
OPERATION_LOCK = threading.Lock()

# Configuración de página
st.set_page_config(
    page_title="AquaSense-Agent — Detector de Fugas",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Carga de modelo y datos (cacheado)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    """Carga el bundle del modelo entrenado."""
    bundle_path = ROOT / "models" / "leak_classifier.joblib"
    if not bundle_path.exists():
        return None, None, None, None
    bundle = joblib.load(bundle_path)
    return (
        bundle.get("model"),
        bundle.get("feature_names", []),
        bundle.get("umbral", 0.5),
        bundle.get("percentil_calibracion", 99.9)
    )

def load_full_dataset():
    """Intenta cargar el dataset completo LeakDB."""
    try:
        from data_loader import load_dataset
        data_path = ROOT / "data"
        if data_path.exists() and any(data_path.iterdir()):
            return load_dataset(data_path)
    except Exception as e:
        st.warning(f"No se pudo cargar dataset completo: {e}")
    return None

def load_demo_dataset():
    """Carga el dataset demo ligero (Scenario-7, 2000 pasos)."""
    try:
        demo_dir = ROOT / "demo_data"
        if not (demo_dir / "X_demo.csv").exists():
            return None
        X = pd.read_csv(demo_dir / "X_demo.csv")
        y = pd.read_csv(demo_dir / "y_demo.csv")["leak"].values
        scenario = pd.read_csv(demo_dir / "scenario_demo.csv")["scenario"].values
        feature_names = list(X.columns)
        # Namedtuple-like object
        class DemoDS:
            pass
        ds = DemoDS()
        ds.X = X
        ds.y_int = y
        ds.scenario = scenario
        ds.feature_names = feature_names
        return ds
    except Exception as e:
        st.warning(f"No se pudo cargar dataset demo: {e}")
    return None

@st.cache_resource
def load_dataset_cached():
    """Carga dataset: completo si existe, sino demo."""
    ds = load_full_dataset()
    if ds is not None:
        st.sidebar.info("📊 Usando dataset completo (LeakDB)")
        return ds
    ds = load_demo_dataset()
    if ds is not None:
        st.sidebar.warning("⚠️ Usando dataset DEMO (solo Scenario-7, 2000 pasos)")
        return ds
    st.sidebar.error("❌ No hay datos disponibles")
    return None

@st.cache_resource
def load_reports():
    """Carga reportes de métricas."""
    reports = {}
    for name in ["train_report.json", "evaluation_report.json"]:
        p = ROOT / "logs" / name
        if p.exists():
            reports[name] = json.loads(p.read_text(encoding="utf-8"))
    return reports

# -----------------------------------------------------------------------------
# Panel de ejecución y descarga
# -----------------------------------------------------------------------------
panel = st.sidebar.radio(
    "Sección",
    ["Demo y análisis", "Ejecutar programa"],
    index=1,
    key="main_section",
)

if panel == "Ejecutar programa":
    st.title("⚙️ Ejecutar AquaSense-Agent")
    st.caption("Descarga LeakDB y ejecuta el pipeline completo o una fase individual.")
    data_dir = ROOT / "data"

    if dataset_is_ready(data_dir):
        st.success("Dataset Hanoi_CMH completo disponible en `data/`.")
    else:
        st.warning(
            "No se encuentra el dataset completo. La descarga ocupa aproximadamente "
            "76 MB y la extracción alrededor de 500 MB."
        )
        if st.button("⬇️ Descargar y extraer Hanoi_CMH", type="primary"):
            if not OPERATION_LOCK.acquire(blocking=False):
                st.warning("Ya hay otra operación en curso. Espera a que termine.")
            else:
                try:
                    progress_bar = st.progress(0)
                    progress_label = st.empty()

                    def update_download_progress(received: int, total: int | None) -> None:
                        received_mb = received / (1024 * 1024)
                        if total:
                            progress_bar.progress(min(received / total, 1.0))
                            total_mb = total / (1024 * 1024)
                            progress_label.caption(
                                f"Descargando: {received_mb:.1f} / {total_mb:.1f} MB"
                            )
                        else:
                            progress_label.caption(f"Descargados: {received_mb:.1f} MB")

                    with st.spinner("Descargando y extrayendo dataset…"):
                        downloaded = download_dataset(data_dir, update_download_progress)
                    load_dataset_cached.clear()
                    load_model_bundle.clear()
                    st.success(f"Dataset listo. Descargados {downloaded / (1024 * 1024):.1f} MB.")
                    st.rerun()
                except (OSError, ValueError, zipfile.BadZipFile) as error:
                    st.error(f"No se pudo descargar o extraer el dataset: {error}")
                finally:
                    OPERATION_LOCK.release()

    st.divider()
    action = st.selectbox(
        "Qué ejecutar",
        ["Todo", "Solo inspect", "Solo train", "Solo evaluate", "Solo agent"],
    )
    scenario = st.selectbox(
        "Escenario del agente",
        ["Todos los escenarios"] + [f"Scenario-{number}" for number in range(1, 11)],
        index=10,
        help="Ejecuta el agente en cada escenario y guarda un CSV/JSON separado.",
    )
    limit_agent = st.checkbox("Limitar observaciones del agente", value=True)
    agent_limit = None
    if limit_agent:
        agent_limit = st.number_input(
            "Número máximo de observaciones",
            min_value=1,
            max_value=17_520,
            value=2_000,
            step=100,
        )

    st.info(
        "El entrenamiento LOSO puede tardar varios minutos y requiere CPU y memoria. "
        "La descarga y los artefactos se guardan en el disco local del proceso."
    )
    model_path = ROOT / "models" / "leak_classifier.joblib"
    if model_path.is_file():
        st.success(f"Modelo disponible: `{model_path}`")
    else:
        st.warning(f"Modelo aún no existe; se creará aquí al ejecutar train: `{model_path}`")

    if st.button("▶️ Ejecutar", type="primary"):
        if not dataset_is_ready(data_dir):
            st.error("Primero descarga el dataset completo.")
        elif action in {"Solo evaluate", "Solo agent"} and not (
            ROOT / "models" / "leak_classifier.joblib"
        ).exists():
            st.error("No se encontró el modelo. Ejecuta primero `Solo train`.")
        elif not OPERATION_LOCK.acquire(blocking=False):
            st.warning("Ya hay otra operación en curso. Espera a que termine.")
        else:
            command = [sys.executable, str(ROOT / "main.py")]
            if action != "Todo":
                command.extend(["--solo", action.removeprefix("Solo ").lower()])
            if action in {"Todo", "Solo agent"}:
                if scenario == "Todos los escenarios":
                    command.append("--todos-escenarios")
                else:
                    command.extend(["--escenario", scenario])
                if agent_limit is not None:
                    command.extend(["--limite", str(int(agent_limit))])

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_logs = ROOT / "logs" / "streamlit_runs"
            run_logs.mkdir(parents=True, exist_ok=True)
            output_path = run_logs / f"{run_id}_{action.lower().replace(' ', '_')}.log"
            output_tail: list[str] = []
            output_view = st.empty()
            status = st.status(f"Ejecutando: {action}…", expanded=True)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                if process.stdout is None:
                    raise OSError("No se pudo capturar la salida del proceso.")

                with output_path.open("w", encoding="utf-8") as log_file:
                    for line_number, line in enumerate(process.stdout, start=1):
                        log_file.write(line)
                        log_file.flush()
                        output_tail.append(line)
                        if len(output_tail) > 100:
                            output_tail.pop(0)
                        if line_number % 10 == 0:
                            output_view.code("".join(output_tail))
                return_code = process.wait()
                output_view.code("".join(output_tail) or "El proceso no generó salida.")
                if return_code == 0:
                    status.update(label=f"{action}: completado", state="complete")
                    st.success("Ejecución completada.")
                    st.cache_resource.clear()
                else:
                    status.update(label=f"{action}: error (código {return_code})", state="error")
                    st.error(f"El proceso terminó con código {return_code}. Revisa la salida.")
                st.caption(f"Registro completo guardado en `{output_path.relative_to(ROOT)}`.")
                st.download_button(
                    "⬇️ Descargar salida completa",
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime="text/plain",
                )
            except OSError as error:
                status.update(label="No se pudo iniciar el proceso", state="error")
                st.error(f"Error al iniciar el programa: {error}")
            finally:
                OPERATION_LOCK.release()

    st.stop()

# Cargar recursos
model, features, default_umbral, default_pct = load_model_bundle()
ds = load_dataset_cached()
reports = load_reports()

# -----------------------------------------------------------------------------
# Sidebar: Configuración
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

if model is None:
    st.sidebar.warning(
        f"No se encontró el modelo en `{ROOT / 'models' / 'leak_classifier.joblib'}`. "
        "Para usar la demo o scoring, descárgalo/entrena desde «Ejecutar programa»."
    )
else:
    st.sidebar.success(f"✅ Modelo cargado desde `models/leak_classifier.joblib` ({len(features)} features)")
if ds is None:
    st.sidebar.warning("No hay dataset completo ni demo_data disponible.")
else:
    st.sidebar.success("✅ Datos disponibles")

# Modo de operación
st.sidebar.caption(
    "Esta sección es para usar el modelo ya entrenado; entrenamiento e inspección "
    "se ejecutan desde «Ejecutar programa»."
)
modo = st.sidebar.radio(
    "Modo de operación",
    [
        "🎯 Demo: Scenario-7 (calibrado + evaluación)",
        "📊 Ver métricas oficiales (Fase 2/3)",
        "📁 Subir CSV propio (batch scoring)"
    ]
)

# Umbral
umbral = st.sidebar.slider(
    "Umbral de decisión",
    0.0, 1.0, float(default_umbral), 0.01,
    help="Probabilidad mínima para clasificar como fuga"
)

# Percentil para calibración
pct_calib = st.sidebar.slider(
    "Percentil de calibración (modo B)",
    90.0, 99.99, float(default_pct), 0.01,
    format="%.2f",
    help="Percentil del histórico 'sin fugas' para fijar umbral adaptativo"
)

# -----------------------------------------------------------------------------
# MODO 1: DEMO SCENARIO-7
# -----------------------------------------------------------------------------
if modo == "🎯 Demo: Scenario-7 (calibrado + evaluación)":
    st.title("🌊 AquaSense-Agent — Demo en Vivo")
    st.caption("Escenario 7: Fuga grande (nodo 28), calibración con 30 días limpios, evaluación en tiempo real")

    if model is None or ds is None:
        st.error("Esta demo requiere un modelo entrenado y un dataset disponible.")
        st.stop()
    from agent import LeakDetectionAgent

    # Preparar datos Scenario-7
    mask_7 = (ds.scenario == "Scenario-7").to_numpy() if hasattr(ds.scenario, 'to_numpy') else (ds.scenario == "Scenario-7")
    X7 = ds.X[mask_7].reset_index(drop=True)
    y7 = pd.Series(ds.y_int[mask_7]).reset_index(drop=True)

    # Controles de demo
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        n_calib = st.number_input("Pasos de calibración (días × 48)", 48, 2000, 1440, 48)
    with col2:
        n_eval = st.number_input("Pasos a evaluar", 100, len(X7) - n_calib, 2000, 100)
    with col3:
        velocidad = st.select_slider("Velocidad", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], 0.1)

    # Placeholders para actualización dinámica
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    table_placeholder = st.empty()
    status_placeholder = st.empty()

    # Botón de inicio
    if st.button("▶️ Iniciar Demo", type="primary", use_container_width=True):
        # Crear agente y calibrar
        agent = LeakDetectionAgent(
            model_path=ROOT / "models" / "leak_classifier.joblib",
            percentil=pct_calib
        )

        # Calibración
        with st.spinner(f"Calibrando con {n_calib} pasos (≈ {n_calib/48:.1f} días)..."):
            calib_obs = [
                {"escenario": "Scenario-7", "timestamp": str(i), "valores": row.to_dict()}
                for i, row in X7.iloc[:n_calib].iterrows()
            ]
            cal_result = agent.calibrar(calib_obs)

        status_placeholder.success(
            f"✅ Calibrado: umbral = **{cal_result['umbral_adaptado']:.4f}** "
            f"(percentil {cal_result['percentil']}, n={cal_result['n_observaciones']})"
        )

        # Evaluación paso a paso
        probs = []
        alertas = []
        reales = []
        timestamps = []

        progress = st.progress(0)
        for idx in range(n_eval):
            i = n_calib + idx
            if i >= len(X7):
                break

            row = X7.iloc[i]
            obs = {"escenario": "Scenario-7", "timestamp": str(i), "valores": row.to_dict()}

            percep = agent.perceive(obs)
            decision = agent.decide(percep)
            action = agent.act(decision)

            probs.append(decision["prob_fuga"])
            alertas.append(1 if action == "ALERTA_FUGA" else 0)
            reales.append(int(y7.iloc[i]))
            timestamps.append(i)

            # Actualizar cada 50 pasos o al final
            if (idx + 1) % 50 == 0 or idx == n_eval - 1:
                df_live = pd.DataFrame({
                    "t": timestamps,
                    "prob_fuga": probs,
                    "alerta": alertas,
                    "real": reales
                })

                with chart_placeholder.container():
                    st.line_chart(df_live.set_index("t")["prob_fuga"], height=300)

                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Pasos evaluados", len(df_live))
                    m2.metric("Alertas", int(df_live["alerta"].sum()))
                    m3.metric("Fugas reales", int(df_live["real"].sum()))
                    if df_live["real"].sum() > 0:
                        tp = ((df_live["alerta"] == 1) & (df_live["real"] == 1)).sum()
                        recall = tp / df_live["real"].sum()
                        m4.metric("Recall (parcial)", f"{recall:.1%}")

                with table_placeholder.container():
                    st.dataframe(df_live.tail(10), use_container_width=True, height=250)

                progress.progress((idx + 1) / n_eval)

        # Resumen final
        df_final = pd.DataFrame({
            "t": timestamps,
            "prob_fuga": probs,
            "alerta": alertas,
            "real": reales
        })

        tp = ((df_final["alerta"] == 1) & (df_final["real"] == 1)).sum()
        fp = ((df_final["alerta"] == 1) & (df_final["real"] == 0)).sum()
        fn = ((df_final["alerta"] == 0) & (df_final["real"] == 1)).sum()
        tn = ((df_final["alerta"] == 0) & (df_final["real"] == 0)).sum()

        st.divider()
        st.subheader("📊 Resumen Final")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Exactitud", f"{(tp+tn)/len(df_final):.2%}")
        c2.metric("Precision", f"{tp/(tp+fp):.2%}" if (tp+fp) else "N/A")
        c3.metric("Recall", f"{tp/(tp+fn):.2%}" if (tp+fn) else "N/A")
        c4.metric("F1", f"{2*tp/(2*tp+fp+fn):.2%}" if (2*tp+fp+fn) else "N/A")
        c5.metric("Falsas alarmas", fp)

        # Descargar resultados
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            csv,
            f"aquasense_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

# -----------------------------------------------------------------------------
# MODO 2: MÉTRICAS OFICIALES
# -----------------------------------------------------------------------------
elif modo == "📊 Ver métricas oficiales (Fase 2/3)":
    st.title("📊 Métricas Oficiales — AquaSense-Agent")

    if not reports:
        st.warning("No hay reportes generados. Ejecuta `python train.py` y `python evaluate.py`.")
    else:
        train_r = reports.get("train_report.json", {})
        eval_r = reports.get("evaluation_report.json", {})

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Resumen Ejecutivo",
            "🔴 Escenario A: Red Nueva",
            "🔵 Escenario B: Red Conocida",
            "🟢 Escenario C: Histórico Propio"
        ])

        with tab1:
            st.subheader("Comparativa de Despliegue")
            dep = train_r.get("validacion", {}).get("escenarios_despliegue", {})

            # Tabla resumen
            rows = []
            for key, label in [
                ("A_red_nueva", "A) Red Nueva"),
                ("B_red_conocida", "B) Red Conocida"),
                ("C_red_conocida_con_historial", "C) Red Conocida + Hist. Propio")
            ]:
                if key in dep:
                    m = dep[key].get("metricas", dep[key].get("global", {}))
                    rows.append({
                        "Escenario": label,
                        "Exactitud": f"{m.get('accuracy', 0):.2%}",
                        "Precision": f"{m.get('precision', 0):.2%}",
                        "Recall": f"{m.get('recall', 0):.2%}",
                        "F1": f"{m.get('f1', 0):.2%}",
                        "AUC": f"{m.get('auc', 0):.4f}" if 'auc' in m else "—",
                        "Meta 80%": "✅" if m.get('accuracy', 0) >= 0.8 else "❌"
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.info("""
            **Interpretación:**
            - **Escenario A**: Despliegue en red sin histórico → umbral global 0.5
            - **Escenario B** (recomendado): Operador tiene 30 días limpios → umbral adaptativo p99.9
            - **Escenario C**: Además entrena con histórico propio → **no mejora** vs B
            """)

        with tab2:
            st.subheader("Escenario A — Red Nueva (umbral global 0.5)")
            a = dep.get("A_red_nueva", {})
            if a:
                m = a.get("metricas", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Exactitud", f"{m.get('accuracy',0):.2%}")
                c2.metric("Precision", f"{m.get('precision',0):.2%}")
                c3.metric("Recall", f"{m.get('recall',0):.2%}")
                c4.metric("F1", f"{m.get('f1',0):.2%}")

                st.write("**Matriz de confusión**")
                cm = a.get("matriz_confusion", [[0,0],[0,0]])
                st.dataframe(pd.DataFrame(cm, index=["Real: Normal", "Real: Fuga"],
                                         columns=["Pred: Normal", "Pred: Fuga"]))

                st.write("**Por escenario (LOOCV)**")
                por_esc = a.get("por_escenario", {})
                if por_esc:
                    df_esc = pd.DataFrame([
                        {"Escenario": k, **{k2: f"{v2:.4f}" if isinstance(v2, float) else v2
                                             for k2, v2 in v.items() if k2 != "umbral"},
                         "Umbral": f"{v.get('umbral', 0):.4f}"}
                        for k, v in por_esc.items()
                    ])
                    st.dataframe(df_esc, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Escenario B — Red Conocida (p99.9 histórico limpio)")
            b = dep.get("B_red_conocida", {})
            if b:
                m = b.get("metricas", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Exactitud", f"{m.get('accuracy',0):.2%}")
                c2.metric("Precision", f"{m.get('precision',0):.2%}")
                c3.metric("Recall", f"{m.get('recall',0):.2%}")
                c4.metric("F1", f"{m.get('f1',0):.2%}")

                st.write("**Matriz de confusión**")
                cm = b.get("matriz_confusion", [[0,0],[0,0]])
                st.dataframe(pd.DataFrame(cm, index=["Real: Normal", "Real: Fuga"],
                                         columns=["Pred: Normal", "Pred: Fuga"]))

                st.write("**Umbrales adaptativos por red**")
                por_esc = b.get("por_escenario", {})
                if por_esc:
                    df_umb = pd.DataFrame([
                        {"Escenario": k, "Umbral": f"{v.get('umbral',0):.4f}",
                         "Exactitud": f"{v.get('accuracy',0):.4f}",
                         "Precision": f"{v.get('precision',0):.4f}",
                         "Recall": f"{v.get('recall',0):.4f}",
                         "F1": f"{v.get('f1',0):.4f}",
                         "Control": "✅" if v.get("n_reales_fuga", 0) == 0 else ""}
                        for k, v in por_esc.items()
                    ])
                    st.dataframe(df_umb, use_container_width=True, hide_index=True)

        with tab4:
            st.subheader("Escenario C — Red Conocida + Histórico Propio en Entrenamiento")
            c = dep.get("C_red_conocida_con_historial", {})
            if c:
                g = c.get("global", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Exactitud", f"{g.get('accuracy',0):.2%}")
                c2.metric("Precision", f"{g.get('precision',0):.2%}")
                c3.metric("Recall", f"{g.get('recall',0):.2%}")
                c4.metric("F1", f"{g.get('f1',0):.2%}")

                st.warning("""
                **Resultado: No mejora vs Escenario B**  
                F1 0.6016 vs 0.6192 (B). El histórico limpio solo aporta ejemplos negativos,
                que desplazan el criterio sin informar cómo es una fuga *en esa red*.
                """)

                if "por_escenario" in c:
                    df_c = pd.DataFrame([
                        {"Escenario": k, **{k2: f"{v2:.4f}" if isinstance(v2, float) else v2
                                             for k2, v2 in v.items()}}
                        for k, v in c["por_escenario"].items()
                    ])
                    st.dataframe(df_c, use_container_width=True, hide_index=True)
            else:
                st.info("Escenario C no ejecutado. Usa `python train.py --historial-propio`.")

# -----------------------------------------------------------------------------
# MODO 3: SCORING BATCH (CSV)
# -----------------------------------------------------------------------------
else:
    st.title("📁 Batch Scoring — Subir CSV")
    st.caption("Sube un archivo CSV con observaciones para obtener probabilidad de fuga por fila.")
    if model is None:
        st.error("El scoring requiere un modelo entrenado. Usa «Ejecutar programa» → «Solo train».")
        st.stop()
    from agent import LeakDetectionAgent

    st.markdown("""
    **Formato esperado:** CSV con columnas:
    - `timestamp` (opcional)
    - `escenario` (opcional)
    - `press_1` ... `press_31` (31 columnas)
    - `flow_1` ... `flow_34` (34 columnas)
    = **65 features + metadata**
    """)

    uploaded = st.file_uploader("Seleccionar CSV", type=["csv"])

    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.success(f"Cargado: {len(df_in)} filas × {len(df_in.columns)} columnas")

            # Validar columnas mínimas
            missing = [f for f in features if f not in df_in.columns]
            if missing:
                st.error(f"Faltan {len(missing)} features requeridas: {missing[:10]}...")
                st.stop()

            # Crear agente
            agent = LeakDetectionAgent(
                model_path=ROOT / "models" / "leak_classifier.joblib",
                percentil=pct_calib
            )

            # Opción: calibrar con primeras N filas si el usuario quiere
            usar_calib = st.checkbox("Calibrar con primeras N filas (asumir limpias)", value=False)
            n_calib = st.number_input("N filas calibración", 10, min(1000, len(df_in)), 100) if usar_calib else 0

            if usar_calib and n_calib > 0:
                calib_obs = [
                    {"escenario": row.get("escenario", "upload"), "timestamp": str(i), "valores": row.to_dict()}
                    for i, row in df_in.iloc[:n_calib].iterrows()
                ]
                cal = agent.calibrar(calib_obs)
                st.info(f"Calibrado: umbral = {cal['umbral_adaptado']:.4f}")

            # Scoring
            if st.button("🚀 Ejecutar Scoring", type="primary"):
                start_idx = n_calib if usar_calib else 0
                results = []

                progress = st.progress(0)
                for idx, row in df_in.iloc[start_idx:].iterrows():
                    obs = {
                        "escenario": row.get("escenario", "upload"),
                        "timestamp": str(row.get("timestamp", idx)),
                        "valores": row.to_dict()
                    }
                    percep = agent.perceive(obs)
                    decision = agent.decide(percep)
                    action = agent.act(decision)

                    results.append({
                        "timestamp": row.get("timestamp", idx),
                        "prob_fuga": decision["prob_fuga"],
                        "umbral": decision["umbral"],
                        "decision": decision["decision"],
                        "accion": action
                    })
                    progress.progress((idx - start_idx + 1) / (len(df_in) - start_idx))

                df_out = pd.DataFrame(results)
                st.success(f"Completado: {len(df_out)} predicciones")

                # Métricas rápidas si hay etiqueta real
                if "leak" in df_in.columns or "label" in df_in.columns:
                    y_true = df_in.iloc[start_idx:]["leak" if "leak" in df_in.columns else "label"].astype(int).values
                    y_pred = (df_out["prob_fuga"] >= umbral).astype(int).values
                    tp = ((y_pred == 1) & (y_true == 1)).sum()
                    fp = ((y_pred == 1) & (y_true == 0)).sum()
                    fn = ((y_pred == 0) & (y_true == 1)).sum()
                    tn = ((y_pred == 0) & (y_true == 0)).sum()
                    st.metric("Exactitud", f"{(tp+tn)/len(y_true):.2%}")
                    st.metric("Precision", f"{tp/(tp+fp):.2%}" if (tp+fp) else "N/A")
                    st.metric("Recall", f"{tp/(tp+fn):.2%}" if (tp+fn) else "N/A")

                # Mostrar y descargar
                st.dataframe(df_out.head(50), use_container_width=True)
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar predicciones (CSV)",
                    csv,
                    f"aquasense_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error procesando CSV: {e}")
            st.exception(e)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption("""
AquaSense-Agent v1.2 | LeakDB Hanoi_CMH | RandomForest 150 árboles | 
Validación LOSO (10 folds) | Umbral adaptativo p99.9 | 
[Repositorio](https://github.com/tu-usuario/AquaSense-Agent) | 
[Documentación](FASE3_RESULTADOS.md)
""")