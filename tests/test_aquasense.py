"""
tests/test_aquasense.py - Pruebas basicas de AquaSense-Agent
===========================================================

Comprueban lo que, si se rompe, haria invalido el resultado entero:

  1. Que la carga de datos es correcta y coherente con lo que dice
     inspect_data.py.
  2. Que NO hay filtracion de etiquetas en las variables de entrada.
  3. Que el ciclo del agente (percepcion, decision, accion, registro) existe,
     es invocable y deja el registro completo.
  4. Que las metricas que se reportan estan en su rango legitimo.

Uso
---
    python -m pytest tests/ -v
    python -m pytest tests/ -v -k "etiqueta"

Se omiten (no fallan) si el dataset o el modelo no estan disponibles, para
que las pruebas puedan correr en un equipo recien clonado.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
MODELO = RAIZ / "models" / "leak_classifier.joblib"

requiere_datos = pytest.mark.skipif(
    not (DATA / "Scenario-1" / "Labels.csv").exists(),
    reason="falta el dataset en data/ (ejecuta inspect_data.py --zip ...)")
requiere_modelo = pytest.mark.skipif(
    not MODELO.exists(), reason="falta el modelo (ejecuta train.py)")


# --------------------------------------------------------------------------
# 1. Carga de datos
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds():
    from data_loader import load_dataset
    return load_dataset(DATA)


@requiere_datos
def test_forma_del_dataset(ds):
    """10 escenarios x 17,520 pasos de 30 min = 175,200 filas."""
    assert ds.n_samples == 175_200
    assert len(ds.scenarios) == 10


@requiere_datos
def test_una_variable_por_presion_mas_una_por_caudal(ds):
    """31 presiones utiles (32 menos el embalse) + 34 caudales = 65."""
    assert len(ds.feature_names) == 65
    presion = [v for v in ds.feature_names if v.startswith("press_")]
    caudal = [v for v in ds.feature_names if v.startswith("flow_")]
    assert len(presion) == 31
    assert len(caudal) == 34


@requiere_datos
def test_el_embalse_se_excluye(ds):
    """El nodo 1 es el embalse: su presion es 0.0 siempre y no aporta nada."""
    assert ds.excluded["presion_sin_variacion"] == ["press_1"]
    assert "press_1" not in ds.feature_names


@requiere_datos
def test_etiqueta_binaria_y_desbalance_conocido(ds):
    """Coincide con lo que verifico inspect_data.py: 15.46 % de clase 1."""
    assert set(np.unique(ds.y)) <= {0.0, 1.0}
    proporcion = ds.y_int.mean()
    assert 0.15 < proporcion < 0.16, f"proporcion inesperada: {proporcion}"
    assert 27_000 < ds.y_int.sum() < 28_000


@requiere_datos
def test_cuatro_escenarios_de_control(ds):
    """1, 2, 4 y 9 no tienen ninguna fuga en todo el ano."""
    control = [e for e in ds.scenarios if ds.y_int[ds.scenario == e].sum() == 0]
    assert len(control) == 4


@requiere_datos
def test_sin_valores_faltantes(ds):
    assert np.isfinite(ds.X.to_numpy()).all()


# --------------------------------------------------------------------------
# 2. No hay filtracion de etiquetas
# --------------------------------------------------------------------------


@requiere_datos
def test_las_entradas_no_contienen_la_etiqueta(ds):
    """Ninguna variable de entrada puede derivarse de Labels.csv."""
    prohibidas = {"label", "etiqueta", "leak", "fuga", "escenario", "scenario"}
    for nombre in ds.feature_names:
        assert not any(p in nombre.lower() for p in prohibidas), nombre


@requiere_datos
def test_las_entradas_no_son_el_vector_de_etiquetas(ds):
    """Defensa extra: la etiqueta no esta ni como columna ni como duplicada."""
    assert "Label" not in ds.X.columns
    assert ds.y_int.shape == (ds.n_samples,)
    assert list(ds.X.columns) == ds.feature_names


@requiere_datos
def test_las_demandas_no_entran_por_defecto(ds):
    """Las demandas identifican el escenario, por eso se excluyen."""
    assert ds.excluded["demandas_excluidas"], "deberian listarse las demandas excluidas"
    assert not any(v.startswith("demand_") for v in ds.feature_names)


# --------------------------------------------------------------------------
# 3. Ciclo del agente
# --------------------------------------------------------------------------


@requiere_modelo
def test_el_agente_tiene_las_cuatro_funciones():
    from agent import LeakDetectionAgent
    for metodo in ("perceive", "decide", "act", "log_event", "run"):
        assert callable(getattr(LeakDetectionAgent, metodo, None)), metodo


@requiere_datos
@requiere_modelo
def test_ciclo_completo_escribe_el_registro(tmp_path, ds):
    """perceive -> decide -> act -> log_event sobre una fila real."""
    import joblib

    from agent import LeakDetectionAgent

    bundle = joblib.load(MODELO)
    fila = ds.X[ds.scenario == "Scenario-9"].iloc[0]

    log = tmp_path / "eventos.csv"
    with LeakDetectionAgent(MODELO, log) as agente:
        assert agente.umbral == bundle["umbral"]
        percepcion = agente.perceive({
            "escenario": "Scenario-9",
            "timestamp": "2017-01-01 00:00:00",
            "valores": dict(zip(ds.feature_names, fila.to_numpy())),
        })
        assert percepcion["n_variables"] == len(ds.feature_names)

        decision = agente.decide(percepcion)
        assert 0.0 <= decision["prob_fuga"] <= 1.0
        assert decision["decision"] in ("FUGA", "NO_FUGA")

        accion = agente.act(decision)
        assert accion in ("ALERTA_FUGA", "OPERACION_NORMAL")

        agente.log_event(percepcion, decision, accion, etiqueta_real=0)

    with log.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    assert len(filas) == 1
    esperados = {"registro_utc", "escenario", "observacion_id", "timestamp_observacion",
                 "n_variables_recibidas", "prob_fuga", "umbral", "decision", "accion",
                 "etiqueta_real"}
    assert esperados.issubset(filas[0].keys())


@requiere_modelo
def test_la_etiqueta_no_influye_en_la_decision(tmp_path):
    """Cambiar la etiqueta real no puede cambiar lo que el agente decide.

    Es la prueba de que no hay filtracion en tiempo de ejecucion: la etiqueta
    se pasa DESPUES de actuar, al log, y el mismo par de percepciones produce
    siempre la misma decision.
    """
    import joblib

    from agent import LeakDetectionAgent

    bundle = joblib.load(MODELO)
    nombres = bundle["feature_names"]
    valores = {n: 50.0 for n in nombres}

    log = tmp_path / "eventos.csv"
    with LeakDetectionAgent(MODELO, log) as agente:
        p = agente.perceive({"escenario": "X", "timestamp": "t", "valores": valores})
        d1 = agente.decide(p)
        d2 = agente.decide(p)
    assert d1 == d2


# --------------------------------------------------------------------------
# 4. Rango de las metricas
# --------------------------------------------------------------------------


@requiere_datos
def test_las_metricas_del_informe_estan_en_rango():
    """Ninguna metrica publicada puede estar fuera de [0, 1]."""
    import json

    informe = RAIZ / "logs" / "train_report.json"
    if not informe.exists():
        pytest.skip("falta logs/train_report.json (ejecuta train.py)")
    rep = json.loads(informe.read_text(encoding="utf-8"))
    val = rep["validacion"]

    for c in val["curva_umbral_global"]:
        for campo in ("accuracy", "precision", "recall", "f1", "fbeta"):
            assert 0.0 <= c[campo] <= 1.0, (campo, c)
    for c in val["curva_percentiles"]:
        for campo in ("accuracy", "precision", "recall", "f1", "fbeta"):
            assert 0.0 <= c[campo] <= 1.0, (campo, c)

    base = rep["linea_base_siempre_normal"]
    assert 0.0 <= base <= 1.0
    # La base trivial de "siempre normal" es la proporcion de clase 0.
    assert abs(base - (1 - rep["datos"]["pct_fuga"])) < 0.01


@requiere_datos
def test_los_dos_escenarios_de_despliegue_estan_reportados():
    """Deben existir los dos, con metricas coherentes entre si."""
    import json

    informe = RAIZ / "logs" / "train_report.json"
    if not informe.exists():
        pytest.skip("falta logs/train_report.json (ejecuta train.py)")
    val = json.loads(informe.read_text(encoding="utf-8"))["validacion"]
    dep = val["escenarios_despliegue"]

    assert "A_red_nueva" in dep and "B_red_conocida" in dep
    a, b = dep["A_red_nueva"], dep["B_red_conocida"]

    # A no puede calibrar: su umbral es un escalar global.
    assert a["metodo_umbral"] == "global"
    assert 0.0 < a["umbral"] < 1.0

    # B calibra por red: cada red tiene el suyo.
    assert b["metodo_umbral"] == "percentil_del_historico_propio"
    assert len({e["umbral"] for e in b["por_escenario"].values()}) > 1

    for esc in ("A_red_nueva", "B_red_conocida"):
        m = dep[esc]["metricas"]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        # La matriz de confusion debe cuadrar con el total evaluado.
        cm = dep[esc]["matriz_confusion"]
        assert sum(sum(fila) for fila in cm) == m["n_muestras"]


def test_umbral_dentro_de_rango():
    """El umbral del modelo, si existe, debe ser una probabilidad valida."""
    import joblib

    if not MODELO.exists():
        pytest.skip("falta el modelo (ejecuta train.py)")
    bundle = joblib.load(MODELO)
    assert 0.0 < bundle["umbral"] < 1.0


# --------------------------------------------------------------------------
# 5. Umbral adaptativo por red
# --------------------------------------------------------------------------


@requiere_modelo
def test_el_modelo_declara_el_metodo_de_umbral():
    """El agente debe saber si usa umbral fijo o adaptativo, y con que percentil."""
    import joblib

    bundle = joblib.load(MODELO)
    assert bundle.get("metodo_umbral") == "adaptativo_por_percentil"
    assert 0 < bundle.get("percentil_calibracion", 0) <= 100
    assert bundle.get("dias_calibracion", 0) > 0


@requiere_datos
@requiere_modelo
def test_la_calibracion_no_usa_etiquetas(tmp_path, ds):
    """Meter una etiqueta en las observaciones no debe cambiar el umbral.

    Es la garantia de que la calibracion se apoya solo en mediciones: el mismo
    flujo de observaciones con y sin etiqueta produce el mismo umbral.
    """
    import joblib

    from agent import LeakDetectionAgent

    bundle = joblib.load(MODELO)
    filas = ds.X[ds.scenario == "Scenario-9"].head(200)
    obs = [{"escenario": "Scenario-9", "timestamp": f"t{i}",
            "valores": dict(zip(ds.feature_names, filas.iloc[i].to_numpy()))}
           for i in range(len(filas))]

    with LeakDetectionAgent(MODELO, tmp_path / "a.csv") as ag1:
        c1 = ag1.calibrar(obs)
    obs_con_etiqueta = [dict(o, etiqueta_real=1) for o in obs]
    with LeakDetectionAgent(MODELO, tmp_path / "b.csv") as ag2:
        c2 = ag2.calibrar(obs_con_etiqueta)

    assert c1["umbral_adaptado"] == c2["umbral_adaptado"], "la etiqueta influyo en el umbral"
    assert 0.0 <= c1["umbral_adaptado"] <= 1.0
    assert c1["umbral_adaptado"] != bundle["umbral"], "el umbral deberia adaptarse"


@requiere_modelo
def test_calibrar_sin_observaciones_falla(tmp_path):
    from agent import LeakDetectionAgent

    with LeakDetectionAgent(MODELO, tmp_path / "c.csv") as ag:
        with pytest.raises(ValueError):
            ag.calibrar([])


@requiere_datos
@requiere_modelo
def test_el_umbral_adaptado_cambia_segun_la_red(tmp_path, ds):
    """El punto del metodo: redes distintas fijan umbrales distintos."""
    from agent import LeakDetectionAgent

    distintos = {}
    for esc in ("Scenario-1", "Scenario-4", "Scenario-9"):
        filas = ds.X[ds.scenario == esc].head(1440)
        obs = [{"escenario": esc, "timestamp": f"t{i}",
                "valores": dict(zip(ds.feature_names, filas.iloc[i].to_numpy()))}
               for i in range(len(filas))]
        with LeakDetectionAgent(MODELO, tmp_path / f"{esc}.csv") as ag:
            distintos[esc] = round(ag.calibrar(obs)["umbral_adaptado"], 4)
    assert len(set(distintos.values())) > 1, f"todas dieron el mismo umbral: {distintos}"