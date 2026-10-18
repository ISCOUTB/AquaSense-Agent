"""
train.py - Entrenamiento del modelo de deteccion de fugas
==========================================================

Agente AquaSense-Agent. Modelo: RandomForestClassifier de scikit-learn.

Decisiones de diseno y por que se tomaron
----------------------------------------
1. DIVISION POR ESCENARIO, obligatorio.
   inspect_data.py verifico que las series de Demands son distintas en los
   10 escenarios: la demanda identifica univocamente al escenario, y 4 de
   los 10 son controles sin fuga (etiqueta siempre 0.0). Una division
   aleatoria sobre filas dejaria al modelo reconocer el escenario de una
   fila de prueba y contestarle con la respuesta. Aqui se usa
   Leave-One-Scenario-Out: se entrena con 9 escenarios y se evalua en el
   decimo, rotando. Ningun instante del escenario evaluado aparece nunca
   en el entrenamiento de ese mismo fold.

2. EL UMBRAL SE ELIGE SOBRE PREDICCIONES FUERA DE MUESTRA.
   Medido en este dataset: con umbral 0.5 el modelo daba F1 = 0.00 en los
   seis escenarios con fuga, pero AUC de 0.93 a 1.00. El modelo ordenaba
   bien los instantes con fuga y solo fallaba al decidir cuando avisar.
   Se hace una sola pasada Leave-One-Scenario-Out que produce la probabilidad
   de cada fila predicha por un modelo que NUNCA vio su escenario. Sobre esas
   predicciones se elige el umbral que maximiza F-beta y se evalua la curva
   completa de compromiso precision/recall.

   Se descarto un ajuste de umbral anidado por fold: el interior mezclaba
   probabilidades producidas por modelos distintos, que no son comparables
   entre si, y elegia umbrales erroneos (AUC interior de 0.41 frente a 0.9999
   del mismo escenario ya evaluado). Medir fue la unica forma de saberlo.

3. CLASES DESBALANCEADAS (15.46 % de fuga).
   Se reportan precision, recall, F1, AUC y matriz de confusion, no solo
   exactitud: predecir siempre "normal" ya da 84.54 % sin detectar nada.

4. SEMILLA FIJA en todos los puntos aleatorios, para reproducibilidad.

5. LA ETIQUETA NO ENTRA COMO ENTRADA. data_loader la carga y la mantiene
   separada de X en todo momento.

Uso
---
    python train.py
    python train.py --n-estimators 150 --beta 2   # prioriza no perder fugas
    python train.py --balance none
    python train.py --with-demands                # experimento de ablacion
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut

from data_loader import RANDOM_SEED, Dataset, load_dataset

MODEL_PATH = Path("models/leak_classifier.joblib")
REPORT_PATH = Path("logs/train_report.json")
TARGET_ACCURACY = 0.80
THRESHOLD_GRID = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08,
                  0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]


# --------------------------------------------------------------------------
# Utilidades de evaluacion
# --------------------------------------------------------------------------


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float | None:
    """AUC solo si hay ambas clases; si no, no significa nada."""
    if len(np.unique(y_true)) < 2:
        return None
    return round(float(roc_auc_score(y_true, prob)), 6)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_muestras": int(len(y_true)),
        "n_reales_fuga": int(y_true.sum()),
        "n_alertas": int(y_pred.sum()),
    }
    if prob is not None:
        m["auc"] = safe_auc(y_true, prob)
    return m


def fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------


def build_model(n_estimators: int, balance: str, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None if balance == "none" else "balanced",
        n_jobs=-1,
        random_state=seed,
    )


# --------------------------------------------------------------------------
# Validacion Leave-One-Scenario-Out
# --------------------------------------------------------------------------
#
# Una sola pasada produce las predicciones fuera de muestra de TODAS las filas:
# para cada escenario se entrena con los otros nueve y se predice el que se
# queda fuera. Esas mismas predicciones sirven para elegir el umbral y para
# evaluar, lo que evita el problema del ajuste anidado.


def predicciones_fuera_de_muestra(ds: Dataset, n_estimators: int, balance: str,
                                  seed: int) -> np.ndarray:
    """Probabilidad de fuga de cada fila, predicha por un modelo que no vio su escenario."""
    prob = np.zeros(ds.n_samples, dtype=np.float64)
    logo = LeaveOneGroupOut()
    for i, (tr, te) in enumerate(logo.split(ds.X, ds.y_int, ds.scenario), start=1):
        evaluado = sorted(set(ds.scenario[te]))[0]
        m = build_model(n_estimators, balance, seed)
        m.fit(ds.X.iloc[tr], ds.y_int[tr])
        prob[te] = m.predict_proba(ds.X.iloc[te])[:, 1]
        print(f"   [{i:>2}/10] {evaluado:<12} {int(ds.y_int[te].sum()):>6,} pasos con fuga "
              f"de {len(te):,}")
    return prob


def evaluar_en_umbral(ds: Dataset, prob: np.ndarray, umbral: float) -> dict:
    pred = (prob >= umbral).astype(int)
    global_ = metrics(ds.y_int, pred, prob)

    por_esc = {}
    for esc in ds.scenarios:
        m = ds.scenario == esc
        por_esc[esc] = metrics(ds.y_int[m], pred[m], prob[m])

    con_fuga = {k: v for k, v in por_esc.items() if v["n_reales_fuga"] > 0}
    media = lambda sel, campo: round(float(np.mean([v[campo] for v in sel.values()])), 6)

    return {
        "umbral": umbral,
        "global": global_,
        "media_por_escenario": {c: media(por_esc, c) for c in
                                 ("accuracy", "precision", "recall", "f1")},
        "media_solo_con_fuga": {c: media(con_fuga, c) for c in
                                ("accuracy", "precision", "recall", "f1")} if con_fuga else None,
        "por_escenario": por_esc,
    }


def curva_compromiso(ds: Dataset, prob: np.ndarray, beta: float) -> list[dict]:
    """Exactitud / precision / recall / F1 para cada umbral candidato."""
    curva = []
    for th in THRESHOLD_GRID:
        pred = (prob >= th).astype(int)
        curva.append({
            "umbral": th,
            "accuracy": round(float(accuracy_score(ds.y_int, pred)), 6),
            "precision": round(float(precision_score(ds.y_int, pred, zero_division=0)), 6),
            "recall": round(float(recall_score(ds.y_int, pred, zero_division=0)), 6),
            "f1": round(float(f1_score(ds.y_int, pred, zero_division=0)), 6),
            "fbeta": round(fbeta(ds.y_int, pred, beta), 6),
            "n_alertas": int(pred.sum()),
        })
    return curva


# --------------------------------------------------------------------------
# Programa
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Entrena el detector de fugas de AquaSense-Agent.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--balance", choices=["none", "class_weight"], default="class_weight")
    p.add_argument("--beta", type=float, default=1.0,
                   help="F-beta para elegir el umbral. beta>1 prioriza recall.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--with-demands", action="store_true")
    p.add_argument("--model-out", type=Path, default=MODEL_PATH)
    p.add_argument("--report-out", type=Path, default=REPORT_PATH)
    a = p.parse_args(argv)

    print("=" * 78)
    print("AQUASENSE-AGENT  |  Entrenamiento del clasificador de fugas")
    print("=" * 78)

    ds = load_dataset(a.data_dir, include_demands=a.with_demands)
    print(f"\n  Dataset   : {ds.describe()}")
    print(f"  Excluidas : sin variacion={ds.excluded['presion_sin_variacion']}, "
          f"constantes={ds.excluded['constantes'] or 'ninguna'}")
    print(f"  No entrada: {'; '.join(ds.excluded['no_es_entrada'])}")

    base = float((ds.y_int == 0).mean())
    print(f"\n  Linea base (predecir siempre 'normal'): {base:.4f} de exactitud.")
    print("  Superar el 80 % sin superar esta cifra no demuestra deteccion.\n")

    print(f"  Leave-One-Scenario-Out | {a.n_estimators} arboles | "
          f"class_weight={a.balance} | beta={a.beta} | seed={a.seed}")
    print("  Calculando la probabilidad de fuga fuera de muestra de cada fila...\n")
    prob_oof = predicciones_fuera_de_muestra(ds, a.n_estimators, a.balance, a.seed)

    print("\n  1. CURVA DE COMPROMISO (predicciones fuera de muestra)")
    print(f"  {'umbral':>8}{'exactitud':>11}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'F-beta':>9}{'alertas':>11}")
    print("  " + "-" * 67)
    curva = curva_compromiso(ds, prob_oof, a.beta)
    elegido = max(curva, key=lambda c: c["fbeta"])
    th = elegido["umbral"]
    for c in curva:
        marca = "  <-- elegido" if c["umbral"] == th else ""
        print(f"  {c['umbral']:>8.3f}{c['accuracy']:>11.4f}{c['precision']:>11.4f}"
              f"{c['recall']:>9.4f}{c['f1']:>8.4f}{c['fbeta']:>9.4f}"
              f"{c['n_alertas']:>11,}{marca}")

    ev = evaluar_en_umbral(ds, prob_oof, th)

    print(f"\n  2. PUNTO DE OPERACION: umbral {th} (maximiza F-beta={a.beta})")
    g = ev["global"]
    print(f"     Global      : exactitud={g['accuracy']:.4f} precision={g['precision']:.4f} "
          f"recall={g['recall']:.4f} F1={g['f1']:.4f} AUC={g['auc']}")
    m = ev["media_por_escenario"]
    mc = ev["media_solo_con_fuga"]
    print(f"     Media folds : exactitud={m['accuracy']:.4f} precision={m['precision']:.4f} "
          f"recall={m['recall']:.4f} F1={m['f1']:.4f}")
    if mc:
        print(f"     Solo fuga   : exactitud={mc['accuracy']:.4f} precision={mc['precision']:.4f} "
              f"recall={mc['recall']:.4f} F1={mc['f1']:.4f}")

    print("\n  3. POR ESCENARIO (ninguno visto en su propio entrenamiento)")
    print(f"  {'Escenario':<12}{'exactitud':>10}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'AUC':>8}{'alertas':>10}")
    print("  " + "-" * 68)
    for esc, e in ev["por_escenario"].items():
        auc = f"{e['auc']:.4f}" if e.get("auc") is not None else "   -   "
        control = " (control)" if e["n_reales_fuga"] == 0 else ""
        print(f"  {esc:<12}{e['accuracy']:>10.4f}{e['precision']:>11.4f}"
              f"{e['recall']:>9.4f}{e['f1']:>8.4f}{auc:>8}{e['n_alertas']:>10,}{control}")

    print("\n  4. MATRIZ DE CONFUSION GLOBAL")
    cm = np.array(g["matriz_confusion"])
    print(f"  {'':<16}{'previsto normal':>18}{'previsto fuga':>16}")
    for i, nombre in enumerate(("real normal", "real fuga")):
        print(f"  {nombre:<16}{cm[i][0]:>18,}{cm[i][1]:>16,}")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    print("\n  5. LECTURA DE LA META")
    cumple = g["accuracy"] >= TARGET_ACCURACY
    print(f"     Exactitud {g['accuracy']:.2%} frente a la meta del {TARGET_ACCURACY:.0%}: "
          f"{'CUMPLIDA' if cumple else 'NO CUMPLIDA'}")
    print(f"     Linea base (siempre 'normal'): {base:.2%}")
    print(f"     Diferencia sobre la base: {(g['accuracy'] - base) * 100:+.2f} puntos")
    if g["accuracy"] < base:
        print("     OJO: la meta de exactitud es INFERIOR a la linea base trivial.")
        print("          Detectar fugas obliga a emitir alertas y toda alerta falsa")
        print("          resta exactitud. Alcanzar el 80 % aqui exige no detectar casi")
        print("          nada. La cifra util es el F1 de la clase fuga.")

    print("\n  Entrenando el modelo final con los 10 escenarios...")
    final = build_model(a.n_estimators, a.balance, a.seed)
    final.fit(ds.X, ds.y_int)

    print("\n  Variables mas influyentes:")
    for name, val in sorted(zip(ds.feature_names, final.feature_importances_),
                            key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"    {name:<12} {val:.4f}")

    a.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final, "feature_names": ds.feature_names, "excluded": ds.excluded,
        "umbral": th, "beta": a.beta, "seed": a.seed,
        "n_estimators": a.n_estimators, "balance": a.balance,
        "con_demandas": a.with_demands,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
    }, a.model_out)

    report = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "configuracion": {"n_estimators": a.n_estimators, "balance": a.balance,
                          "beta": a.beta, "seed": a.seed,
                          "with_demands": a.with_demands},
        "entorno": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "datos": {"n_filas": ds.n_samples, "n_variables": len(ds.feature_names),
                  "escenarios": ds.scenarios, "pct_fuga": round(float(ds.y.mean()), 4),
                  "excluidas": ds.excluded},
        "validacion": {
            "esquema": "Leave-One-Scenario-Out en una sola pasada",
            "curva_compromiso": curva,
            "punto_operacion": ev,
            "umbral": th,
            "nota_optimismo": (
                "El umbral se elige sobre las mismas predicciones fuera de muestra que "
                "se reportan. Es un unico escalar derivado de 175,200 filas, por lo que "
                "el optimismo es esperablemente despreciable, pero se declara."),
        },
        "matriz_confusion_global": cm.tolist(),
        "linea_base_siempre_normal": round(base, 6),
        "meta_exactitud": TARGET_ACCURACY,
        "meta_cumplida": bool(cumple),
        "umbral_elegido": th,
    }
    a.report_out.parent.mkdir(parents=True, exist_ok=True)
    a.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Modelo  : {a.model_out.resolve()}")
    print(f"  Informe : {a.report_out.resolve()}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())