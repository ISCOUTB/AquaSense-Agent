"""
evaluate.py - Evaluacion reproducible del detector de fugas
===========================================================

Produce el informe oficial de metricas de AquaSense-Agent.

De donde salen los numeros
-------------------------
Este script NO se inventa metricas. Lee el Leave-One-Scenario-Out que dejo
train.py en logs/train_report.json y lo presenta. Esa validacion es la
honesta: cada fold se evalua sobre un escenario que el modelo no vio nunca.

Contraste con --escenario, que evalua el modelo YA DESPLEGADO (entrenado con
los 10 escenarios) sobre un escenario concreto. Ahi el modelo ya vio esos
datos, asi que el resultado es EN MUESTRA y sale optimisticamente sesgado.
Se marca como tal para que no se confunda con la cifra oficial.

Uso
---
    python evaluate.py
    python evaluate.py --escenario Scenario-10      # contraste en muestra
    python evaluate.py --min-f1 0.50                # criterio de exito
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from data_loader import load_dataset

TRAIN_REPORT = Path("logs/train_report.json")
EVAL_REPORT = Path("logs/evaluation_report.json")
MODEL_PATH = Path("models/leak_classifier.joblib")
TARGET_ACCURACY = 0.80


def linea(char: str = "=", n: int = 78) -> str:
    return char * n


def mostrar_validacion(rep: dict, f1_min: float) -> dict:
    """Presenta los resultados oficiales: Leave-One-Scenario-Out."""
    val = rep["validacion"]
    op = val["punto_operacion"]
    base = rep["linea_base_siempre_normal"]
    g = op["global"]
    m = op["media_por_escenario"]
    mc = op["media_solo_con_fuga"]
    th = val["umbral"]

    print(linea())
    print("EVALUACION OFICIAL - Leave-One-Scenario-Out")
    print(linea())
    print(f"\n  Esquema    : {val['esquema']}")
    print(f"  Semilla    : {rep['configuracion']['seed']}")
    print(f"  Modelo     : RandomForest, {rep['configuracion']['n_estimators']} arboles, "
          f"class_weight={rep['configuracion']['balance']}, beta={rep['configuracion']['beta']}")
    print(f"  Datos      : {rep['datos']['n_filas']:,} filas, "
          f"{rep['datos']['n_variables']} variables, "
          f"{rep['datos']['pct_fuga'] * 100:.2f}% de fuga")
    print(f"  Umbral     : {th}")

    print("\n  1. CURVA DE COMPROMISO PRECISION / RECALL")
    print(f"  {'umbral':>8}{'exactitud':>11}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'alertas':>11}")
    print("  " + "-" * 58)
    for c in val["curva_compromiso"]:
        marca = "  <-- punto de operacion" if c["umbral"] == th else ""
        print(f"  {c['umbral']:>8.3f}{c['accuracy']:>11.4f}{c['precision']:>11.4f}"
              f"{c['recall']:>9.4f}{c['f1']:>8.4f}{c['n_alertas']:>11,}{marca}")

    print("\n  2. METRICAS EN EL PUNTO DE OPERACION")
    print(f"  {'':<22}{'exactitud':>11}{'precision':>11}{'recall':>10}{'F1':>9}")
    print("  " + "-" * 63)
    print(f"  {'agregado':<22}{g['accuracy']:>11.4f}{g['precision']:>11.4f}"
          f"{g['recall']:>10.4f}{g['f1']:>9.4f}")
    print(f"  {'media por fold':<22}{m['accuracy']:>11.4f}{m['precision']:>11.4f}"
          f"{m['recall']:>10.4f}{m['f1']:>9.4f}")
    if mc:
        print(f"  {'solo con fuga':<22}{mc['accuracy']:>11.4f}{mc['precision']:>11.4f}"
              f"{mc['recall']:>10.4f}{mc['f1']:>9.4f}")
    print(f"  {'base: siempre normal':<22}{base:>11.4f}{'n/a':>11}{'0.0000':>10}{'0.0000':>9}")
    print(f"\n  AUC global: {g['auc']}")

    print("\n  3. MATRIZ DE CONFUSION (agregada)")
    cm = np.array(rep["matriz_confusion_global"])
    print(f"  {'':<18}{'previsto normal':>18}{'previsto fuga':>16}")
    print("  " + "-" * 52)
    print(f"  {'real normal':<18}{cm[0][0]:>18,}{cm[0][1]:>16,}")
    print(f"  {'real fuga':<18}{cm[1][0]:>18,}{cm[1][1]:>16,}")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"\n  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print("  (TP fuga detectada | FP falsa alarma | FN fuga perdida | TN normal correcto)")

    print("\n  4. RESULTADO POR ESCENARIO EVALUADO")
    print(f"  {'Escenario':<12}{'exactitud':>10}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'AUC':>8}{'alertas':>10}")
    print("  " + "-" * 68)
    for esc, e in op["por_escenario"].items():
        auc = f"{e['auc']:.4f}" if e.get("auc") is not None else "   -   "
        control = " (control)" if e["n_reales_fuga"] == 0 else ""
        print(f"  {esc:<12}{e['accuracy']:>10.4f}{e['precision']:>11.4f}"
              f"{e['recall']:>9.4f}{e['f1']:>8.4f}{auc:>8}{e['n_alertas']:>10,}{control}")
    print("\n  Los escenarios marcados (control) no tienen fugas: su exactitud no es")
    print("  un acierto del modelo, es la consecuencia de no haber avisado nunca.")

    print("\n  5. VEREDICTO")
    cumple_acc = g["accuracy"] >= TARGET_ACCURACY
    print(f"    Exactitud {g['accuracy']:.2%} frente a la meta del {TARGET_ACCURACY:.0%}: "
          f"{'CUMPLIDA' if cumple_acc else 'NO CUMPLIDA'}")
    print(f"    Linea base trivial ({base:.2%}). Diferencia: {(g['accuracy'] - base) * 100:+.2f} puntos")
    print(f"    F1 de la clase fuga: {g['f1']:.4f} "
          f"({'supera' if g['f1'] >= f1_min else 'NO supera'} el minimo de {f1_min:.2f})")
    if not cumple_acc and g["accuracy"] < base:
        print("\n    La meta de exactitud del 80 % es INFERIOR a la linea base trivial.")
        print("    No es un objetivo util: se supera sin detectar ninguna fuga. Lo")
        print("    que mide capacidad de deteccion es el F1 de la clase fuga.")

    print(f"\n  Nota declarada: {val['nota_optimismo']}")

    return {
        "umbral": th,
        "agregado": g,
        "media_por_fold": m,
        "solo_con_fuga": mc,
        "curva_compromiso": val["curva_compromiso"],
        "matriz_confusion": cm.tolist(),
        "base_siempre_normal": base,
        "meta_exactitud": TARGET_ACCURACY, "meta_cumplida": bool(cumple_acc),
        "f1_minimo": f1_min, "f1_alcanzado": bool(g["f1"] >= f1_min),
    }


def mostrar_escenario(ds, esc: str) -> dict:
    """Contraste EN MUESTRA del modelo desplegado sobre un escenario concreto."""
    if esc not in ds.scenarios:
        raise SystemExit(f"{esc} no existe. Opciones: {', '.join(ds.scenarios)}")

    bundle = joblib.load(MODEL_PATH)
    modelo, umbral = bundle["model"], bundle["umbral"]
    mask = ds.scenario == esc
    X, y = ds.X[mask], ds.y_int[mask]
    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= umbral).astype(int)

    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    res = {
        "escenario": esc, "EN_MUESTRA": True,
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 6),
        "auc": round(float(roc_auc_score(y, prob)), 6) if len(np.unique(y)) > 1 else None,
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "n_muestras": int(len(y)), "n_reales_fuga": int(y.sum()),
        "n_alertas": int(pred.sum()), "umbral": umbral,
    }
    print(linea())
    print(f"CONTRASTE - {esc}  (EN MUESTRA)")
    print(linea())
    print("\n  ATENCION: el modelo desplegado se entreno con los 10 escenarios,")
    print("  incluido este. Estas cifras NO sirven como evaluacion degeneralizacion.")
    print("  La cifra oficial es la de la seccion 1 de este informe.\n")
    print(f"  {'':<20}{'valor':>10}")
    print("  " + "-" * 30)
    for k in ("accuracy", "precision", "recall", "f1", "auc", "n_alertas", "n_reales_fuga"):
        v = res[k]
        print(f"  {k:<20}{v if v is None else (f'{v:,.4f}' if isinstance(v, float) else f'{v:,}'):>10}")
    cm = np.array(res["matriz_confusion"])
    print(f"\n  matriz: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluacion reproducible de AquaSense-Agent.")
    p.add_argument("--escenario", help="Contraste en muestra del modelo desplegado")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--min-f1", type=float, default=0.50)
    p.add_argument("--train-report", type=Path, default=TRAIN_REPORT)
    p.add_argument("--out", type=Path, default=EVAL_REPORT)
    a = p.parse_args(argv)

    if not a.train_report.exists():
        raise SystemExit(
            f"No existe {a.train_report}.\n"
            "Ejecuta primero:  python train.py\n"
            "Alli se calcula el Leave-One-Scenario-Out que despues se resume aqui.")

    rep = json.loads(a.train_report.read_text(encoding="utf-8"))
    resumen = mostrar_validacion(rep, a.min_f1)

    salida = {"generado": rep["generado"], "oficial": resumen}
    if a.escenario:
        salida["contraste_en_muestra"] = mostrar_escenario(load_dataset(a.data_dir), a.escenario)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Informe de evaluacion: {a.out.resolve()}")
    print(linea())
    return 0


if __name__ == "__main__":
    sys.exit(main())