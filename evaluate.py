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
    """Presenta los resultados oficiales, separados por escenario de despliegue."""
    val = rep["validacion"]
    dep = val["escenarios_despliegue"]
    a = dep["A_red_nueva"]
    b = dep["B_red_conocida"]
    base = rep["linea_base_siempre_normal"]

    print(linea())
    print("EVALUACION OFICIAL - Leave-One-Scenario-Out")
    print(linea())
    print(f"\n  Modelo  : RandomForest, {rep['configuracion']['n_estimators']} arboles, "
          f"class_weight={rep['configuracion']['balance']}, beta={rep['configuracion']['beta']}, "
          f"semilla {rep['configuracion']['seed']}")
    print(f"  Datos   : {rep['datos']['n_filas']:,} filas, "
          f"{rep['datos']['n_variables']} variables, "
          f"{rep['datos']['pct_fuga'] * 100:.2f}% de fuga")
    print(f"  Base trivial (siempre 'normal'): {base:.4f}")

    print("\n" + "=" * 78)
    print("ESCENARIO A) RED NUEVA  (el operador no tiene datos de esta red)")
    print("=" * 78)
    print(f"\n  {a['descripcion']}\n")
    print(f"  Umbral global: {a['umbral']} (no se puede calibrar sin historico)\n")
    print("  1. CURVA DE COMPROMISO")
    print(f"  {'umbral':>8}{'exactitud':>11}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'alertas':>11}")
    print("  " + "-" * 58)
    for c in val["curva_umbral_global"]:
        marca = "  <-- punto A" if c["umbral"] == a["umbral"] else ""
        print(f"  {c['umbral']:>8.3f}{c['accuracy']:>11.4f}{c['precision']:>11.4f}"
              f"{c['recall']:>9.4f}{c['f1']:>8.4f}{c['n_alertas']:>11,}{marca}")

    ga = a["metricas"]
    print("\n  2. METRICAS")
    print(f"  {'':<22}{'exactitud':>11}{'precision':>11}{'recall':>10}{'F1':>9}")
    print("  " + "-" * 63)
    print(f"  {'agregado':<22}{ga['accuracy']:>11.4f}{ga['precision']:>11.4f}"
          f"{ga['recall']:>10.4f}{ga['f1']:>9.4f}")
    print(f"  {'media por fold':<22}{a['media_por_escenario']['accuracy']:>11.4f}"
          f"{a['media_por_escenario']['precision']:>11.4f}"
          f"{a['media_por_escenario']['recall']:>10.4f}"
          f"{a['media_por_escenario']['f1']:>9.4f}")
    mc = a["media_solo_con_fuga"]
    if mc:
        print(f"  {'solo con fuga':<22}{mc['accuracy']:>11.4f}{mc['precision']:>11.4f}"
              f"{mc['recall']:>10.4f}{mc['f1']:>9.4f}")
    print(f"\n  AUC global: {ga['auc']}")

    print("\n  3. MATRIZ DE CONFUSION")
    cm_a = np.array(a["matriz_confusion"])
    print(f"  {'':<18}{'previsto normal':>18}{'previsto fuga':>16}")
    print("  " + "-" * 52)
    print(f"  {'real normal':<18}{cm_a[0][0]:>18,}{cm_a[0][1]:>16,}")
    print(f"  {'real fuga':<18}{cm_a[1][0]:>18,}{cm_a[1][1]:>16,}")
    print(f"\n  TP={cm_a[1][1]:,}  FP={cm_a[0][1]:,}  "
          f"FN={cm_a[1][0]:,}  TN={cm_a[0][0]:,}")

    print("\n  4. POR RED")
    print(f"  {'Escenario':<12}{'exactitud':>10}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'AUC':>8}{'alertas':>10}")
    print("  " + "-" * 68)
    for esc, e in a["por_escenario"].items():
        auc = f"{e['auc']:.4f}" if e.get("auc") is not None else "   -   "
        control = " (control)" if e["n_reales_fuga"] == 0 else ""
        print(f"  {esc:<12}{e['accuracy']:>10.4f}{e['precision']:>11.4f}"
              f"{e['recall']:>9.4f}{e['f1']:>8.4f}{auc:>8}{e['n_alertas']:>10,}{control}")

    print("\n" + "=" * 78)
    print("ESCENARIO B) RED CONOCIDA  (el operador tiene historico de esta red)")
    print("=" * 78)
    print(f"\n  {b['descripcion']}\n")
    print(f"  Percentil {b['percentil']:g} ({b['percentil_modo']}) de los primeros "
          f"{b['dias_calibracion']} dias ({b['pasos_calibracion']:,} pasos)\n")
    print("  1. CURVA DE PERCENTILES")
    print(f"  {'percentil':>10}{'exactitud':>11}{'precision':>11}{'recall':>10}{'F1':>9}")
    print("  " + "-" * 51)
    for c in val["curva_percentiles"]:
        marca = "  <-- punto B" if c["percentil"] == b["percentil"] else ""
        print(f"  {c['percentil']:>10g}{c['accuracy']:>11.4f}{c['precision']:>11.4f}"
              f"{c['recall']:>10.4f}{c['f1']:>9.4f}{marca}")

    gb = b["metricas"]
    print("\n  2. METRICAS")
    print(f"  {'':<22}{'exactitud':>11}{'precision':>11}{'recall':>10}{'F1':>9}")
    print("  " + "-" * 63)
    print(f"  {'agregado':<22}{gb['accuracy']:>11.4f}{gb['precision']:>11.4f}"
          f"{gb['recall']:>10.4f}{gb['f1']:>9.4f}")

    print("\n  3. MATRIZ DE CONFUSION")
    cm_b = np.array(b["matriz_confusion"])
    print(f"  {'':<18}{'previsto normal':>18}{'previsto fuga':>16}")
    print("  " + "-" * 52)
    print(f"  {'real normal':<18}{cm_b[0][0]:>18,}{cm_b[0][1]:>16,}")
    print(f"  {'real fuga':<18}{cm_b[1][0]:>18,}{cm_b[1][1]:>16,}")
    print(f"\n  TP={cm_b[1][1]:,}  FP={cm_b[0][1]:,}  "
          f"FN={cm_b[1][0]:,}  TN={cm_b[0][0]:,}")

    print("\n  4. POR RED")
    print(f"  {'Escenario':<12}{'umbral':>9}{'exactitud':>11}{'precision':>11}"
          f"{'recall':>9}{'F1':>8}")
    print("  " + "-" * 60)
    for esc, e in b["por_escenario"].items():
        control = " (control)" if e["n_reales_fuga"] == 0 else ""
        print(f"  {esc:<12}{e['umbral']:>9.4f}{e['accuracy']:>11.4f}{e['precision']:>11.4f}"
              f"{e['recall']:>9.4f}{e['f1']:>8.4f}{control}")
    umbrales = [e["umbral"] for e in b["por_escenario"].values()]
    print(f"\n  Los umbrales por red van de {min(umbrales):.3f} a {max(umbrales):.3f}.")
    print("  Un unico valor global no cubre ese rango: ese es el motivo del metodo.")

    print("\n" + "=" * 78)
    print("COMPARACION Y VEREDICTO")
    print("=" * 78)
    print(f"\n  {'escenario':<34}{'exactitud':>11}{'precision':>11}{'recall':>10}{'F1':>9}")
    print("  " + "-" * 75)
    print(f"  {'A) red nueva':<34}{ga['accuracy']:>11.4f}{ga['precision']:>11.4f}"
          f"{ga['recall']:>10.4f}{ga['f1']:>9.4f}")
    print(f"  {'B) red conocida':<34}{gb['accuracy']:>11.4f}{gb['precision']:>11.4f}"
          f"{gb['recall']:>10.4f}{gb['f1']:>9.4f}")
    print(f"  {'diferencia':<34}{gb['accuracy'] - ga['accuracy']:>+11.4f}"
          f"{gb['precision'] - ga['precision']:>+11.4f}"
          f"{gb['recall'] - ga['recall']:>+10.4f}{gb['f1'] - ga['f1']:>+9.4f}")
    print(f"\n  Conocer la red aporta {100 * (gb['f1'] - ga['f1']) / ga['f1']:+.1f} % de F1.")

    print(f"\n  Meta de exactitud {TARGET_ACCURACY:.0%}:")
    print(f"    A) red nueva   : {'CUMPLIDA' if a['meta_cumplida'] else 'NO CUMPLIDA'} "
          f"({ga['accuracy']:.2%}, base {base:.2%}, "
          f"{(ga['accuracy'] - base) * 100:+.2f} puntos)")
    print(f"    B) red conocida : {'CUMPLIDA' if b['meta_cumplida'] else 'NO CUMPLIDA'} "
          f"({gb['accuracy']:.2%}, base {base:.2%}, "
          f"{(gb['accuracy'] - base) * 100:+.2f} puntos)")
    print(f"\n  F1 de la clase fuga: A={ga['f1']:.4f}  B={gb['f1']:.4f} "
          f"(minimo exigido {f1_min:.2f}: {'superado' if gb['f1'] >= f1_min else 'NO superado'})")
    if ga["accuracy"] < base:
        print("\n  En el escenario A la meta es INFERIOR a la linea base trivial:")
        print("  se puede 'cumplir' sin detectar ninguna fuga. Por eso el F1 importa mas.")
    print(f"\n  Nota declarada: {val['nota_optimismo']}")

    return {
        "escenarios_despliegue": {
            "A_red_nueva": {"umbral": a["umbral"], "metricas": ga,
                            "por_escenario": a["por_escenario"]},
            "B_red_conocida": {"percentil": b["percentil"], "metricas": gb,
                               "por_escenario": b["por_escenario"]},
        },
        "base_siempre_normal": base,
        "meta_exactitud": TARGET_ACCURACY,
        "meta_cumplida_A": a["meta_cumplida"],
        "meta_cumplida_B": b["meta_cumplida"],
        "f1_minimo": f1_min, "f1_alcanzado": bool(gb["f1"] >= f1_min),
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