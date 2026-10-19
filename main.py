r"""
main.py - Orquestador del proyecto AquaSense-Agent
===================================================

Ejecuta la cadena completa en el orden correcto y delega en cada modulo:

    inspect  ->  train  ->  evaluate  ->  agent

Cada paso se puede lanzar solo con --solo, para no repetir el entrenamiento
cada vez que se prueba el agente.

    python main.py                    # pipeline completo
    python main.py --solo agente      # solo ejecuta el agente
    python main.py --escenario Scenario-7
    python main.py --solo agent --todos-escenarios

Por que este orden
-----------------
  inspect  confirma el formato real del dataset antes de programar nada.
           Si el dataset cambio, el resto de pasos operarian sobre supuestos.
  train    produce el modelo y el umbral, y valida por Leave-One-Scenario-Out.
           Compara el metodo de umbral fijo con el adaptativo por red.
  evaluate resume esa validacion en el informe oficial de metricas.
  agent    ejecuta el ciclo percecion -> decision -> accion -> registro,
           calibrando antes su umbral con el historico limpio de esa red.

El paso de train es el lento (ajusta un umbral por fold). Los demas tardan
pocos segundos, salvo agent si se procesa un ano entero.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PASOS = {
    "inspect": "inspect_data",
    "train": "train",
    "evaluate": "evaluate",
    "agent": "agent",
}


def ejecutar(nombre: str, argv_extra: list[str]) -> float:
    """Delega en el main() del modulo correspondiente y mide cuanto tarda."""
    import importlib

    modulo = importlib.import_module(PASOS[nombre])
    print("\n" + "#" * 78)
    print(f"# PASO: {nombre.upper()}  (modulo {PASOS[nombre]}.py)")
    print("#" * 78 + "\n")
    t0 = time.perf_counter()
    codigo = modulo.main(argv_extra)
    dt = time.perf_counter() - t0
    if codigo != 0:
        raise SystemExit(f"El paso '{nombre}' termino con codigo {codigo}.")
    print(f"\n  [paso {nombre} completado en {dt:.1f} s]")
    return dt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AquaSense-Agent: pipeline completo.")
    p.add_argument("--solo", choices=list(PASOS), help="Ejecuta un unico paso")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--escenario", default="Scenario-10",
                   help="Escenario que procesa el agente")
    p.add_argument("--todos-escenarios", action="store_true",
                   help="Ejecuta el agente en los 10 escenarios")
    p.add_argument("--limite", type=int, default=None,
                   help="Limita el numero de observaciones del agente")
    p.add_argument("--n-estimators", type=int, default=None,
                   help="Arboles del RandomForest (por defecto, el de train.py)")
    a, resto = p.parse_known_args(argv)

    comunes = ["--data-dir", str(a.data_dir)]
    pasos = [a.solo] if a.solo else list(PASOS)
    tiempos: dict[str, float] = {}

    for paso in pasos:
        extra = list(comunes)
        if paso == "agent":
            escenarios = (
                [f"Scenario-{number}" for number in range(1, 11)]
                if a.todos_escenarios else [a.escenario]
            )
            for escenario in escenarios:
                args_agente = extra + ["--escenario", escenario]
                if a.limite is not None:
                    args_agente += ["--limite", str(a.limite)]
                if a.todos_escenarios:
                    args_agente += [
                        "--log", f"logs/agent_events_{escenario}.csv",
                        "--resumen-out", f"logs/agent_run_{escenario}.json",
                    ]
                key = f"agent_{escenario}" if a.todos_escenarios else paso
                tiempos[key] = ejecutar(paso, args_agente + resto)
            continue
        if paso == "train" and a.n_estimators is not None:
            extra += ["--n-estimators", str(a.n_estimators)]
        tiempos[paso] = ejecutar(paso, extra + resto)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETO")
    print("=" * 78)
    for paso, dt in tiempos.items():
        print(f"  {paso:<10} {dt:>8.1f} s")
    print(f"  {'TOTAL':<10} {sum(tiempos.values()):>8.1f} s")
    print("\n  Artefactos generados:")
    artefactos = [
        "models/leak_classifier.joblib", "logs/train_report.json",
        "logs/evaluation_report.json", "logs/agent_events.csv",
        "logs/agent_run.json", "logs/data_inspection.json",
    ]
    if a.todos_escenarios:
        artefactos.extend(
            f"logs/{prefix}_{scenario}.{suffix}"
            for scenario in (f"Scenario-{number}" for number in range(1, 11))
            for prefix, suffix in (
                ("agent_events", "csv"),
                ("agent_run", "json"),
            )
        )
    for art in artefactos:
        if Path(art).exists():
            print(f"    {art}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())