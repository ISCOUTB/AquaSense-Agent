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

Por que este orden
-----------------
  inspect  confirma el formato real del dataset antes de programar nada.
           Si el dataset cambio, el resto de pasos operarian sobre supuestos.
  train    produce el modelo y el umbral, y valida por Leave-One-Scenario-Out.
  evaluate resume esa validacion en el informe oficial de metricas.
  agent    ejecuta el ciclo percecion -> decision -> accion -> registro.

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
            extra += ["--escenario", a.escenario]
            if a.limite is not None:
                extra += ["--limite", str(a.limite)]
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
    for art in ("models/leak_classifier.joblib", "logs/train_report.json",
                "logs/evaluation_report.json", "logs/agent_events.csv",
                "logs/agent_run.json", "logs/data_inspection.json"):
        if Path(art).exists():
            print(f"    {art}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())