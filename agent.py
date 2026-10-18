r"""
agent.py - El agente de deteccion de fugas de AquaSense-Agent
==============================================================

Ciclo clasico de agente, con las cuatro responsabilidades separadas en metodos
identificables, como exige la especificacion de la Fase 2:

    perceive()   Recibe la medicion del entorno (lo que darian los sensores).
    decide()     Consulta al modelo de aprendizaje automatico y decide.
    act()        Ejecuta una accion: lanzar alerta o registrar operacion normal.
    log_event()  Persiste el evento para poder reconstruir que ocurrio.

Que decide el agente y que decide el modelo
-------------------------------------------
El razonamiento NO esta en este archivo. El modelo (RandomForestClassifier
entrenado en train.py) es el que decide; este archivo solo transporta la
percepcion, aplica el umbral aprendido y ejecuta la accion. No hay reglas
fijas de deteccion escritas a mano.

FUGA DE ETIQUETAS
-----------------
La etiqueta real NUNCA entra en perceive() ni en decide(). Se adjunta al
registro solo despues de actuar, y solo si el evaluador la proporciona, para
poder comparar. El agente no puede verla aunque se le pase como columna de
entrada: decide() selecciona explicitamente las variables del modelo.

Uso
---
    python agent.py --escenario Scenario-10 --limite 2000
    python agent.py --escenario Scenario-7
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/leak_classifier.joblib")
LOG_PATH = Path("logs/agent_events.csv")

ALERTA = "ALERTA_FUGA"
NORMAL = "OPERACION_NORMAL"

LOG_FIELDS = [
    "registro_utc", "escenario", "observacion_id", "timestamp_observacion",
    "n_variables_recibidas", "prob_fuga", "umbral", "decision", "accion",
    "etiqueta_real",
]


class LeakDetectionAgent:
    """Agente que clasifica cada observacion como fuga o no."""

    def __init__(self, model_path: Path = MODEL_PATH, log_path: Path = LOG_PATH):
        if not Path(model_path).exists():
            raise SystemExit(
                f"No existe el modelo {model_path}.\n"
                "Entrenalo primero:  python train.py")
        self.bundle = joblib.load(model_path)
        self.modelo = self.bundle["model"]
        self.variables = self.bundle["feature_names"]
        self.umbral = self.bundle["umbral"]
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._log, fieldnames=LOG_FIELDS)
        self._writer.writeheader()
        self._log.flush()
        self.n_alertas = 0
        self.n_observaciones = 0

    # ---------------- PERCEPCION ----------------
    def perceive(self, observacion: dict) -> dict:
        """Recibe la medicion del entorno y la deja lista para el modelo.

        observacion: {timestamp, escenario, valores: {variable: valor}}
        Devuelve un vector con EXACTAMENTE las variables que el modelo
        espera, en el mismo orden. Aqui es donde se decide que se ve:
        cualquier clave que no este en self.variables se descarta.
        """
        faltantes = [v for v in self.variables if v not in observacion["valores"]]
        if faltantes:
            raise ValueError(
                f"Faltan {len(faltantes)} variables en la percepcion: {faltantes[:5]}...")
        # DataFrame con los nombres de columna: el modelo se entreno con un
        # DataFrame y sin los nombres sklearn avisa en cada prediccion.
        vector = pd.DataFrame([[observacion["valores"][v] for v in self.variables]],
                              columns=self.variables)
        return {
            "escenario": observacion["escenario"],
            "timestamp": observacion["timestamp"],
            "vector": vector,
            "n_variables": len(self.variables),
        }

    # ---------------- DECISION ----------------
    def decide(self, percepcion: dict) -> dict:
        """Consulta al modelo entrenado. Aqui ocurre el aprendizaje automatico."""
        prob = float(self.modelo.predict_proba(percepcion["vector"])[0, 1])
        return {
            "prob_fuga": round(prob, 6),
            "umbral": self.umbral,
            "decision": "FUGA" if prob >= self.umbral else "NO_FUGA",
        }

    # ---------------- ACCION ----------------
    def act(self, decision: dict) -> str:
        """Ejecuta la accion correspondiente. En un sistema real, aqui se
        enviaria el aviso al centro de control; aqui queda registrado."""
        return ALERTA if decision["decision"] == "FUGA" else NORMAL

    # ---------------- REGISTRO ----------------
    def log_event(self, contexto: dict, decision: dict, accion: str,
                  etiqueta_real: float | None = None) -> None:
        """Persiste el evento. La etiqueta real se escribe DESPUES de decidir."""
        self._writer.writerow({
            "registro_utc": datetime.now().isoformat(timespec="seconds"),
            "escenario": contexto["escenario"],
            "observacion_id": self.n_observaciones,
            "timestamp_observacion": contexto["timestamp"],
            "n_variables_recibidas": contexto["n_variables"],
            "prob_fuga": decision["prob_fuga"],
            "umbral": decision["umbral"],
            "decision": decision["decision"],
            "accion": accion,
            # Vacio si el evaluador no la fourni: el agente no la necesita.
            "etiqueta_real": "" if etiqueta_real is None else etiqueta_real,
        })
        self._log.flush()

    # ---------------- CICLO ----------------
    def run(self, observaciones, incluir_etiquetas: dict | None = None) -> dict:
        """Recorre las observaciones y ejecuta el ciclo completo en cada una.

        observaciones: iterable de dicts {escenario, timestamp, valores}
        incluir_etiquetas: {observacion_id: etiqueta} SOLO para evaluacion.
            Se usa despues de decidir; nunca entra en el modelo.
        """
        inicio = datetime.now()
        for obs in observaciones:
            percepcion = self.perceive(obs)
            decision = self.decide(percepcion)
            accion = self.act(decision)
            etiqueta = (incluir_etiquetas or {}).get(self.n_observaciones)
            self.log_event(percepcion, decision, accion, etiqueta)
            self.n_observaciones += 1
            if accion == ALERTA:
                self.n_alertas += 1

        return {
            "observaciones": self.n_observaciones,
            "alertas": self.n_alertas,
            "tasa_alerta": round(self.n_alertas / self.n_observaciones, 6)
            if self.n_observaciones else 0.0,
            "umbral": self.umbral,
            "n_variables": len(self.variables),
            "log": str(self.log_path.resolve()),
            "inicio": inicio.isoformat(timespec="seconds"),
            "fin": datetime.now().isoformat(timespec="seconds"),
        }

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()

    def __enter__(self) -> "LeakDetectionAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def observaciones_de_escenario(ds, escenario: str, limite: int | None = None):
    """Convierte un escenario del dataset en el flujo de observaciones."""
    mask = ds.scenario == escenario
    X = ds.X[mask]
    ts = ds.timestamp[mask]
    total = len(X) if limite is None else min(limite, len(X))
    for i in range(total):
        yield {
            "escenario": escenario,
            "timestamp": str(ts[i]),
            "valores": dict(zip(ds.feature_names, X.iloc[i].to_numpy())),
        }


def main(argv: list[str] | None = None) -> int:
    from data_loader import load_dataset

    p = argparse.ArgumentParser(description="Ejecuta el agente de deteccion de fugas.")
    p.add_argument("--escenario", default="Scenario-10")
    p.add_argument("--limite", type=int, default=None,
                   help="Solo procesa las primeras N observaciones")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--log", type=Path, default=LOG_PATH)
    p.add_argument("--resumen-out", type=Path, default=Path("logs/agent_run.json"))
    a = p.parse_args(argv)

    ds = load_dataset(a.data_dir)
    if a.escenario not in ds.scenarios:
        raise SystemExit(f"{a.escenario} no existe. Opciones: {', '.join(ds.scenarios)}")

    mask = ds.scenario == a.escenario
    y = ds.y_int[mask]
    total = len(y) if a.limite is None else min(a.limite, len(y))
    etiquetas = {i: int(y[i]) for i in range(total)}

    print("=" * 78)
    print("AQUASENSE-AGENT  |  Ejecucion del agente")
    print("=" * 78)
    print(f"\n  Escenario   : {a.escenario}")
    print(f"  Observaciones a procesar: {total:,}")
    print(f"  Variables de entrada    : {len(ds.feature_names)}")
    print(f"  Umbral del modelo       : {joblib.load(a.model)['umbral']}")
    print(f"  Registro                : {a.log.resolve()}")

    with LeakDetectionAgent(a.model, a.log) as agente:
        resumen = agente.run(observaciones_de_escenario(ds, a.escenario, a.limite),
                             incluir_etiquetas=etiquetas)

    print(f"\n  Observaciones procesadas: {resumen['observaciones']:,}")
    print(f"  Alertas generadas       : {resumen['alertas']:,}")
    print(f"  Tasa de alerta          : {resumen['tasa_alerta']:.2%}")

    # Comparacion posterior, ya con el registro escrito.
    import csv as _csv
    aciertos = tp = fp = fn = total = 0
    with a.log.open(encoding="utf-8") as fh:
        for fila in _csv.DictReader(fh):
            real = fila["etiqueta_real"]
            if real == "":
                continue
            real = int(real)
            pred = 1 if fila["decision"] == "FUGA" else 0
            total += 1
            aciertos += pred == real
            tp += pred == 1 and real == 1
            fp += pred == 1 and real == 0
            fn += pred == 0 and real == 1
    resumen.update({
        "escenario": a.escenario, "etiqueta_real_proporcionada": True,
        "comparacion": {
            # El denominador de la exactitud son TODAS las filas, no solo las positivas.
            "accuracy": round(aciertos / total, 6) if total else None,
            "precision": round(tp / (tp + fp), 6) if tp + fp else None,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            "n_evaluadas": total, "tp": tp, "fp": fp, "fn": fn,
        },
        "advertencia": ("El modelo se entreno con los 10 escenarios, incluido este."
                        " Esta comparacion es EN MUESTRA y no es la cifra oficial;"
                        " la oficial es el Leave-One-Scenario-Out de train.py."),
    })
    print(f"\n  Comparacion con la etiqueta real (EN MUESTRA, no es la cifra oficial):")
    c = resumen["comparacion"]

    def fmt(v):
        return "  n/a" if v is None else f"{v:.4f}"

    print(f"    exactitud {fmt(c['accuracy'])}  precision {fmt(c['precision'])}  "
          f"recall {fmt(c['recall'])}")
    print(f"    TP={c['tp']:,}  FP={c['fp']:,}  FN={c['fn']:,}")
    if c["tp"] + c["fp"] == 0:
        print("    (n/a = no habia fugas reales en las observaciones procesadas;")
        print("     las metricas de la clase fuga no se pueden calcular)")

    a.resumen_out.parent.mkdir(parents=True, exist_ok=True)
    a.resumen_out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Resumen: {a.resumen_out.resolve()}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())