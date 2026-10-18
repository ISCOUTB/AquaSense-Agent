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
    """Agente que clasifica cada observacion como fuga o no.

    Umbral adaptativo: en vez de usar una constante valida para todas las
    redes, el agente fija el SUYO a partir de un periodo que el operador
    declara sin fugas. Es necesario porque cada red es una realizacion
    hidraulica distinta y la escala de las probabilidades no es comparable
    entre redes: medido en este dataset, el umbral optimo va de 0.010 a
    0.400 segun el escenario.
    """

    def __init__(self, model_path: Path = MODEL_PATH, log_path: Path = LOG_PATH,
                 percentil: float | None = None):
        if not Path(model_path).exists():
            raise SystemExit(
                f"No existe el modelo {model_path}.\n"
                "Entrenalo primero:  python train.py")
        self.bundle = joblib.load(model_path)
        self.modelo = self.bundle["model"]
        self.variables = self.bundle["feature_names"]
        # Variables temporales: el agente necesita historial para calcularlas.
        self.temporal = self.bundle.get("temporal", [])
        self.ventana_max = max((w for _, _, w in self.temporal), default=0)
        self._base = [v for v in self.variables
                      if not any(v.endswith(f"_{t}") for t, _, _ in self.temporal)]
        self._buffer: list[np.ndarray] = []
        self.percentil = percentil if percentil is not None else \
            self.bundle.get("percentil_calibracion", 90.0)
        # Hasta calibrar, se usa el umbral global como valor provisional.
        self.umbral = self.bundle["umbral"]
        self.calibrado = False
        self.calibracion: dict = {}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._log, fieldnames=LOG_FIELDS)
        self._writer.writeheader()
        self._log.flush()
        self.n_alertas = 0
        self.n_observaciones = 0

    # ---------------- PERCEPCION ----------------
    def _probabilidad(self, observacion: dict) -> float:
        """Puntua una observacion sin registrarla. Usado por calibrar y decidir."""
        percepcion = self.perceive(observacion)
        return self.decide(percepcion)["prob_fuga"]

    # ---------------- CALIBRACION ----------------
    def calibrar(self, observaciones_limpias) -> dict:
        """Fija el umbral con un periodo que el operador declara SIN FUGAS.

        Es el equivalente a operar la red un mes y ver que no ocurre nada:
        se recogen las puntuaciones del modelo y se toma el percentil
        indicado. A partir de ahi, superar ese percentil significa salirse
        del comportamiento habitual de ESA red concreta.

        No usa etiquetas en ningun momento: solo recibe mediciones.
        """
        puntuaciones = [self._probabilidad(obs) for obs in observaciones_limpias]
        if not puntuaciones:
            raise ValueError("La calibracion necesita al menos una observacion.")

        anterior = self.umbral
        self.umbral = float(np.percentile(puntuaciones, self.percentil))
        self.calibrado = True
        self.calibracion = {
            "periodil": self.percentil,
            "n_observaciones": len(puntuaciones),
            "puntuaciones": {
                "min": round(float(np.min(puntuaciones)), 6),
                "mediana": round(float(np.median(puntuaciones)), 6),
                "p90": round(float(np.percentile(puntuaciones, 90)), 6),
                "max": round(float(np.max(puntuaciones)), 6),
            },
            "umbral_global_previo": anterior,
            "umbral_adaptado": round(self.umbral, 6),
        }
        return self.calibracion

    # ---------------- PERCEPCION ----------------
    def perceive(self, observacion: dict) -> dict:
        """Recibe la medicion del entorno y la deja lista para el modelo.

        observacion: {timestamp, escenario, valores: {variable: valor}}
        Devuelve un vector con EXACTAMENTE las variables que el modelo
        espera, en el mismo orden. Aqui es donde se decide que se ve:
        cualquier clave que no este en self.variables se descarta.

        Si el modelo usa variables temporales, se calculan aqui a partir de
        un buffer de las ultimas observaciones. Mientras el buffer no este
        lleno la ventana es parcial: es la fase de calentamiento, y el
        agente lo indica con 'caliente=False' para que no se alerte todavia.
        """
        faltantes = [v for v in self._base if v not in observacion["valores"]]
        if faltantes:
            raise ValueError(
                f"Faltan {len(faltantes)} variables en la percepcion: {faltantes[:5]}...")

        vector_base = np.array([observacion["valores"][v] for v in self._base],
                               dtype=np.float64)
        self._buffer.append(vector_base)
        if len(self._buffer) > self.ventana_max:
            self._buffer.pop(0)

        fila = dict(zip(self._base, vector_base))
        for tag, metodo, w in self.temporal:
            reciente = np.array(self._buffer[-w:])
            valor = reciente.max(axis=0) if metodo == "max" else reciente.mean(axis=0)
            fila.update({f"{c}_{tag}": valor[i] for i, c in enumerate(self._base)})

        # DataFrame con los nombres de columna: el modelo se entreno con un
        # DataFrame y sin los nombres sklearn avisa en cada prediccion.
        vector = pd.DataFrame([[fila[v] for v in self.variables]],
                              columns=self.variables)
        return {
            "escenario": observacion["escenario"],
            "timestamp": observacion["timestamp"],
            "vector": vector,
            "n_variables": len(self.variables),
            "caliente": len(self._buffer) >= self.ventana_max,
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
    def act(self, decision: dict, caliente: bool = True) -> str:
        """Ejecuta la accion correspondiente. En un sistema real, aqui se
        enviaria el aviso al centro de control; aqui queda registrado.

        Durante el calentamiento (buffer de historial incompleto) no se alerta:
        las variables temporales estan calculadas sobre una ventana parcial y
        no son comparables con las de regime estable.
        """
        if not caliente:
            return "CALENTAMIENTO"
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
    def run(self, observaciones, incluir_etiquetas: dict | None = None,
            n_calibracion: int = 0) -> dict:
        """Recorre las observaciones y ejecuta el ciclo completo en cada una.

        observaciones: iterable de dicts {escenario, timestamp, valores}
        n_calibracion: numero de observaciones iniciales usadas para calibrar.
            Esas NO se registran como eventos de monitorizacion: son el
            periodo de referencia declarado sin fugas.
        incluir_etiquetas: {observacion_id: etiqueta} SOLO para evaluacion.
            Se usa despues de decidir; nunca entra en el modelo.
        """
        inicio = datetime.now()
        n_calentamiento = 0
        for i, obs in enumerate(observaciones):
            if i == n_calibracion and n_calibracion > 0 and not self.calibrado:
                # El agente ya vio el periodo limpio; fija su umbral ahora.
                self.calibrar(observaciones[:n_calibracion])

            percepcion = self.perceive(obs)
            decision = self.decide(percepcion)
            accion = self.act(decision, percepcion["caliente"])
            etiqueta = (incluir_etiquetas or {}).get(i)
            self.log_event(percepcion, decision, accion, etiqueta)
            self.n_observaciones += 1
            if accion == "CALENTAMIENTO":
                n_calentamiento += 1
            elif accion == ALERTA:
                self.n_alertas += 1

        return {
            "observaciones": self.n_observaciones,
            "observaciones_calibracion": n_calibracion,
            "observaciones_calentamiento": n_calentamiento,
            "ventana_maxima_pasos": self.ventana_max,
            "alertas": self.n_alertas,
            "tasa_alerta": round(self.n_alertas / self.n_observaciones, 6)
            if self.n_observaciones else 0.0,
            "metodo_umbral": "adaptativo_por_percentil" if self.calibrado else "global_fijo",
            "percentil": self.percentil,
            "umbral": self.umbral,
            "calibracion": self.calibracion,
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
    p.add_argument("--dias-calibracion", type=int, default=30,
                   help="Dias iniciales que se toman como periodo sin fugas (0 = sin calibrar)")
    p.add_argument("--percentil", type=float, default=None,
                   help="Percentil del historico limpio; por defecto, el del modelo")
    p.add_argument("--resumen-out", type=Path, default=Path("logs/agent_run.json"))
    a = p.parse_args(argv)

    ds = load_dataset(a.data_dir)
    if a.escenario not in ds.scenarios:
        raise SystemExit(f"{a.escenario} no existe. Opciones: {', '.join(ds.scenarios)}")

    mask = ds.scenario == a.escenario
    y = ds.y_int[mask]
    total = len(y) if a.limite is None else min(a.limite, len(y))
    n_cal = min(a.dias_calibracion * 48, total)
    etiquetas = {i: int(y[i]) for i in range(total)}
    observaciones = list(observaciones_de_escenario(ds, a.escenario, a.limite))

    print("=" * 78)
    print("AQUASENSE-AGENT  |  Ejecucion del agente")
    print("=" * 78)
    print(f"\n  Escenario      : {a.escenario}")
    print(f"  Observaciones  : {total:,}")
    print(f"  Variables      : {len(ds.feature_names)}")

    if n_cal > 0:
        contaminados = sum(etiquetas[i] for i in range(n_cal))
        print(f"  Calibracion    : primeros {a.dias_calibracion} dias ({n_cal:,} pasos)")
        if contaminados:
            print(f"  ATENCION: el periodo de calibracion contiene {contaminados:,} pasos")
            print(f"           marcados como fuga. El operador asumio que estaba limpio")
            print(f"           y no lo estaba; el umbral saldra demasiado alto.")
        else:
            print("  Periodo de calibracion sin fugas: verificado.")
    else:
        print("  Calibracion    : desactivada, se usa el umbral global del modelo")

    with LeakDetectionAgent(a.model, a.log, percentil=a.percentil) as agente:
        resumen = agente.run(observaciones, incluir_etiquetas=etiquetas,
                             n_calibracion=n_cal)
        if n_cal > 0 and not agente.calibrado:
            agente.calibrar(observaciones[:n_cal])

    print(f"\n  Metodo de umbral: {resumen['metodo_umbral']}")
    if agente.calibracion:
        c = agente.calibracion
        print(f"  Percentil {c['periodil']:g} de {c['n_observaciones']:,} observaciones limpias")
        print(f"    puntuaciones: min={c['puntuaciones']['min']:.4f} "
              f"mediana={c['puntuaciones']['mediana']:.4f} "
              f"p90={c['puntuaciones']['p90']:.4f} max={c['puntuaciones']['max']:.4f}")
        print(f"    umbral global del modelo  : {c['umbral_global_previo']}")
        print(f"    umbral adaptado a ESTA red: {c['umbral_adaptado']}")

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