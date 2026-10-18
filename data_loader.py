"""
data_loader.py - Carga del subconjunto Hanoi_CMH de LeakDB
==========================================================

Unica fuente de verdad sobre el formato del dataset. La usan train.py,
evaluate.py y agent.py, para que los tres vean exactamente las mismas
variables en el mismo orden.

Reglas que este modulo respeta (verificadas en inspect_data.py):

  1. Todos los archivos comparten la MISMA rejilla temporal de 30 min.
     Aun asi se verifica con una asercion antes de unir, en vez de confiar.

  2. Se EXCLUYE del conjunto de entradas:
       - Labels.csv ................ es la variable objetivo
       - Leaks/ .................... es la verdad de terreno de la fuga
       - Scenario-N ............... identifica el escenario, y con el
                                     escenario se sabe si hay fuga
       - nodos con presion siempre 0 (el embalse) ..... no aportan nada

  3. La etiqueta se lee aparte y nunca se concatena con las entradas.

Formatos (confirmados leyendo los archivos, no de documentacion externa):
    Pressures/Node_<n>.csv  ->  Timestamp, Value
    Flows/Link_<n>.csv      ->  Timestamp, Value
    Demands/Node_<n>.csv    ->  Timestamp, Value
    Labels.csv              ->  Timestamp, Label
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRESSURES = "Pressures"
FLOWS = "Flows"
DEMANDS = "Demands"

RANDOM_SEED = 42


@dataclass
class Dataset:
    """Dataset listo para el modelo, con la trazabilidad de su origen."""

    X: pd.DataFrame            # filas = pasos, columnas = variables de entrada
    y: np.ndarray              # 0.0 normal / 1.0 fuga
    scenario: np.ndarray       # 'Scenario-3' ... una etiqueta por fila
    timestamp: np.ndarray      # marca de tiempo por fila
    feature_names: list[str]
    excluded: dict[str, list[str]]   # que se dejo fuera y por que

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def scenarios(self) -> list[str]:
        return sorted(set(self.scenario))

    @property
    def y_int(self) -> np.ndarray:
        return self.y.astype(int)

    def describe(self) -> str:
        n1 = int(self.y_int.sum())
        return (f"{self.n_samples:,} filas | {len(self.feature_names)} variables | "
                f"{len(self.scenarios)} escenarios | fuga {n1:,} ({100 * n1 / self.n_samples:.2f}%)")


def scenario_number(path_or_name: Path | str) -> int:
    name = Path(path_or_name).name
    m = re.search(r"Scenario-(\d+)", name)
    return int(m.group(1)) if m else 0


def discover_scenarios(data_dir: Path) -> list[Path]:
    """Escenarios ordenados de forma numerica (Scenario-2 antes que Scenario-10)."""
    return sorted((p for p in data_dir.glob("Scenario-*") if p.is_dir()),
                  key=scenario_number)


def _read_series(path: Path) -> pd.Series:
    """Lee un CSV de serie temporal y devuelve la columna 'Value' indexada por Timestamp."""
    df = pd.read_csv(path, usecols=["Timestamp", "Value"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return pd.Series(df["Value"].to_numpy(dtype=np.float64),
                     index=df["Timestamp"], name=path.stem)


def _read_block(folder: Path, prefix: str, tag: str) -> pd.DataFrame | None:
    """Concatena todas las series de una carpeta en un DataFrame indexado por tiempo.

    tag da nombre legible a las columnas: press_node_2, flow_link_1, demand_node_2.
    """
    files = sorted(folder.glob(f"{prefix}_*.csv"),
                   key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)))
    if not files:
        return None
    names = [f"{tag}_{f.stem.split('_', 1)[1]}" for f in files]
    return pd.concat([_read_series(f) for f in files], axis=1, keys=names)


def _assert_same_grid(blocks: list[pd.DataFrame], reference: pd.Index, context: str) -> None:
    for b in blocks:
        if not b.index.equals(reference):
            raise ValueError(
                f"Rejillas temporales distintas en {context}.\n"
                f"  {reference[0]} .. {reference[-1]}  vs  {b.index[0]} .. {b.index[-1]}\n"
                "Las series de LeakDB deben compartir la misma rejilla de 30 min.")


def load_dataset(data_dir: Path = Path("data"),
                 include_demands: bool = False) -> Dataset:
    """Carga los 10 escenarios y devuelve el Dataset.

    include_demands
        Las demandas son un dato legitimo del operador, pero tambien
        identifican univocamente al escenario (verificado en inspect_data.py).
        Se dejan fuera por defecto: no aportan senal dentro de un mismo
        escenario y anadirian 32 variables que actuan como huella del
        escenario. Activalas solo para un experimento de ablicacion.
    """
    data_dir = Path(data_dir)
    scenarios = discover_scenarios(data_dir)
    if not scenarios:
        raise SystemExit(
            f"No hay carpetas 'Scenario-*' en {data_dir}.\n"
            "Coloca Hanoi_CMH en data/ o ejecuta antes:  python inspect_data.py --zip <ruta>")

    # 1) Etiquetas y rejilla de referencia (solo se usa para verificar).
    labels = pd.read_csv(scenarios[0] / "Labels.csv")
    grid = pd.DatetimeIndex(pd.to_datetime(labels["Timestamp"]).to_numpy())
    n_steps = len(grid)

    frames: list[pd.DataFrame] = []
    y_parts, scen_parts, ts_parts = [], [], []

    for scen in scenarios:
        own = pd.read_csv(scen / "Labels.csv")
        own_grid = pd.DatetimeIndex(pd.to_datetime(own["Timestamp"]).to_numpy())
        if len(own_grid) != n_steps or not own_grid.equals(grid):
            raise ValueError(f"{scen.name}: Labels.csv no coincide con la rejilla de referencia.")

        press = _read_block(scen / PRESSURES, "Node", "press")
        flows = _read_block(scen / FLOWS, "Link", "flow")
        if press is None or flows is None:
            raise ValueError(f"{scen.name}: faltan las carpetas {PRESSURES}/ o {FLOWS}/.")
        _assert_same_grid([press, flows], grid, f"{scen.name}/{PRESSURES}+{FLOWS}")

        parts = [press, flows]
        if include_demands:
            dem = _read_block(scen / DEMANDS, "Node", "demand")
            if dem is not None:
                parts.append(dem)

        frames.append(pd.concat(parts, axis=1))
        y_parts.append(own["Label"].to_numpy(dtype=np.float64))
        scen_parts.append(np.full(len(own), scen.name))
        ts_parts.append(grid.to_numpy())

    X = pd.concat(frames, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    scenario = np.concatenate(scen_parts)
    timestamp = np.concatenate(ts_parts)

    # 2) Columnas degeneradas: una serie siempre a cero no aporta informacion.
    #    En Hanoi_CMH es el embalse (presion no definida por EPANET).
    #    Se detecta sobre los datos, no se codifica a mano el numero de nodo.
    zero_all = (X == 0).all(axis=0)
    degeneradas = list(X.columns[zero_all])

    # 3) Columnas constantes: tampoco aportan varianza al clasificador.
    constantes = [c for c in X.columns
                  if c not in degeneradas and X[c].nunique(dropna=False) == 1]

    excluded = {
        "presion_sin_variacion": degeneradas,
        "constantes": constantes,
        "demandas_excluidas": ([] if include_demands
                               else list(_read_block(scenarios[0] / DEMANDS, "Node", "demand").columns)),
        "no_es_entrada": ["Labels.csv (es la etiqueta)", "Leaks/ (verdad de terreno)",
                          "Scenario-N (identificador del escenario)"],
    }

    X = X.drop(columns=degeneradas + constantes)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Las entradas contienen NaN o infinito. Revisar la carga.")

    return Dataset(X=X, y=y, scenario=scenario, timestamp=timestamp,
                   feature_names=list(X.columns), excluded=excluded)


if __name__ == "__main__":
    ds = load_dataset()
    print(ds.describe())
    print(f"\nVariables de entrada ({len(ds.feature_names)}):")
    print("  " + ", ".join(ds.feature_names[:8]) + " ...")
    print(f"\nExcluidas: {ds.excluded['presion_sin_variacion']}")