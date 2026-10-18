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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PRESSURES = "Pressures"
FLOWS = "Flows"
DEMANDS = "Demands"

RANDOM_SEED = 42

# Variables temporales opcionales: (etiqueta, metodo, ventana_en_pasos).
# Ventana TRASERA: solo pasado y presente, nunca futuro.
#   x6  = maximo en las ultimas 6 horas  (12 pasos de 30 min)
#   x24 = maximo en las ultimas 24 horas (48 pasos)
#   x48 = maximo en las ultimas 48 horas (96 pasos)
# El maximo, y no la media, es lo que conserva la senal de una fuga
# intermitente: si hubo caudal de fuga hace 3 horas, el maximo lo recuerda,
# mientras que la media lo diluye. Medido en el cribado: la media empeora
# el AUC, el maximo lo mejora.
TEMPORAL_PRESETS: dict[str, list[tuple[str, str, int]]] = {
    "ninguno": [],
    "max6-24-48": [("x6", "max", 12), ("x24", "max", 48), ("x48", "max", 96)],
    "max24": [("x24", "max", 48)],
}


@dataclass
class Dataset:
    """Dataset listo para el modelo, con la trazabilidad de su origen."""

    X: pd.DataFrame            # filas = pasos, columnas = variables de entrada
    y: np.ndarray              # 0.0 normal / 1.0 fuga
    scenario: np.ndarray       # 'Scenario-3' ... una etiqueta por fila
    timestamp: np.ndarray      # marca de tiempo por fila
    feature_names: list[str]
    excluded: dict[str, list[str]]   # que se dejo fuera y por que
    temporal: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def base_names(self) -> list[str]:
        """Solo las variables instantaneas, sin las temporales."""
        etiquetas = {tag for tag, _, _ in self.temporal}
        return [n for n in self.feature_names
                if not any(n.endswith(f"_{t}") for t in etiquetas)]

    @property
    def ventana_maxima(self) -> int:
        return max((w for _, _, w in self.temporal), default=0)

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

    def temporal_descrito(self) -> str:
        if not self.temporal:
            return "solo variables instantaneas"
        return ", ".join(f"{tag} ({metodo}, {w} pasos)" for tag, metodo, w in self.temporal)


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


def agregar_temporales(X: pd.DataFrame, escenario: np.ndarray,
                       specs: list[tuple[str, str, int]]) -> pd.DataFrame:
    """Anade rolling CAUSAL, calculado DENTRO de cada escenario.

    Dos cuidados que no son opcionales:

    1. Nunca se cruza la frontera entre escenarios. X es la concatenacion de
       los 10 escenarios; un rolling sobre el bloque entero mezclaria el final
       de una red con el principio de la siguiente.

    2. min_periods=1: al principio de cada escenario la ventana esta incompleta
       y se usa lo que haya. Es lo que hace un sistema real al arrancar, y
       ademas evita NaN. El agente en produccion tiene el mismo comportamiento
       durante su fase de calentamiento.
    """
    if not specs:
        return X
    m = np.asarray(escenario)
    nombres = [f"{c}_{tag}" for tag, _, _ in specs for c in X.columns]
    datos = {n: np.full(len(X), np.nan) for n in nombres}
    for esc in sorted(set(m)):
        mask = m == esc
        bloque = X[mask]
        for tag, metodo, w in specs:
            r = getattr(bloque.rolling(w, min_periods=1), metodo)()
            for c in X.columns:
                datos[f"{c}_{tag}"][mask] = r[c].to_numpy()
    return pd.concat([X, pd.DataFrame(datos, index=X.index)], axis=1)


def load_dataset(data_dir: Path = Path("data"),
                 include_demands: bool = False,
                 temporal: str = "ninguno") -> Dataset:
    """Carga los 10 escenarios y devuelve el Dataset.

    temporal
        Lista de variables temporales a calcular sobre las instantaneas.
        'ninguno' (por defecto), 'max6-24-48' o 'max24'. Ver TEMPORAL_PRESETS.

    include_demands
        Las demandas son un dato legitimo del operador, pero tambien
        identifican univocamente al escenario (verificado en inspect_data.py).
        Se dejan fuera por defecto: no aportan senal dentro de un mismo
        escenario y anadirian 32 variables que actuan como huella del
        escenario. Activalas solo para un experimento de ablicacion.
    """
    specs = TEMPORAL_PRESETS[temporal] if isinstance(temporal, str) else list(temporal)
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

    if specs:
        X = agregar_temporales(X, scenario, specs)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Las entradas contienen NaN o infinito. Revisar la carga.")

    return Dataset(X=X, y=y, scenario=scenario, timestamp=timestamp,
                   feature_names=list(X.columns), excluded=excluded,
                   temporal=specs)


if __name__ == "__main__":
    for nombre in ("ninguno", "max6-24-48"):
        ds = load_dataset(temporal=nombre)
        print(f"[{nombre}] {ds.describe()}")
        print(f"   temporales: {ds.temporal_descrito()}")
        print(f"   instantiated vars: {len(ds.base_names)} | total: {len(ds.feature_names)}")
        print(f"   excluded: {ds.excluded['presion_sin_variacion']}")