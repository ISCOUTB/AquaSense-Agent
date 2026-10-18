r"""
inspect_data.py - Paso 2: identificacion de los datos de entrada de AquaSense-Agent
=============================================================================

Verifica el FORMATO REAL del benchmark LeakDB (subconjunto Hanoi_CMH) leyendo
los archivos. No inventa columnas, etiquetas ni relaciones: todo lo que imprime
sale de recorrer el dataset.

Arquitectura: dos fases separadas a proposito.
    collect()  -> LOGICA.   Recorre los archivos y devuelve estructuras de datos.
    render()   -> SALIDA.   Solo formatea esos datos para consola y para JSON.
    main()     -> CLI.      Une ambas.

Uso
---
    python inspect_data.py --zip 'C:\Users\Luis Felipe\Downloads\Hanoi_CMH.zip'
    python inspect_data.py --data-dir data      # si ya esta extraido
    python inspect_data.py --quick              # menos estadisticos
    python inspect_data.py --force-extract      # reextrae el ZIP

Solo libreria estandar, a proposito: si faltara pandas, esta herramienta debe
poder diagnosticar el problema de entorno sin depender de el.

No modifica nunca los archivos del dataset: solo lectura.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_ZIP = r"C:\Users\Luis Felipe\Downloads\Hanoi_CMH.zip"
MEASUREMENT_DIRS = ("Demands", "Pressures", "Flows")
N_LEAK_SCENARIOS_HINT = 6


# ==========================================================================
# FASE 1 - LOGICA: leer y comprobar
# ==========================================================================


def rows(path: Path) -> list[list[str]]:
    """Filas del CSV. utf-8-sig elimina el BOM que anteponen algunos exportadores."""
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def values(path: Path) -> list[float]:
    """Segunda columna como float; los valores no numericos pasan a NaN."""
    out = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            try:
                out.append(float(r[1].strip()))
            except (ValueError, IndexError):
                out.append(float("nan"))
    return out


def first_index_at(ts: list[str], target: str) -> int:
    """Primer indice cuyo timestamp es >= target (orden lexicografico = orden temporal)."""
    return next((i for i, t in enumerate(ts) if t >= target), len(ts) - 1)


def scenario_number(path: Path) -> int:
    m = re.search(r"Scenario-(\d+)", path.name)
    return int(m.group(1)) if m else 0


def ensure_extracted(zip_path: Path | None, data_dir: Path, force: bool) -> dict:
    """Deja data_dir poblado. No reextrae si ya hay escenarios, salvo --force."""
    existing = sorted(p.name for p in data_dir.glob("Scenario-*") if p.is_dir()) if data_dir.exists() else []
    info = {"zip": str(zip_path) if zip_path else None,
            "zip_existe": bool(zip_path and zip_path.exists()),
            "data_dir": str(data_dir), "ya_presentes": existing, "accion": None}
    if existing and not force:
        info["accion"] = "datos ya extraidos"
        return info
    if not zip_path or not zip_path.exists():
        info["accion"] = "sin datos y sin ZIP utilizable"
        return info
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        zf.extractall(data_dir)
    info.update(accion=f"ZIP extraido ({len(members)} entradas)", entradas=len(members))
    return info


def read_info(scen: Path) -> dict:
    """Metadatos declarados en Scenario-N_info.csv."""
    files = sorted(scen.glob("*_info.csv"))
    if not files:
        return {}
    return {r[0].strip(): r[1].strip() for r in rows(files[0])[1:] if len(r) >= 2}


def label_stats(scen: Path) -> dict:
    dist = Counter(v for v in values(scen / "Labels.csv") if v == v)
    n0, n1 = dist.get(0.0, 0), dist.get(1.0, 0)
    leaks = scen / "Leaks"
    return {"n0": n0, "n1": n1, "pct1": round(100 * n1 / (n0 + n1), 4) if n0 + n1 else 0.0,
            "n_fugas": len(list(leaks.glob("*_demand.csv"))) if leaks.is_dir() else 0}


def leak_windows(scen: Path, ts: list[str], labels: list[float]) -> dict:
    """REGLA VERIFICADA: Label=1 durante [primer paso con caudal>0, Leak End].

    Devuelve las ventanas detectadas, los pasos esperados y si la regla se cumple.
    """
    leaks = scen / "Leaks"
    demands = sorted(leaks.glob("*_demand.csv")) if leaks.is_dir() else []
    windows, esperado = [], [False] * len(labels)

    for df in demands:
        meta = {r[0].strip(): r[1].strip()
                for r in rows(df.with_name(df.name.replace("_demand.csv", "_info.csv")))[1:]
                if len(r) >= 2}
        caudal = values(df)
        onset = next((i for i, v in enumerate(caudal) if v > 0), None)
        if onset is None:
            continue
        fin = first_index_at(ts, meta.get("Leak End", ts[-1]))
        for i in range(onset, fin + 1):
            esperado[i] = True
        windows.append({
            "nodo": meta.get("Leak Node"), "tipo": meta.get("Leak Type"),
            "area": meta.get("Leak Area"), "diametro": meta.get("Leak Diameter"),
            "leak_start": meta.get("Leak Start"), "pico": meta.get("Peak Time"),
            "onset_caudal": ts[onset], "fin_ventana": ts[fin],
            "retraso_pasos": onset - first_index_at(ts, meta.get("Leak Start", ts[0])),
            "pasos_con_caudal": sum(1 for v in caudal if v > 0),
            "pasos_ventana": fin - onset + 1,
        })

    actual = [v == 1.0 for v in labels]
    return {
        "n_fugas": len(windows),
        "es_control": not windows,
        "ventanas": windows,
        "coincidencia": sum(1 for a, b in zip(actual, esperado) if a == b),
        "total_pasos": len(labels),
        "n_label1": sum(actual),
        "verificado": all(a == b for a, b in zip(actual, esperado)),
    }


def check_alignment(scenarios: list[Path], reference: list[str], quick: bool) -> dict:
    """Todos los archivos deben compartir la misma rejilla temporal (hash de Timestamp)."""
    ref = hashlib.md5("\n".join(reference).encode()).hexdigest()[:10]
    revisados, desalineados = 0, []
    for scen in scenarios[:2] if quick else scenarios:
        for sub in MEASUREMENT_DIRS:
            files = sorted((scen / sub).glob("*.csv")) if (scen / sub).is_dir() else []
            for f in files[:5] if quick else files:
                with f.open("r", newline="", encoding="utf-8-sig") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)
                    col = [r[0] for r in reader if r]
                revisados += 1
                if hashlib.md5("\n".join(col).encode()).hexdigest()[:10] != ref:
                    desalineados.append(f"{scen.name}/{sub}/{f.name}")
    return {"hash_referencia": ref, "archivos_revisados": revisados,
            "desalineados": desalineados, "alineados": not desalineados}


def describe_series(folder: Path, limit: int | None) -> list[dict]:
    files = sorted(folder.glob("*.csv"))[:limit] if limit else sorted(folder.glob("*.csv"))
    out = []
    for f in files:
        with f.open("r", newline="", encoding="utf-8-sig") as fh:
            head = next(csv.reader(fh), [])
        v = [x for x in values(f) if x == x]
        out.append({"archivo": f.name, "columnas": head, "n_filas": len(v),
                    "min": min(v) if v else None, "max": max(v) if v else None,
                    "constante_cero": bool(v) and all(x == 0.0 for x in v)})
    return out


def read_network(inp: Path) -> dict:
    """Secciones relevantes del .inp de EPANET."""
    txt = inp.read_text(encoding="utf-8", errors="replace")

    def sec(name: str) -> list[list[str]]:
        m = re.search(r"^\[" + name + r"\](.*?)(?=^\[)", txt, re.M | re.S)
        return [ln.split(";")[0].strip().split() for ln in (m.group(1).splitlines() if m else [])
                if ln.split(";")[0].strip()]

    return {"inp": inp.name,
            "embalses": [r[0] for r in sec("RESERVOIRS")],
            "nodos_union": len(sec("JUNCTIONS")),
            "tuberias": len(sec("PIPES")),
            "n_emitters": len(sec("EMITTERS"))}


def demands_identify_scenario(scenarios: list[Path]) -> bool:
    """Si las demandas difieren, el escenario es recuperable desde los datos.

    Importa porque 4 escenarios son control (Label=0 todo el ano): si el modelo
    pudiera identificar el escenario, memorizaria la respuesta.
    """
    digests = set()
    for scen in scenarios:
        f = scen / "Demands" / "Node_1.csv"
        if f.exists():
            digests.add(hashlib.md5(f.read_bytes()).hexdigest())
    return len(digests) == len(scenarios)


def collect(zip_path: Path | None, data_dir: Path, force: bool, quick: bool) -> dict:
    """FASE LOGICA completa: devuelve el informe como diccionario. No imprime nada."""
    prep = ensure_extracted(zip_path, data_dir, force)
    scenarios = sorted((p for p in data_dir.glob("Scenario-*") if p.is_dir()), key=scenario_number)
    if not scenarios:
        raise SystemExit(
            f"No hay escenarios en {data_dir}.\n"
            f"  ZIP esperado en: {zip_path}\n"
            "  Descarga Hanoi_CMH de LeakDB, o indica --data-dir.")

    ref = scenarios[0]
    lab_rows = rows(ref / "Labels.csv")
    timestamps = [r[0] for r in lab_rows[1:]]

    out: dict[str, Any] = {"generado": datetime.now().isoformat(timespec="seconds"),
                           "preparacion": prep, "escenarios": {}, "formato": {}, "red": {}}

    for scen in scenarios:
        own = rows(scen / "Labels.csv")[1:]
        out["escenarios"][scen.name] = {
            "carpetas": {d.name: len(list(d.glob("*.csv")))
                         for d in sorted(scen.iterdir()) if d.is_dir()},
            "archivos": sorted(p.name for p in scen.iterdir() if p.is_file()),
            "info": read_info(scen),
            "etiquetas": label_stats(scen),
            "relacion": leak_windows(scen, timestamps, [float(r[1]) for r in own]),
        }

    for sub in MEASUREMENT_DIRS:
        if (ref / sub).is_dir():
            out["formato"][sub] = describe_series(ref / sub, 4 if quick else None)
    out["formato"]["Labels.csv"] = {"columnas": lab_rows[0], "n_filas": len(timestamps),
                                     "dist_escenario_referencia":
                                         {str(k): v for k, v in
                                          Counter(float(r[1]) for r in lab_rows[1:]).items()}}

    dt0 = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
    dt1 = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
    out["linea_temporal"] = {
        "primer_timestamp": timestamps[0], "ultimo_timestamp": timestamps[-1],
        "pasos_por_escenario": len(timestamps),
        "delta_entre_filas": str((dt1 - dt0) / (len(timestamps) - 1)) if len(timestamps) > 1 else None,
        "duracion_total": str(dt1 - dt0),
        "total_filas": len(timestamps) * len(scenarios),
        "clase0": sum(e["etiquetas"]["n0"] for e in out["escenarios"].values()),
        "clase1": sum(e["etiquetas"]["n1"] for e in out["escenarios"].values()),
        "alineacion": check_alignment(scenarios, timestamps, quick),
    }
    out["linea_temporal"]["pct_clase1"] = round(
        100 * out["linea_temporal"]["clase1"] / out["linea_temporal"]["total_filas"], 4)

    out["red"] = read_network(sorted(ref.glob("*.inp"))[0]) if list(ref.glob("*.inp")) else {}
    out["red"]["presiones_degeneradas"] = [
        d["archivo"].replace("Node_", "").replace(".csv", "")
        for d in out["formato"].get("Pressures", [])
        if d["constante_cero"]]
    # Las presiones degeneradas no dependen de --quick: se comprueban todas.
    for presion in sorted((ref / "Pressures").glob("*.csv")):
        if all(v == 0.0 for v in values(presion)):
            n = presion.stem.replace("Node_", "")
            if n not in out["red"]["presiones_degeneradas"]:
                out["red"]["presiones_degeneradas"].append(n)

    out["variables"] = {
        "series_presion": len(list((ref / "Pressures").glob("*.csv"))),
        "series_caudal": len(list((ref / "Flows").glob("*.csv"))),
        "n_escenarios": len(scenarios),
        "demandas_identifican_escenario": demands_identify_scenario(scenarios),
        "prohibidas_como_entrada": ["Labels.csv", "Leaks/Leak_*_demand.csv",
                                   "Leaks/Leak_*_info.csv", "identificador Scenario-N"],
    }
    out["relacion_verificada"] = all(
        e["relacion"]["es_control"] or e["relacion"]["verificado"] for e in out["escenarios"].values())
    return out


# ==========================================================================
# FASE 2 - SALIDA: formatear lo recogido (sin volver a leer el dataset)
# ==========================================================================


SEP = "=" * 78


def sec(t: str) -> None:
    print("\n" + SEP + f"\n{t}\n" + SEP)


def kv(k: str, v: Any, w: int = 34) -> None:
    print(f"  {k:<{w}} : {v}")


def nota(msg: str, tag: str = "") -> None:
    print(f"  {tag:<8}{msg}" if tag else f"  {msg}")


def render(d: dict) -> None:
    esc, tmp, rel = d["escenarios"], d["linea_temporal"], d["variables"]

    sec("0. PREPARACION")
    for k, v in d["preparacion"].items():
        kv(k, v if not isinstance(v, list) else f"{len(v)} escenarios")

    sec("1. INVENTARIO DE ESCENARIOS")
    kv("Escenarios", len(esc))
    carpetas = sorted({c for e in esc.values() for c in e["carpetas"]})
    print()
    for c in carpetas:
        n = [e["carpetas"].get(c, 0) for e in esc.values()]
        nota(f"{c:<10} -> {min(n)}-{max(n)} CSV por escenario")
    print("\n  Por escenario:")
    for name, e in sorted(esc.items(), key=lambda kv_: scenario_number(Path(kv_[0]))):
        c = ", ".join(f"{k}={v}" for k, v in sorted(e["carpetas"].items()))
        nota(f"{name:<12} {c}")

    sec("2. METADATOS DECLARADOS (*_info.csv)")
    claves = [k for k in next(iter(esc.values()))["info"]]
    print()
    print("  " + f"{'Scenario':<12}" + "".join(f"{k[:22]:<24}" for k in claves))
    for name, e in sorted(esc.items(), key=lambda kv_: scenario_number(Path(kv_[0]))):
        print("  " + f"{name:<12}" + "".join(f"{str(e['info'].get(k,'-'))[:22]:<24}" for k in claves))
    ref_info = next(iter(esc.values()))["info"]
    print()
    for campo in ("Network_Name", "Duration", "Time_Step"):
        if campo in ref_info:
            kv(campo, ref_info[campo])
    nota("La incertidumbre de longitud/diametro/rugosidad cambia entre escenarios:")
    nota("cada uno es una realizacion hidraulica distinta de la misma red.")

    sec("3. COLUMNAS Y FORMATO REAL")
    for tipo, series in d["formato"].items():
        if tipo == "Labels.csv":
            kv("Labels.csv columnas", series["columnas"])
            kv("Labels.csv filas", series["n_filas"])
            kv("Labels.csv valores (esc. ref.)", series["dist_escenario_referencia"])
            nota("Etiqueta BINARIA por instante, no por evento.")
            continue
        print(f"\n  --- {tipo}/ ({len(series)} archivos mostrados) ---")
        for s in series:
            nota(f"{s['archivo']:<14} columnas={s['columnas']} filas={s['n_filas']:,} "
                 f"min={s['min']:.2f} max={s['max']:.2f}")

    sec("4. REJILLA TEMPORAL Y DESBALANCE")
    kv("Primer timestamp", tmp["primer_timestamp"])
    kv("Ultimo timestamp", tmp["ultimo_timestamp"])
    kv("Pasos por escenario", f"{tmp['pasos_por_escenario']:,}")
    kv("Delta entre filas", tmp["delta_entre_filas"])
    kv("Duracion total", tmp["duracion_total"])
    print()
    print(f"  {'Scenario':<12}{'0.0 (normal)':>16}{'1.0 (fuga)':>14}{'% fuga':>10}{'Fugas':>8}")
    for name, e in sorted(esc.items(), key=lambda kv_: scenario_number(Path(kv_[0]))):
        t = e["etiquetas"]
        print(f"  {name:<12}{t['n0']:>16,}{t['n1']:>14,}{t['pct1']:>9.2f}%{t['n_fugas']:>8}")
    print()
    kv("TOTAL dataset", f"{tmp['total_filas']:,} filas")
    kv("Clase 1.0 (fuga)", f"{tmp['pct_clase1']:.2f}%")
    nota(f"Desbalance: predecir siempre 'normal' ya da {100 - tmp['pct_clase1']:.2f}% de exactitud,")
    nota("por eso la exactitud sola no basta: hacen falta precision, recall y F1.")
    al = tmp["alineacion"]
    kv("\nArchivos con rejilla verificados", al["archivos_revisados"])
    nota("Todos comparten la MISMA rejilla temporal." if al["alineados"]
         else f"DESALINEADOS: {al['desalineados'][:3]}", "  [OK]" if al["alineados"] else "  [AVISO]")

    sec("5. RELACION ENTRE Labels.csv Y LAS FUGAS (verificacion)")
    nota("Regla verificada: Label=1 durante [primer paso con caudal>0, Leak End].")
    for name, e in sorted(esc.items(), key=lambda kv_: scenario_number(Path(kv_[0]))):
        r = e["relacion"]
        if r["es_control"]:
            nota(f"{name:<12} sin fugas -> ESCENARIO CONTROL (etiqueta siempre 0.0)")
            continue
        print(f"\n  {name}")
        for w in r["ventanas"]:
            ret = "" if w["retraso_pasos"] == 0 else f"  (+{w['retraso_pasos']} paso de retraso)"
            nota(f"fuga nodo {w['nodo']:>2}  tipo={w['tipo']:<9} area={float(w['area']):.2e} m2")
            nota(f"  Leak Start declarado: {w['leak_start']}{ret}")
            nota(f"  primer caudal > 0     : {w['onset_caudal']}")
            nota(f"  etiqueta hasta        : {w['fin_ventana']}")
            nota(f"  pasos con caudal>0    : {w['pasos_con_caudal']:,} de {w['pasos_ventana']:,}")
        nota(f"coincidencia: {r['coincidencia']:,}/{r['total_pasos']:,} -> "
             f"{'EXACTA' if r['verificado'] else 'FALLA'}")
        if any(w["pasos_con_caudal"] < w["pasos_ventana"] for w in r["ventanas"]):
            nota("AVISO: Label=1 cubre mas pasos que los que tienen caudal. Son fugas")
            nota("       'incipient': la ETIQUETA marca toda la ventana, el caudal se corta.")
            nota("       Leak_*_demand.csv NO es la etiqueta; usarlo seria filtracion.")
    print()
    nota("Regla confirmada al 100% en los escenarios con fuga." if d["relacion_verificada"]
         else "La regla falla en algun escenario.", "  [OK]" if d["relacion_verificada"] else "  [AVISO]")

    sec("6. RED HIDRAULICA (.inp)")
    for k, v in d["red"].items():
        kv(k, v)
    nota("Las fugas NO estan como EMITTERS en el .inp: viven en Leaks/.")
    nota("El .inp por si solo no delata que escenario tiene fuga.")
    if d["red"].get("presiones_degeneradas"):
        nota(f"Presiones siempre 0.0 (nodos {', '.join(d['red']['presiones_degeneradas'])}):")
        nota("  son embalses; EPANET no define presion ahi. Excluir de las entradas.")

    sec("7. ENTRADAS LEGITIMAS Y FILTRACION DE ETIQUETAS")
    print("\n  USABLES (datos que un sensor hidraulico mediria):")
    kv("Pressures/Node_*.csv", f"{rel['series_presion']} series")
    kv("Flows/Link_*.csv", f"{rel['series_caudal']} series")
    kv("Demands/Node_*.csv", "demanda por nodo (dato de contorno)")
    print("\n  PROHIBIDAS como entrada:")
    for p in rel["prohibidas_como_entrada"]:
        nota(f"- {p}")
    print()
    nota("Comprobacion: las demandas son distintas en "
         f"{rel['n_escenarios']} de {rel['n_escenarios']} escenarios."
         if rel["demandas_identifican_escenario"] else "Comprobacion: demandas repetidas.")
    nota("La demanda identifica el escenario, y con el escenario se sabe si hay fuga.")
    nota("Por eso la division de datos DEBE ser por escenario, nunca aleatoria")
    nota("sobre filas: si no, el modelo memoriza y la evaluacion es optimista.")
    print()
    kv("Escenarios", rel["n_escenarios"])
    kv("Pasos por escenario", f"{tmp['pasos_por_escenario']:,}")
    kv("Filas totales", f"{tmp['total_filas']:,}")

    sec("8. RESUMEN EJECUTIVO")
    control = [n for n, e in sorted(esc.items(), key=lambda kv_: scenario_number(Path(kv_[0])))
               if e["relacion"]["es_control"]]
    con_fuga = [n for n, e in sorted(esc.items(), key=lambda kv_: scenario_number(Path(kv_[0])))
                if not e["relacion"]["es_control"]]
    print(f"""
  Dataset    : {rel['n_escenarios']} escenarios Hanoi_CMH, {tmp['total_filas']:,} filas
  Cobertura  : {tmp['primer_timestamp']} -> {tmp['ultimo_timestamp']} (paso 30 min)
  Entradas   : {rel['series_presion']} series de presion + {rel['series_caudal']} de caudal
  Etiqueta   : binaria por instante en Labels.csv (Timestamp, Label)
  Control    : {', '.join(control)}
  Con fuga   : {', '.join(con_fuga)}
  Desbalance : {tmp['pct_clase1']:.2f}% de clase 1.0
  Regla      : {'VERIFICADA en todos los escenarios' if d['relacion_verificada'] else 'NO VERIFICADA'}
  Division   : POR ESCENARIO (obligatorio, ver seccion 7)
""")


# ==========================================================================
# CLI
# ==========================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Inspecciona el formato real del subconjunto Hanoi_CMH de LeakDB.")
    p.add_argument("--zip", default=DEFAULT_ZIP, help="Ruta del ZIP de LeakDB")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Datos ya extraidos")
    p.add_argument("--report", type=Path, default=Path("logs/data_inspection.json"),
                   help="Informe JSON de salida")
    p.add_argument("--quick", action="store_true", help="Menos estadisticos; ejecuta antes")
    p.add_argument("--force-extract", action="store_true", help="Reextrae el ZIP")
    a = p.parse_args(argv)

    print(SEP + "\nAQUASENSE-AGENT  |  Inspeccion de datos  |  LeakDB / Hanoi_CMH\n" + SEP)
    datos = collect(Path(a.zip) if a.zip else None, a.data_dir, a.force_extract, a.quick)
    render(datos)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Informe JSON: {a.report.resolve()}")
    print(SEP + "\nInspeccion completada. No se modifico ningun archivo del dataset.\n" + SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())