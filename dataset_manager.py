"""Download and validate the LeakDB Hanoi_CMH dataset."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

DATASET_URL = (
    "https://raw.githubusercontent.com/KIOS-Research/LeakDB/refs/heads/master/CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip"
)
SCENARIOS = tuple(f"Scenario-{number}" for number in range(1, 11))
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def dataset_is_ready(data_dir: Path) -> bool:
    """Return whether all ten scenarios have the structure required by the app."""
    return all(
        (data_dir / scenario / "Labels.csv").is_file()
        and all(
            (data_dir / scenario / folder).is_dir()
            and any((data_dir / scenario / folder).glob("*.csv"))
            for folder in ("Demands", "Pressures", "Flows")
        )
        for scenario in SCENARIOS
    )


def _extract_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        relative_path = PurePosixPath(member.filename.replace("\\", "/"))
        if (relative_path.is_absolute() or ".." in relative_path.parts
                or (relative_path.parts and ":" in relative_path.parts[0])):
            raise ValueError(f"Ruta no válida dentro del ZIP: {member.filename}")

        target = (destination / Path(*relative_path.parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Ruta fuera del directorio de extracción: {member.filename}") from error

        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def download_dataset(
    data_dir: Path,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> int:
    """Download and extract LeakDB into an empty data directory.

    Returns the number of bytes downloaded. Existing complete data is left alone.
    """
    data_dir = data_dir.resolve()
    if dataset_is_ready(data_dir):
        return 0
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(
            f"{data_dir} ya contiene datos incompletos. Muévelos o vacía la carpeta "
            "antes de descargar de nuevo."
        )

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "AquaSense-Agent/1.0"},
    )
    with tempfile.TemporaryDirectory(prefix=".leakdb-", dir=data_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / "Hanoi_CMH.zip"
        downloaded = 0

        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length and content_length.isdigit() else None
            with archive_path.open("wb") as archive_file:
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    archive_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded, total)
        if total is not None and downloaded != total:
            raise OSError(f"Descarga incompleta: llegaron {downloaded} de {total} bytes.")

        extraction_dir = temporary_dir / "extracted"
        extraction_dir.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            required_space = sum(member.file_size for member in archive.infolist())
            available_space = shutil.disk_usage(extraction_dir).free
            if available_space < required_space:
                raise OSError(
                    f"Espacio insuficiente para extraer el dataset: se requieren "
                    f"{required_space:,} bytes y hay {available_space:,} disponibles."
                )
            _extract_safely(archive, extraction_dir)

        dataset_root = extraction_dir / "Hanoi_CMH"
        if not dataset_is_ready(dataset_root):
            dataset_root = extraction_dir
        if not dataset_is_ready(dataset_root):
            raise ValueError(
                "El ZIP descargado no contiene los 10 escenarios Hanoi_CMH "
                "con sus carpetas Demands, Pressures, Flows y Labels.csv."
            )

        if data_dir.exists():
            data_dir.rmdir()
        shutil.move(str(dataset_root), str(data_dir))

    return downloaded
