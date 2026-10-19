from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dataset_manager import SCENARIOS, dataset_is_ready, download_dataset


def _dataset_zip() -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for scenario in SCENARIOS:
            for folder in ("Demands", "Pressures", "Flows"):
                archive.writestr(f"{scenario}/{folder}/sample.csv", "value\n1\n")
            archive.writestr(f"{scenario}/Labels.csv", "Timestamp,Label\n")
    return archive_buffer.getvalue()


class _FakeResponse(io.BytesIO):
    def __init__(self, content: bytes):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}


def test_download_extracts_all_scenarios_and_reports_progress(tmp_path: Path):
    content = _dataset_zip()
    progress: list[tuple[int, int | None]] = []

    with patch("dataset_manager.urllib.request.urlopen", return_value=_FakeResponse(content)):
        downloaded = download_dataset(tmp_path / "data", lambda received, total: progress.append((received, total)))

    assert downloaded == len(content)
    assert dataset_is_ready(tmp_path / "data")
    assert progress[-1] == (len(content), len(content))


def test_download_rejects_zip_path_traversal(tmp_path: Path):
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with patch(
        "dataset_manager.urllib.request.urlopen",
        return_value=_FakeResponse(archive_buffer.getvalue()),
    ):
        with pytest.raises(ValueError, match="Ruta no válida"):
            download_dataset(tmp_path / "data")

    assert not (tmp_path / "outside.txt").exists()


def test_download_does_not_overwrite_nonempty_incomplete_directory(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="incompletos"):
        download_dataset(data_dir)

    assert (data_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
