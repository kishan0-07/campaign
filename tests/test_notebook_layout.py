"""Regression checks for the randomized notebook's isolated entry point."""

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def random_entry_cell():
    notebook = json.loads((PROJECT_ROOT / "analysisRandom.ipynb").read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


@pytest.mark.parametrize("marker", ["final_access.json", "final_result.json"])
def test_random_notebook_stops_before_tracking_or_writes(tmp_path, monkeypatch, marker):
    (tmp_path / "pyproject.toml").touch()
    results = tmp_path / "results_Random"
    results.mkdir()
    record = results / marker
    record.write_text("{}", encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.chdir(tmp_path)
    namespace = {}

    with pytest.raises(RuntimeError, match="randomized experiment is already recorded"):
        exec(compile(random_entry_cell(), "analysisRandom.ipynb:first-cell", "exec"), namespace)

    assert "mlflow" not in namespace
    assert namespace["RUN_FINAL"] is False
    assert namespace["DATA_DIR"] == tmp_path / "data_Random"
    assert namespace["RESULTS_DIR"] == tmp_path / "results_Random"
    after = {path.relative_to(tmp_path): path.read_bytes()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_random_entry_does_not_use_original_final_records(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    original_results = tmp_path / "results"
    original_results.mkdir()
    (original_results / "final_access.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    namespace = {}

    exec(compile(random_entry_cell(), "analysisRandom.ipynb:first-cell", "exec"), namespace)

    assert namespace["RESULTS_DIR"] == tmp_path / "results_Random"
    assert not (tmp_path / "results_Random").exists()
    assert not (tmp_path / "data_Random").exists()
