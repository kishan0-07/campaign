import json
from pathlib import Path
import subprocess
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature

from helpers import FEATURES, build_pipeline, positive_scores, prepare_inputs, rank_customers
from tracking import configure_tracking, load_selected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = {"budget_numerator": 1, "budget_denominator": 10, "tie_break": "record_id_ascending"}


@pytest.fixture
def temporary_tracking(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    if mlflow.active_run() is not None:
        pytest.fail("End the active run before running isolated integration tests.")
    context = configure_tracking(state_dir=tmp_path / "state", allow_create=True)
    experiment_id = mlflow.create_experiment("tracking-test", artifact_location=context["artifact_uri"])
    try:
        yield context, experiment_id
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(previous_uri)


def test_native_round_trip_and_fresh_cli(tmp_path, temporary_tracking):
    context, experiment_id = temporary_tracking
    row = np.arange(40)
    raw = pd.DataFrame({
        "record_id": row, "age": 20 + row % 30, "previous": row % 4,
        "job": "admin.", "marital": "married", "education": "high.school",
        "housing": "yes", "loan": "no", "poutcome": "nonexistent",
        "y": np.where(row % 4 == 3, "yes", "no"),
    })
    ids, X, y = prepare_inputs(raw)
    model = build_pipeline("lr_c1").fit(X, y)
    example = X.head(3)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        info = mlflow.sklearn.log_model(
            model, name="pipeline", input_example=example,
            signature=infer_signature(example, model.predict_proba(example)),
            pyfunc_predict_fn="predict_proba", serialization_format="cloudpickle",
        )
    selection = {
        "candidate_id": "lr_c1", "run_id": run.info.run_id,
        "model_uri": info.model_uri, "state_dir": context["state_dir"],
        "features": list(FEATURES), "policy": dict(POLICY),
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection, allow_nan=False), encoding="utf-8")
    loaded, saved = load_selected(selection_path)
    assert saved == selection
    np.testing.assert_allclose(positive_scores(loaded, X), positive_scores(model, X),
                               rtol=1e-7, atol=1e-10)

    demo_path, output_path = tmp_path / "demo.csv", tmp_path / "ranked.csv"
    raw.head(20).drop(columns="y").to_csv(demo_path, index=False)
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "predict.py"), str(demo_path), str(output_path),
        "--selection", str(selection_path),
    ], check=True, capture_output=True, text=True, cwd=tmp_path)
    expected = rank_customers(ids.head(20), positive_scores(model, X.head(20)), POLICY)
    actual = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-7, atol=1e-10)
    assert len(actual) == 20 and actual["selected"].sum() == 2


@pytest.mark.parametrize("missing_part", ["directory", "database", "artifacts"])
def test_load_only_requires_existing_state(tmp_path, missing_part):
    state = tmp_path / "incomplete-state"
    if missing_part != "directory":
        state.mkdir()
    if missing_part == "database":
        (state / "artifacts").mkdir()
    if missing_part == "artifacts":
        (state / "mlflow.db").touch()
    previous_uri = mlflow.get_tracking_uri()
    with pytest.raises(FileNotFoundError):
        configure_tracking(state_dir=state, allow_create=False)
    assert mlflow.get_tracking_uri() == previous_uri
    if missing_part == "directory":
        assert not state.exists()


def test_feature_mismatch_stops_before_loading(tmp_path, monkeypatch):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"features": FEATURES + ["y"]}), encoding="utf-8")

    def unexpected_load(*args, **kwargs):
        pytest.fail("A feature mismatch must fail before SDK loading.")

    monkeypatch.setattr(mlflow.sklearn, "load_model", unexpected_load)
    with pytest.raises(ValueError, match="Saved features"):
        load_selected(selection_path)