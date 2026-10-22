import os
from pathlib import Path
from helpers import FEATURES
import json
import mlflow

def configure_tracking(state_dir=None, allow_create=True):
    if state_dir is None:
        state_dir = Path(os.environ["LOCALAPPDATA"]) / "campaign-mini-state"

    state = Path(state_dir).resolve()
    database = state / "mlflow.db"
    artifacts = state / "artifacts"

    if allow_create:
        state.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(exist_ok=True)
    elif not database.is_file() or not artifacts.is_dir():
        raise FileNotFoundError("The selected MLflow store is missing")

    tracking_uri = "sqlite:///" + database.as_posix()
    mlflow.set_tracking_uri(tracking_uri)

    return {
        "state_dir": str(state),
        "tracking_uri": tracking_uri,
        "artifact_uri": artifacts.as_uri(),
    }

def load_selected(selection_path):
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection["features"] != FEATURES:
        raise ValueError("Saved features do not match helpers.FEATURES")
    configure_tracking(state_dir=selection["state_dir"], allow_create=False)
    model = mlflow.sklearn.load_model(selection["model_uri"])
    return model, selection
