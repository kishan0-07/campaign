import os
from pathlib import Path

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