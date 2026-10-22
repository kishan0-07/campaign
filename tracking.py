import json
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
from helpers import FEATURES


def configure_tracking(state_dir=None, *, allow_create=True):
    if state_dir is None:
        state_dir = Path(os.environ["LOCALAPPDATA"]) / "campaign-mini-state"
    state_dir = Path(state_dir).expanduser().resolve()
    database = state_dir / "mlflow.db"
    artifacts = state_dir / "artifacts"
    if allow_create:
        artifacts.mkdir(parents=True, exist_ok=True)
    elif not database.is_file() or not artifacts.is_dir():
        raise FileNotFoundError(f"Existing mlflow.db and artifacts/ are required in {state_dir}")
    tracking_uri = "sqlite:///" + database.as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    return {"state_dir": str(state_dir), "tracking_uri": tracking_uri,
            "artifact_uri": artifacts.as_uri()}

def load_selected(selection_path):
    """Load our own saved native sklearn pipeline and its decision record."""
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection["features"] != FEATURES:
        raise ValueError("Saved features do not match helpers.FEATURES")
    configure_tracking(state_dir=selection["state_dir"], allow_create=False)
    model = mlflow.sklearn.load_model(selection["model_uri"])
    return model, selection
