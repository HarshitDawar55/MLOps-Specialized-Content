"""
One place that decides how every session talks to MLflow.

IMPORTANT, AND A REAL TRAP
  MLflow 3.x raises on the old filesystem tracking backend ('./mlruns').
  It is in maintenance mode and the library refuses it unless you opt out
  with MLFLOW_ALLOW_FILE_STORE=true. Every session here uses a SQLite
  backend instead, which is the supported path and is still a single local
  file with no server to run.

  If you follow an older blog post mid-training and type
  `--backend-store-uri artifacts/mlruns`, you will get an exception in front
  of the room. Use the command printed by ui_command() below.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set before mlflow is imported anywhere, or the hints still print.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DB_PATH = ARTIFACTS / "mlflow.db"
ARTIFACT_ROOT = ARTIFACTS / "mlartifacts"

TRACKING_URI = f"sqlite:///{DB_PATH}"


def init(experiment: str) -> str:
    """Point MLflow at the local SQLite store and select an experiment."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)

    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(experiment)
    return TRACKING_URI


def ui_command(port: int = 5000) -> str:
    """The exact command to start the UI against this store."""
    return f"mlflow ui --backend-store-uri {TRACKING_URI} --port {port}"


def print_ui_hint(port: int = 5000) -> None:
    print("\n  Open the MLflow UI to show this on screen:")
    print(f"    {ui_command(port)}")
    print(f"    http://127.0.0.1:{port}\n")
