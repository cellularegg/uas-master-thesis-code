"""Shared MLflow logging interface for training notebooks.

Logs params/metrics only (no model artifacts) to the local file store
in ./mlruns/. Trained models are saved separately to ./models/.
"""


def set_experiment(name: str) -> None:
    """Set (creating if needed) the active MLflow experiment."""
    raise NotImplementedError


def log_run(params: dict, metrics: dict) -> None:
    """Start an MLflow run and log the given params and metrics dicts."""
    raise NotImplementedError
