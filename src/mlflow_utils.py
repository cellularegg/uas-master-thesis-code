"""Shared MLflow logging interface for training notebooks.

Logs params/metrics only (no model artifacts) to a local SQLite store
(./mlflow.db). Trained models are saved separately to ./models/.
"""

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def set_experiment(name: str) -> None:
    """Set (creating if needed) the active MLflow experiment."""
    raise NotImplementedError


def log_run(params: dict, metrics: dict) -> None:
    """Start an MLflow run and log the given params and metrics dicts."""
    raise NotImplementedError
