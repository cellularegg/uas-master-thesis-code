"""Shared fit-time and score-time helpers for joined-data models."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def numeric_predictors(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Convert selected predictors to numeric floating-point model inputs.

    Args:
        frame: Frame containing the selected predictors.
        feature_columns: Ordered predictor columns to convert.

    Returns:
        A floating-point frame containing only the selected predictors.

    Raises:
        KeyError: If a selected predictor is missing from the frame.
        ValueError: If a selected value cannot be converted to a number.
    """
    return (
        frame[list(feature_columns)].apply(pd.to_numeric, errors="raise").astype(float)
    )


def validate_predictions(
    predictions: np.ndarray | Sequence[Sequence[float]],
    *,
    expected_rows: int,
    target_columns: Sequence[str],
    artifact_name: str,
) -> np.ndarray:
    """Return predictions as floats after shape and finiteness checks.

    Args:
        predictions: Two-dimensional prediction values to validate.
        expected_rows: Required number of prediction rows.
        target_columns: Ordered targets defining the required column count.
        artifact_name: Human-readable prediction label used in errors.

    Returns:
        A finite floating-point prediction array with the expected shape.

    Raises:
        ValueError: If conversion fails or predictions have the wrong shape or
            contain non-finite values.
    """
    values = np.asarray(predictions, dtype=float)
    expected_shape = (expected_rows, len(target_columns))
    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected {artifact_name} prediction shape: "
            f"{values.shape} != {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{artifact_name} predictions contain non-finite values")
    return values


def summarize_cv_metrics(
    fold_aggregate_metrics: pd.DataFrame,
    fold_horizon_metrics: pd.DataFrame,
) -> dict[str, float]:
    """Summarize aggregate and per-horizon fold metrics for one candidate.

    Args:
        fold_aggregate_metrics: One aggregate MAE, RMSE, ME, and R² row per fold.
        fold_horizon_metrics: Per-fold metric rows keyed by horizon.

    Returns:
        Mean and population-standard-deviation metrics for the candidate.
    """
    summary = {
        "cv_mae_mean": float(fold_aggregate_metrics["mae"].mean()),
        "cv_mae_std": float(fold_aggregate_metrics["mae"].std(ddof=0)),
        "cv_rmse_mean": float(fold_aggregate_metrics["rmse"].mean()),
        "cv_rmse_std": float(fold_aggregate_metrics["rmse"].std(ddof=0)),
        "cv_me_mean": float(fold_aggregate_metrics["me"].mean()),
        "cv_me_std": float(fold_aggregate_metrics["me"].std(ddof=0)),
        "cv_r2_mean": float(fold_aggregate_metrics["r2"].mean()),
        "cv_r2_std": float(fold_aggregate_metrics["r2"].std(ddof=0)),
    }
    for horizon, horizon_metrics in fold_horizon_metrics.groupby("horizon_hours"):
        horizon_number = int(str(horizon))
        for metric in ("mae", "rmse", "me", "r2"):
            summary[f"cv_{metric}_horizon_{horizon_number:02d}_mean"] = float(
                horizon_metrics[metric].mean()
            )
            summary[f"cv_{metric}_horizon_{horizon_number:02d}_std"] = float(
                horizon_metrics[metric].std(ddof=0)
            )
    return summary


def prediction_preview(
    frame: pd.DataFrame,
    predictions: np.ndarray | Sequence[Sequence[float]],
    *,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Combine issue timestamps, actual targets, and direct predictions.

    Args:
        frame: Frame containing timestamps and actual targets.
        predictions: Direct predictions ordered like ``target_columns``.
        target_columns: Ordered actual target columns.

    Returns:
        Timestamps, actual targets, and correspondingly named predictions.

    Raises:
        ValueError: If predictions cannot be converted or fail validation.
    """
    target_columns = list(target_columns)
    prediction_values = validate_predictions(
        predictions,
        expected_rows=len(frame),
        target_columns=target_columns,
        artifact_name="preview",
    )
    predicted = pd.DataFrame(
        prediction_values,
        columns=[f"prediction_{target}" for target in target_columns],
        index=frame.index,
    )
    return pd.concat([frame[["timestamp", *target_columns]], predicted], axis=1)


def mlflow_run_series(runs: pd.DataFrame, column: str) -> pd.Series:
    """Return a run column, or a null series when MLflow has no such field.

    Args:
        runs: MLflow run rows as returned by ``mlflow.search_runs``.
        column: Column name to read.

    Returns:
        The requested column, or an all-null object series with the same
        index when the column is absent.
    """
    if column in runs.columns:
        return runs[column]
    return pd.Series(pd.NA, index=runs.index, dtype="object")


def mlflow_finite_float(value: object) -> float | None:
    """Convert an MLflow value to a finite float, if possible.

    Args:
        value: Raw MLflow tag, param, or metric value.

    Returns:
        The finite float value, or ``None`` if it is missing or not finite.
    """
    if value is None or pd.isna(value):  # type: ignore[call-overload]
        return None
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
