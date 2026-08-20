"""Reusable metric calculations for multi-output forecasts."""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.config import WATER_LEVEL_ALARM_THRESHOLD_CM


def metric_tables(
    actual: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_columns: Sequence[str],
    station_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate aggregate and horizon-specific forecast metrics.

    Args:
        actual: Actual target values, with one column per forecast horizon.
        predictions: Predicted values in the same row and horizon order.
        target_columns: Ordered target column names corresponding to predictions.
        station_id: Identifier included in the returned metric tables.

    Returns:
        A tuple containing aggregate metrics and one row per forecast horizon.
        ME is prediction minus actual, and aggregate R² is calculated after
        flattening all horizons together.
    """
    target_columns = list(target_columns)
    if not target_columns:
        raise ValueError("At least one target column is required")
    missing_columns = sorted(set(target_columns).difference(actual.columns))
    if missing_columns:
        raise ValueError(f"Actual values are missing target columns: {missing_columns}")

    actual_values = actual[target_columns].to_numpy()
    if predictions.shape != actual_values.shape:
        raise ValueError(
            "Prediction shape must match actual target shape: "
            f"{predictions.shape} != {actual_values.shape}"
        )

    aggregate = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "scored_issue_times": len(actual),
                "scored_values": actual_values.size,
                "mae": mean_absolute_error(actual_values.ravel(), predictions.ravel()),
                "rmse": root_mean_squared_error(
                    actual_values.ravel(), predictions.ravel()
                ),
                "me": float(np.mean(predictions.ravel() - actual_values.ravel())),
                "r2": r2_score(actual_values.ravel(), predictions.ravel()),
            }
        ]
    )
    per_horizon = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "horizon_hours": horizon,
                "target": target,
                "mae": mean_absolute_error(
                    actual_values[:, horizon - 1], predictions[:, horizon - 1]
                ),
                "rmse": root_mean_squared_error(
                    actual_values[:, horizon - 1], predictions[:, horizon - 1]
                ),
                "me": float(
                    np.mean(predictions[:, horizon - 1] - actual_values[:, horizon - 1])
                ),
                "r2": r2_score(
                    actual_values[:, horizon - 1], predictions[:, horizon - 1]
                ),
            }
            for horizon, target in enumerate(target_columns, start=1)
        ]
    )
    return aggregate, per_horizon


def water_level_regime_tables(
    actual: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_columns: Sequence[str],
    station_id: str,
    quartile_cutoffs_cm: Sequence[float],
    alarm_threshold_cm: float = WATER_LEVEL_ALARM_THRESHOLD_CM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive metrics by actual water-level regime.

    Regimes are assigned independently to every scored forecast value using
    the actual value.  The four quartile regimes are Q1 (``actual <= Q25``),
    Q2 (``Q25 < actual <= Q50``), Q3 (``Q50 < actual <= Q75``), and Q4
    (``actual > Q75``).  Alarm values (``actual >= alarm_threshold_cm``) are
    scored in a separate overlapping regime.  Thus an alarm value can also be
    present in Q4.

    This is a post-hoc diagnostic only: the supplied cutoffs must have been
    calculated from the training reference population by the caller, and no
    model-selection metric is calculated here.

    Args:
        actual: Actual target values, with one column per forecast horizon.
        predictions: Predicted values in the same row and horizon order.
        target_columns: Ordered target column names corresponding to predictions.
        station_id: Identifier included in the returned tables.
        quartile_cutoffs_cm: Ordered Q25, Q50, and Q75 training cutoffs in cm.
        alarm_threshold_cm: Inclusive threshold for the overlapping alarm regime.

    Returns:
        A pair of aggregate and per-horizon tables.  Each table retains all
        five regimes even when no scored values are available.  Empty groups
        have a zero ``scored_values`` count and null MAE, RMSE, and ME values.
        The aggregate table uses a nullable ``horizon_hours`` column.

    Raises:
        ValueError: If target columns, prediction shape, cutoffs, or the alarm
            threshold are invalid.
    """
    target_columns = list(target_columns)
    missing_columns = sorted(set(target_columns).difference(actual.columns))
    if missing_columns:
        raise ValueError(f"Actual values are missing target columns: {missing_columns}")
    if len(target_columns) != len(set(target_columns)):
        raise ValueError("Target columns must be unique")

    actual_values = actual[target_columns].to_numpy(dtype=float)
    prediction_values = np.asarray(predictions, dtype=float)
    if prediction_values.shape != actual_values.shape:
        raise ValueError(
            "Prediction shape must match actual target shape: "
            f"{prediction_values.shape} != {actual_values.shape}"
        )
    if prediction_values.ndim != 2:
        raise ValueError("Predictions must be a two-dimensional array")

    try:
        cutoffs = tuple(float(cutoff) for cutoff in quartile_cutoffs_cm)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Quartile cutoffs must contain three finite numbers"
        ) from error
    if len(cutoffs) != 3 or not all(np.isfinite(cutoff) for cutoff in cutoffs):
        raise ValueError("Quartile cutoffs must contain three finite numbers")
    if any(lower > upper for lower, upper in pairwise(cutoffs)):
        raise ValueError("Quartile cutoffs must be ordered Q25 <= Q50 <= Q75")

    try:
        alarm_threshold = float(alarm_threshold_cm)
    except (TypeError, ValueError) as error:
        raise ValueError("Alarm threshold must be a finite number") from error
    if not np.isfinite(alarm_threshold):
        raise ValueError("Alarm threshold must be a finite number")

    q25, q50, q75 = cutoffs
    regime_definitions: tuple[tuple[str, float | None, float | None], ...] = (
        ("Q1", None, q25),
        ("Q2", q25, q50),
        ("Q3", q50, q75),
        ("Q4", q75, None),
        ("Alarm", alarm_threshold, None),
    )

    def regime_mask(values: np.ndarray, regime: str) -> np.ndarray:
        finite = np.isfinite(values)
        if regime == "Q1":
            return finite & (values <= q25)
        if regime == "Q2":
            return finite & (values > q25) & (values <= q50)
        if regime == "Q3":
            return finite & (values > q50) & (values <= q75)
        if regime == "Q4":
            return finite & (values > q75)
        if regime == "Alarm":
            return finite & (values >= alarm_threshold)
        raise AssertionError(f"Unknown water-level regime: {regime}")

    def metric_values(
        actual_subset: np.ndarray,
        prediction_subset: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, float, float, float]:
        scored = mask & np.isfinite(actual_subset) & np.isfinite(prediction_subset)
        actual_scored = actual_subset[scored]
        prediction_scored = prediction_subset[scored]
        scored_values = int(actual_scored.size)
        if scored_values == 0:
            return scored_values, np.nan, np.nan, np.nan
        errors = prediction_scored - actual_scored
        return (
            scored_values,
            float(np.mean(np.abs(errors))),
            float(np.sqrt(np.mean(np.square(errors)))),
            float(np.mean(errors)),
        )

    aggregate_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for regime, lower_bound, upper_bound in regime_definitions:
        aggregate_mask = np.zeros(actual_values.shape, dtype=bool)
        for horizon in range(actual_values.shape[1]):
            aggregate_mask[:, horizon] = regime_mask(actual_values[:, horizon], regime)
        scored_values, mae, rmse, me = metric_values(
            actual_values, prediction_values, aggregate_mask
        )
        aggregate_rows.append(
            {
                "station_id": station_id,
                "regime": regime,
                "lower_bound_cm": lower_bound,
                "upper_bound_cm": upper_bound,
                "horizon_hours": None,
                "scored_values": scored_values,
                "mae": mae,
                "rmse": rmse,
                "me": me,
            }
        )
        for horizon in range(actual_values.shape[1]):
            scored_values, mae, rmse, me = metric_values(
                actual_values[:, horizon],
                prediction_values[:, horizon],
                regime_mask(actual_values[:, horizon], regime),
            )
            horizon_rows.append(
                {
                    "station_id": station_id,
                    "regime": regime,
                    "lower_bound_cm": lower_bound,
                    "upper_bound_cm": upper_bound,
                    "horizon_hours": horizon + 1,
                    "scored_values": scored_values,
                    "mae": mae,
                    "rmse": rmse,
                    "me": me,
                }
            )

    columns = [
        "station_id",
        "regime",
        "lower_bound_cm",
        "upper_bound_cm",
        "horizon_hours",
        "scored_values",
        "mae",
        "rmse",
        "me",
    ]
    aggregate = pd.DataFrame(aggregate_rows, columns=columns)
    per_horizon = pd.DataFrame(horizon_rows, columns=columns)
    aggregate["horizon_hours"] = aggregate["horizon_hours"].astype("Int64")
    per_horizon["horizon_hours"] = per_horizon["horizon_hours"].astype("Int64")
    aggregate["scored_values"] = aggregate["scored_values"].astype("Int64")
    per_horizon["scored_values"] = per_horizon["scored_values"].astype("Int64")
    return aggregate, per_horizon
