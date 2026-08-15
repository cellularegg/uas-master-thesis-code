"""Reusable metric calculations for multi-output forecasts."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


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
