"""Shared data preparation and evaluation helpers for joined-data models."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]


@dataclass(frozen=True)
class JoinedFeatureContract:
    """Ordered feature contract for one engineered target station.

    Attributes:
        station_id: Identifier of the station whose targets are predicted.
        target_valid_column: Column marking rows with a complete target horizon.
        predictor_columns: Ordered predictor columns declared by feature metadata.
        target_columns: Ordered direct-forecast target columns.
    """

    station_id: str
    target_valid_column: str
    predictor_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


def _validate_frame_contract(
    frame: pd.DataFrame,
    contract: JoinedFeatureContract,
    *,
    artifact_name: str,
) -> None:
    required_columns = {
        "timestamp",
        contract.target_valid_column,
        *contract.predictor_columns,
        *contract.target_columns,
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{artifact_name} artifact is missing required columns: {missing}"
        )


def load_joined_training_data(
    metadata_path: Path,
    train_path: Path,
    test_path: Path,
    *,
    station_id: str,
    forecast_horizon_hours: int,
) -> tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame]:
    """Load a station contract and its joined train and test artifacts.

    Args:
        metadata_path: Joined-feature metadata JSON path.
        train_path: Joined training-feature Parquet path.
        test_path: Joined sealed-test-feature Parquet path.
        station_id: Target station expected in the metadata.
        forecast_horizon_hours: Required direct-forecast horizon.

    Returns:
        The validated station contract, training frame, and sealed-test frame.

    Raises:
        FileNotFoundError: If the metadata, train, or test artifact is missing.
        ValueError: If metadata or either frame violates the expected contract.
    """
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing joined feature artifact: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("configuration", {}).get("horizon_hours") != forecast_horizon_hours:
        raise ValueError(
            "Feature metadata horizon does not match forecast_horizon_hours"
        )
    if station_id not in metadata.get("engineered_station_ids", []):
        raise ValueError(
            f"Target station {station_id!r} is not engineered in feature metadata"
        )
    expected_targets = tuple(
        f"{station_id}__target_t_plus_{offset:02d}"
        for offset in range(1, forecast_horizon_hours + 1)
    )
    if tuple(metadata.get("target_columns", ())) != expected_targets:
        raise ValueError(
            "Feature metadata target columns do not match the configured horizon"
        )
    predictor_columns = tuple(metadata.get("predictor_columns", ()))
    if not predictor_columns:
        raise ValueError("The metadata predictor contract is empty")
    contract = JoinedFeatureContract(
        station_id=station_id,
        target_valid_column=f"{station_id}__target_valid",
        predictor_columns=predictor_columns,
        target_columns=expected_targets,
    )
    for artifact_path in (train_path, test_path):
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing joined feature artifact: {artifact_path}")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    _validate_frame_contract(train, contract, artifact_name="train")
    _validate_frame_contract(test, contract, artifact_name="test")
    return contract, train, test


def prepare_model_rows(
    frame: pd.DataFrame,
    contract: JoinedFeatureContract,
    *,
    artifact_name: str,
) -> pd.DataFrame:
    """Return the complete common cohort in chronological issue-time order.

    Args:
        frame: Joined feature frame to filter.
        contract: Predictor, target, and eligibility-column contract.
        artifact_name: Human-readable artifact label used in validation errors.

    Returns:
        Eligible rows sorted by timestamp with a fresh integer index.

    Raises:
        ValueError: If required columns are missing, target-valid rows have null
            targets, or no eligible rows remain.
    """
    _validate_frame_contract(frame, contract, artifact_name=artifact_name)
    target_valid_rows = frame[contract.target_valid_column].eq(True)
    if (
        frame.loc[target_valid_rows, list(contract.target_columns)]
        .isna()
        .any(axis=None)
    ):
        raise ValueError(
            f"{artifact_name} artifact has null targets in target-valid rows"
        )
    eligible = (
        target_valid_rows
        & frame[list(contract.predictor_columns)].notna().all(axis=1)
        & frame[list(contract.target_columns)].notna().all(axis=1)
    )
    rows = (
        frame.loc[eligible]
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    if rows.empty:
        raise ValueError(f"{artifact_name} artifact has no eligible model rows")
    return rows


def _feature_parts(column: str) -> tuple[str, str]:
    station_id, separator, base_name = column.partition("__")
    if not separator or not station_id or not base_name:
        raise ValueError(f"Predictor {column!r} must use '<station>__<feature>' format")
    return station_id, base_name


def _is_hydrology_quality_or_time(base_name: str) -> bool:
    return base_name in {"water_level", "imputed"} or base_name.startswith(
        ("water_level_", "imputed_count_", "utc_")
    )


def build_feature_subsets(
    contract: JoinedFeatureContract,
    *,
    weather_variables: Sequence[str],
) -> dict[str, list[str]]:
    """Build six ordered ablation subsets from a joined predictor contract.

    Args:
        contract: Ordered joined-feature contract for the target station.
        weather_variables: Base names treated as raw weather predictors.

    Returns:
        The six named feature subsets in contract order.

    Raises:
        ValueError: If predictors are duplicated or malformed, or a required
            subset is empty.
    """
    if len(contract.predictor_columns) != len(set(contract.predictor_columns)):
        raise ValueError("The predictor contract contains duplicates")
    parsed_columns = [
        (column, *_feature_parts(column)) for column in contract.predictor_columns
    ]
    raw_names = {"water_level", "imputed", *weather_variables}
    subsets = {
        "full": list(contract.predictor_columns),
        "all_station_hydrology_quality_time": [
            column
            for column, _station_id, base_name in parsed_columns
            if _is_hydrology_quality_or_time(base_name)
        ],
        "raw_all_stations": [
            column
            for column, _station_id, base_name in parsed_columns
            if base_name in raw_names
        ],
        "target_station_full": [
            column
            for column, station_id, _base_name in parsed_columns
            if station_id == contract.station_id
        ],
        "target_station_hydrology_quality_time": [
            column
            for column, station_id, base_name in parsed_columns
            if station_id == contract.station_id
            and _is_hydrology_quality_or_time(base_name)
        ],
        "current_water_levels_all_stations": [
            column
            for column, _station_id, base_name in parsed_columns
            if base_name == "water_level"
        ],
    }
    for subset_name, columns in subsets.items():
        if not columns:
            raise ValueError(f"Feature subset {subset_name!r} is empty")
    return subsets


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


def time_series_splits(
    n_rows: int,
    *,
    initial_train_fraction: float,
    n_validation_folds: int,
    embargo_rows: int,
) -> tuple[TimeSeriesSplit, list[tuple[np.ndarray, np.ndarray]], int]:
    """Build expanding chronological validation folds with an embargo.

    Args:
        n_rows: Number of eligible chronological training rows.
        initial_train_fraction: Fraction reserved before validation allocation.
        n_validation_folds: Number of expanding validation folds.
        embargo_rows: Rows excluded between each training and validation window.

    Returns:
        The configured splitter, materialized folds, and validation-window size.

    Raises:
        ValueError: If there are too few rows or the generated folds violate the
            configured count, chronology, embargo, or non-overlap invariants.
    """
    initial_train_rows = int(n_rows * initial_train_fraction)
    validation_budget = n_rows - initial_train_rows - embargo_rows
    validation_test_size = validation_budget // n_validation_folds
    if validation_test_size < 1:
        raise ValueError(
            "Not enough eligible training rows for the configured CV policy"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_validation_folds,
        gap=embargo_rows,
        test_size=validation_test_size,
    )
    splits = list(splitter.split(np.arange(n_rows)))
    if len(splits) != n_validation_folds:
        raise ValueError(
            f"Expected {n_validation_folds} validation folds, got {len(splits)}"
        )

    previous_validation_end = -1
    for fold_number, (fold_train_indices, fold_validation_indices) in enumerate(
        splits, start=1
    ):
        if fold_train_indices.size == 0 or fold_validation_indices.size == 0:
            raise ValueError(f"Fold {fold_number} is empty")
        if not np.array_equal(fold_train_indices, np.arange(fold_train_indices.size)):
            raise ValueError(f"Fold {fold_number} training rows are not chronological")
        if fold_validation_indices[0] - fold_train_indices[-1] - 1 != embargo_rows:
            raise ValueError(f"Fold {fold_number} does not have the configured embargo")
        if fold_validation_indices[0] <= previous_validation_end:
            raise ValueError("Validation folds overlap or are out of order")
        previous_validation_end = fold_validation_indices[-1]
    return splitter, splits, validation_test_size


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


def sha256_file(path: Path) -> str:
    """Compute the chunked SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        The hex-encoded SHA-256 digest, used for MLflow provenance params.
    """
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
