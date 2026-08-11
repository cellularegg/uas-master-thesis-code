"""Build leakage-safe predictors and multi-step targets for station artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from src.config import FORECAST_HORIZON_HOURS, WEATHER_VARIABLES

WATER_LEVEL_CHANGE_HOURS = (1, 3, 6, 12, 24)
PRECIPITATION_VARIABLES = ("precipitation",)
TEMPERATURE_VARIABLE = "temperature_2m"


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration shared by feature calculation and artifact metadata."""

    horizon_hours: int = FORECAST_HORIZON_HOURS
    calendar_timezone: str = "UTC"
    lag_hours: tuple[int, ...] = (1, 3, 6, 12, 24, 48, 72, 168)
    rolling_windows: tuple[int, ...] = (6, 24, 72, 168)

    def __post_init__(self) -> None:
        """Reject values that cannot define the stage-03 feature contract."""
        if self.horizon_hours < 1:
            raise ValueError("horizon_hours must be at least 1")
        if self.calendar_timezone != "UTC":
            raise ValueError("calendar_timezone must be UTC")
        for name, values in (
            ("lag_hours", self.lag_hours),
            ("rolling_windows", self.rolling_windows),
        ):
            if not values or any(value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing and unique")


DEFAULT_FEATURE_CONFIG = FeatureConfig()


def feature_column_names(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> tuple[str, ...]:
    """Return predictor names in their stable downstream order."""
    names = ["water_level", "imputed", *WEATHER_VARIABLES]
    names.extend(f"water_level_lag_{hours}h" for hours in config.lag_hours)
    names.extend(f"water_level_change_{hours}h" for hours in WATER_LEVEL_CHANGE_HOURS)
    names.extend(
        f"water_level_rolling_{statistic}_{window}h"
        for statistic in ("mean", "std", "min", "max")
        for window in config.rolling_windows
    )
    names.extend(f"imputed_count_{window}h" for window in config.rolling_windows)
    names.extend(
        f"{variable}_rolling_sum_{window}h"
        for variable in PRECIPITATION_VARIABLES
        for window in config.rolling_windows
    )
    names.extend(
        f"{TEMPERATURE_VARIABLE}_rolling_mean_{window}h"
        for window in config.rolling_windows
    )
    names.extend(
        (
            f"{TEMPERATURE_VARIABLE}_rolling_min_24h",
            f"{TEMPERATURE_VARIABLE}_rolling_max_24h",
        )
    )
    names.extend(
        (
            "utc_hour_sin",
            "utc_hour_cos",
            "utc_day_of_week_sin",
            "utc_day_of_week_cos",
            "utc_day_of_year_sin",
            "utc_day_of_year_cos",
        )
    )
    return tuple(names)


def target_column_names(
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> tuple[str, ...]:
    """Return ordered target names from t+1 through the configured horizon."""
    width = max(2, len(str(config.horizon_hours)))
    return tuple(
        f"target_t_plus_{offset:0{width}d}"
        for offset in range(1, config.horizon_hours + 1)
    )


def calculate_target_eligibility(
    frame: pd.DataFrame, *, horizon_hours: int = FORECAST_HORIZON_HOURS
) -> pd.Series:
    """Identify issue times whose complete future target window is observed.

    Args:
        frame: Chronologically ordered observations containing ``water_level`` and
            ``imputed``.
        horizon_hours: Number of future hourly targets required per issue time.

    Returns:
        A Boolean Series aligned with ``frame.index``.

    Raises:
        ValueError: If the horizon is not positive.
    """
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least 1")
    observed = frame["water_level"].notna() & ~frame["imputed"]
    future_observed = observed.shift(-1)
    valid_count = (
        future_observed.iloc[::-1]
        .rolling(horizon_hours, min_periods=horizon_hours)
        .sum()
        .iloc[::-1]
    )
    return valid_count.eq(horizon_hours).astype(bool).rename("target_valid")


def _validate_feature_input(
    frame: pd.DataFrame, *, station_id: str, config: FeatureConfig
) -> None:
    required = {
        "timestamp",
        "water_level",
        "imputed",
        "station_id",
        *WEATHER_VARIABLES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("station timeline is empty")

    timestamp_dtype = frame["timestamp"].dtype
    if (
        not isinstance(timestamp_dtype, pd.DatetimeTZDtype)
        or str(timestamp_dtype.tz) != "UTC"
    ):
        raise ValueError("timestamp must be timezone-aware UTC")
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if timestamps.has_duplicates:
        raise ValueError("timestamps must be unique")
    expected_grid = pd.date_range(timestamps[0], periods=len(frame), freq="h")
    if not timestamps.equals(expected_grid):
        raise ValueError(
            "timestamps must be in ascending order on a contiguous hourly grid"
        )

    station_ids = frame["station_id"].drop_duplicates().tolist()
    if station_ids != [station_id]:
        raise ValueError(
            f"frame must contain one station matching {station_id!r}; got {station_ids!r}"
        )
    if not is_bool_dtype(frame["imputed"].dtype) or frame["imputed"].isna().any():
        raise ValueError("imputed must be a non-null Boolean column")


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    station_id: str,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> pd.DataFrame:
    """Transform one physical station artifact without reading outside its rows.

    Rolling predictors include issue time ``t``, require a complete lookback, and
    retain missing values. Targets are future water levels at ``t+1`` through the
    configured horizon; the full vector is blanked for ineligible issue times.

    Args:
        frame: One train or sealed-test station artifact.
        station_id: Expected identifier for every row.
        config: Feature and target configuration.

    Returns:
        A copy containing every original row and column, target eligibility,
        engineered predictors, and the ordered target vector.

    Raises:
        ValueError: If the physical artifact violates the input contract.
    """
    _validate_feature_input(frame, station_id=station_id, config=config)
    result = frame.copy(deep=True)
    water_level = frame["water_level"]
    valid_targets = calculate_target_eligibility(
        frame, horizon_hours=config.horizon_hours
    )
    result["target_valid"] = valid_targets

    for hours in config.lag_hours:
        result[f"water_level_lag_{hours}h"] = water_level.shift(hours)
    for hours in WATER_LEVEL_CHANGE_HOURS:
        result[f"water_level_change_{hours}h"] = water_level - water_level.shift(hours)

    for statistic in ("mean", "std", "min", "max"):
        for window in config.rolling_windows:
            rolling = water_level.rolling(window, min_periods=window)
            if statistic == "std":
                values = rolling.std(ddof=0)
            else:
                values = getattr(rolling, statistic)()
            result[f"water_level_rolling_{statistic}_{window}h"] = values

    for window in config.rolling_windows:
        result[f"imputed_count_{window}h"] = (
            frame["imputed"].rolling(window, min_periods=window).sum()
        )

    for variable in PRECIPITATION_VARIABLES:
        for window in config.rolling_windows:
            result[f"{variable}_rolling_sum_{window}h"] = (
                frame[variable].rolling(window, min_periods=window).sum()
            )

    temperature = frame[TEMPERATURE_VARIABLE]
    for window in config.rolling_windows:
        result[f"{TEMPERATURE_VARIABLE}_rolling_mean_{window}h"] = temperature.rolling(
            window, min_periods=window
        ).mean()
    result[f"{TEMPERATURE_VARIABLE}_rolling_min_24h"] = temperature.rolling(
        24, min_periods=24
    ).min()
    result[f"{TEMPERATURE_VARIABLE}_rolling_max_24h"] = temperature.rolling(
        24, min_periods=24
    ).max()

    timestamps = frame["timestamp"].dt.tz_convert(config.calendar_timezone)
    hour_angle = 2.0 * math.pi * timestamps.dt.hour / 24.0
    weekday_angle = 2.0 * math.pi * timestamps.dt.dayofweek / 7.0
    days_in_year = np.where(timestamps.dt.is_leap_year, 366.0, 365.0)
    year_angle = 2.0 * math.pi * (timestamps.dt.dayofyear - 1) / days_in_year
    result["utc_hour_sin"] = np.sin(hour_angle)
    result["utc_hour_cos"] = np.cos(hour_angle)
    result["utc_day_of_week_sin"] = np.sin(weekday_angle)
    result["utc_day_of_week_cos"] = np.cos(weekday_angle)
    result["utc_day_of_year_sin"] = np.sin(year_angle)
    result["utc_day_of_year_cos"] = np.cos(year_angle)

    for offset, column in enumerate(target_column_names(config), start=1):
        result[column] = water_level.shift(-offset).where(valid_targets)

    return result


def extract_station_frame(joined: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Select one station's own columns from a joined frame, unprefixed, usable rows only.

    Args:
        joined: A wide frame produced by :func:`src.preprocess.join_station_frames`
            (or its feature-engineered counterpart), prefixed as ``<station_id>__<column>``.
        station_id: Station whose columns to select.

    Returns:
        A frame with ``timestamp`` plus every one of the station's own columns,
        unprefixed, restricted to rows where the station has data.

    Raises:
        ValueError: If the station has no ``station_id`` column in ``joined``.
    """
    prefix = f"{station_id}__"
    prefixed_columns = [
        column for column in joined.columns if column.startswith(prefix)
    ]
    station_id_column = f"{prefix}station_id"
    if station_id_column not in prefixed_columns:
        raise ValueError(f"{station_id} is missing joined column: {station_id_column}")

    available = joined[station_id_column].notna()
    station = (
        joined.loc[available, ["timestamp", *prefixed_columns]]
        .rename(columns={column: column[len(prefix) :] for column in prefixed_columns})
        .reset_index(drop=True)
    )
    if "imputed" in station.columns:
        # Joined null regions make this column object-typed; usable rows are Boolean.
        station["imputed"] = station["imputed"].astype(bool)
    return station


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), Path.cwd().resolve())


def _utc_text(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _frame_profile(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(path),
        "sha256": _sha256(path),
        "rows": len(frame),
        "schema": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "timestamp_range": {
            "start_utc": _utc_text(frame["timestamp"].iloc[0]),
            "end_utc": _utc_text(frame["timestamp"].iloc[-1]),
        },
        "null_counts": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
    }


def _validate_output_against_source(
    features: pd.DataFrame,
    source: pd.DataFrame,
    *,
    label: str,
    station_id: str,
    config: FeatureConfig,
) -> None:
    _validate_feature_input(source, station_id=station_id, config=config)
    missing = sorted(
        set(feature_column_names(config) + target_column_names(config)).difference(
            features.columns
        )
    )
    if missing:
        raise ValueError(f"{label} features are missing contract columns: {missing}")
    if len(features) != len(source) or not features["timestamp"].equals(
        source["timestamp"]
    ):
        raise ValueError(f"{label} features do not preserve source rows and timestamps")
    for column in source.columns:
        if not features[column].equals(source[column]):
            raise ValueError(f"{label} features changed original column {column!r}")
    expected_valid = calculate_target_eligibility(
        source, horizon_hours=config.horizon_hours
    )
    if not features["target_valid"].equals(expected_valid):
        raise ValueError(f"{label} features have incorrect target eligibility")


def write_feature_artifacts(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    station_id: str,
    train_source_path: Path,
    test_source_path: Path,
    output_dir: Path,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> dict[str, Any]:
    """Replace station feature Parquets and write their lineage manifest.

    Args:
        train_features: Independently calculated train feature frame.
        test_features: Independently calculated sealed-test feature frame.
        station_id: Identifier used in artifact paths and metadata.
        train_source_path: Stage-02 train Parquet.
        test_source_path: Stage-02 sealed-test Parquet.
        output_dir: Destination directory for the feature artifacts.
        config: Feature and target configuration.

    Returns:
        The manifest dictionary written to JSON.

    Raises:
        FileNotFoundError: If either source artifact does not exist.
        ValueError: If feature rows or preserved source columns differ.
    """
    for source_path in (train_source_path, test_source_path):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    train_source = pd.read_parquet(train_source_path)
    test_source = pd.read_parquet(test_source_path)
    _validate_output_against_source(
        train_features,
        train_source,
        label="train",
        station_id=station_id,
        config=config,
    )
    _validate_output_against_source(
        test_features,
        test_source,
        label="test",
        station_id=station_id,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{station_id}_train_features.parquet"
    test_path = output_dir / f"{station_id}_test_features.parquet"
    metadata_path = output_dir / f"{station_id}_feature_metadata.json"
    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)

    generator_path = Path(__file__)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "station_id": station_id,
        "configuration": {
            "horizon_hours": config.horizon_hours,
            "calendar_timezone": config.calendar_timezone,
            "lag_hours": list(config.lag_hours),
            "rolling_windows": list(config.rolling_windows),
        },
        "predictor_columns": list(feature_column_names(config)),
        "target_columns": list(target_column_names(config)),
        "semantics": {
            "issue_time": "Predictors use information available through timestamp t.",
            "rolling": "Trailing windows include t, require every source value, and never read future rows.",
            "target": "water_level at t+1 through t+horizon_hours; the full vector is null unless target_valid is true.",
            "physical_independence": "Train and sealed-test features are calculated separately; unavailable lookback rows remain null.",
        },
        "inputs": {
            "train": _frame_profile(train_source, train_source_path),
            "test": _frame_profile(test_source, test_source_path),
        },
        "artifacts": {
            "train_features": _frame_profile(train_features, train_path),
            "test_features": _frame_profile(test_features, test_path),
        },
        "generator": {
            "module": _relative_path(generator_path),
            "sha256": _sha256(generator_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": version("pyarrow"),
            "platform": sys.platform,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _validate_joined_output_against_source(
    features: pd.DataFrame, source: pd.DataFrame, *, label: str
) -> None:
    if len(features) != len(source):
        raise ValueError(f"{label} features do not preserve source rows")
    if features.columns[: len(source.columns)].tolist() != source.columns.tolist():
        raise ValueError(f"{label} features do not preserve source columns")
    for column in source.columns:
        if not features[column].equals(source[column]):
            raise ValueError(f"{label} features changed original column {column!r}")


def _joined_predictor_columns(
    station_ids: Sequence[str],
    engineered_station_ids: Sequence[str],
    config: FeatureConfig,
) -> list[str]:
    """Return ordered joined predictors for the requested station scope."""
    raw_predictors = ("water_level", "imputed", *WEATHER_VARIABLES)
    engineered = set(engineered_station_ids)
    return [
        f"{station_id}__{column}"
        for station_id in station_ids
        for column in (
            feature_column_names(config) if station_id in engineered else raw_predictors
        )
    ]


def write_joined_feature_artifacts(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    station_ids: Sequence[str],
    engineered_station_ids: Sequence[str],
    train_source_path: Path,
    test_source_path: Path,
    output_dir: Path,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> dict[str, Any]:
    """Replace the joined feature Parquets and write their lineage manifest.

    Args:
        train_features: Joined train partition engineered per-station.
        test_features: Joined sealed-test partition engineered per-station.
        station_ids: Every station represented in the joined frames.
        engineered_station_ids: Stations receiving the full feature and target
            contract. Other stations remain raw joined inputs.
        train_source_path: Stage-02 joined train Parquet.
        test_source_path: Stage-02 joined sealed-test Parquet.
        output_dir: Destination directory for the feature artifacts.
        config: Feature and target configuration.

    Returns:
        The manifest dictionary written to JSON.

    Raises:
        FileNotFoundError: If either source artifact does not exist.
        ValueError: If feature rows do not preserve the joined source's columns
            or values, or if the engineered station scope is invalid.
    """
    station_ids = list(station_ids)
    engineered_station_ids = list(engineered_station_ids)
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("station_ids must be unique")
    if len(engineered_station_ids) != len(set(engineered_station_ids)):
        raise ValueError("engineered_station_ids must be unique")
    missing_engineered = sorted(set(engineered_station_ids).difference(station_ids))
    if missing_engineered:
        raise ValueError(
            "engineered_station_ids must be included in station_ids: "
            f"{missing_engineered}"
        )
    for source_path in (train_source_path, test_source_path):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    train_source = pd.read_parquet(train_source_path)
    test_source = pd.read_parquet(test_source_path)
    _validate_joined_output_against_source(train_features, train_source, label="train")
    _validate_joined_output_against_source(test_features, test_source, label="test")

    raw_predictors = {"water_level", "imputed", *WEATHER_VARIABLES}
    derived_columns = {
        *set(feature_column_names(config)).difference(raw_predictors),
        "target_valid",
        *target_column_names(config),
    }
    for station_id in station_ids:
        if station_id in engineered_station_ids:
            continue
        forbidden = [
            f"{station_id}__{column}"
            for column in derived_columns
            if f"{station_id}__{column}" in train_features.columns
            or f"{station_id}__{column}" in test_features.columns
        ]
        if forbidden:
            raise ValueError(
                f"non-engineered stations must not contain derived columns: {forbidden}"
            )
    predictor_columns = _joined_predictor_columns(
        station_ids, engineered_station_ids, config
    )
    target_columns = [
        f"{station_id}__{column}"
        for station_id in engineered_station_ids
        for column in target_column_names(config)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "all_stations_train_features.parquet"
    test_path = output_dir / "all_stations_test_features.parquet"
    metadata_path = output_dir / "all_stations_feature_metadata.json"
    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)

    generator_path = Path(__file__)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "station_ids": station_ids,
        "engineered_station_ids": engineered_station_ids,
        "configuration": {
            "horizon_hours": config.horizon_hours,
            "calendar_timezone": config.calendar_timezone,
            "lag_hours": list(config.lag_hours),
            "rolling_windows": list(config.rolling_windows),
        },
        "predictor_columns": predictor_columns,
        "target_columns": target_columns,
        "semantics": {
            "issue_time": "Predictors use information available through timestamp t.",
            "rolling": "Trailing windows include t, require every source value, and never read future rows.",
            "target": "water_level at t+1 through t+horizon_hours; the full vector is null unless target_valid is true.",
            "physical_independence": "Train and sealed-test features are calculated separately; unavailable lookback rows remain null.",
            "station_scope": "All joined station_ids are preserved. Full predictors, target_valid, and target vectors are calculated only for engineered_station_ids; other stations contribute raw water_level, imputed, and weather predictors, while their station_id columns remain raw metadata only.",
        },
        "inputs": {
            "train": _frame_profile(train_source, train_source_path),
            "test": _frame_profile(test_source, test_source_path),
        },
        "artifacts": {
            "train": _frame_profile(train_features, train_path),
            "test": _frame_profile(test_features, test_path),
        },
        "generator": {
            "module": _relative_path(generator_path),
            "sha256": _sha256(generator_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": version("pyarrow"),
            "platform": sys.platform,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
