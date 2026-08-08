"""Create leakage-safe chronological train, validation, and test splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    FORECAST_HORIZON_HOURS,
    INITIAL_TRAIN_FRACTION,
    N_CV_FOLDS,
    TEST_FRACTION,
)

REQUIRED_COLUMNS = frozenset({"timestamp", "water_level", "imputed", "station_id"})
ROLE_CATEGORIES = ["train", "embargo", "validation", "future"]


@dataclass(frozen=True)
class SplitConfig:
    """Configuration shared by splitting, fold assignment, and metadata."""

    test_fraction: float = TEST_FRACTION
    n_folds: int = N_CV_FOLDS
    initial_train_fraction: float = INITIAL_TRAIN_FRACTION
    horizon_hours: int = FORECAST_HORIZON_HOURS

    def __post_init__(self) -> None:
        """Reject values that cannot describe a chronological split."""
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least 1")
        if not 0.0 < self.initial_train_fraction < 1.0:
            raise ValueError("initial_train_fraction must be between 0 and 1")
        if self.horizon_hours < 1:
            raise ValueError("horizon_hours must be at least 1")


DEFAULT_SPLIT_CONFIG = SplitConfig()


@dataclass(frozen=True)
class StationSplit:
    """The physical station split and its cross-validation layout."""

    train: pd.DataFrame
    test: pd.DataFrame
    validation_block_sizes: tuple[int, ...]
    config: SplitConfig

    @property
    def role_columns(self) -> tuple[str, ...]:
        """Return the ordered fold-role column names."""
        return tuple(
            f"fold_{number:02d}_role"
            for number in range(1, len(self.validation_block_sizes) + 1)
        )


def validate_station_frame(frame: pd.DataFrame, *, station_id: str) -> pd.DataFrame:
    """Return a sorted station frame after enforcing the strict input contract.

    Args:
        frame: Candidate station observations.
        station_id: Expected station identifier.

    Returns:
        A timestamp-sorted copy with a fresh row index.

    Raises:
        ValueError: If columns, timestamps, or station identifiers are invalid.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
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

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    if ordered["timestamp"].duplicated().any():
        raise ValueError("timestamps must be unique")
    expected_grid = pd.date_range(
        ordered["timestamp"].iloc[0], periods=len(ordered), freq="h"
    )
    if not pd.DatetimeIndex(ordered["timestamp"]).equals(expected_grid):
        raise ValueError("timestamps must form a contiguous hourly grid")

    station_ids = ordered["station_id"].drop_duplicates().tolist()
    if station_ids != [station_id]:
        raise ValueError(
            f"frame must contain one station matching {station_id!r}; got {station_ids!r}"
        )
    return ordered


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
    """
    observed = frame["water_level"].notna() & ~frame["imputed"]
    future_observed = observed.shift(-1)
    valid_count = (
        future_observed.iloc[::-1]
        .rolling(horizon_hours, min_periods=horizon_hours)
        .sum()
        .iloc[::-1]
    )
    return valid_count.eq(horizon_hours).astype(bool).rename("target_valid")


def assign_expanding_fold_roles(
    development: pd.DataFrame,
    *,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Add expanding-window role columns to a development frame.

    Args:
        development: Chronologically ordered development observations.
        config: Shared physical-split and cross-validation configuration.

    Returns:
        A copy with categorical role columns and the nominal validation block sizes.

    Raises:
        ValueError: If the configuration cannot yield usable folds.
    """
    initial_size = math.floor(config.initial_train_fraction * len(development))
    validation_size = len(development) - initial_size
    base_size, remainder = divmod(validation_size, config.n_folds)
    block_sizes = tuple(
        base_size + (1 if fold_index < remainder else 0)
        for fold_index in range(config.n_folds)
    )
    if initial_size <= config.horizon_hours or any(
        size <= config.horizon_hours for size in block_sizes
    ):
        raise ValueError(
            "insufficient history for non-empty training and validation windows "
            f"with a {config.horizon_hours}-hour embargo"
        )

    result = development.copy()
    block_start = initial_size
    for fold_number, block_size in enumerate(block_sizes, start=1):
        block_end = block_start + block_size
        train_end = block_start - config.horizon_hours
        validation_end = block_end - config.horizon_hours
        roles = pd.Series("future", index=result.index, dtype="object")
        roles.iloc[:train_end] = "train"
        roles.iloc[train_end:block_start] = "embargo"
        roles.iloc[block_start:validation_end] = "validation"
        result[f"fold_{fold_number:02d}_role"] = pd.Categorical(
            roles, categories=ROLE_CATEGORIES
        )
        block_start = block_end
    return result, block_sizes


def split_station_frame(
    frame: pd.DataFrame,
    *,
    station_id: str,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> StationSplit:
    """Validate and split one station into development folds and sealed test data.

    Args:
        frame: Strict hourly station observations.
        station_id: Expected station identifier.
        config: Shared physical-split and cross-validation configuration.

    Returns:
        The train/test frames and validation block sizes.
    """
    ordered = validate_station_frame(frame, station_id=station_id)
    development_size = math.floor((1.0 - config.test_fraction) * len(ordered))
    development = ordered.iloc[:development_size].copy()
    test = ordered.iloc[development_size:].copy()
    development["target_valid"] = calculate_target_eligibility(
        development, horizon_hours=config.horizon_hours
    )
    test["target_valid"] = calculate_target_eligibility(
        test, horizon_hours=config.horizon_hours
    )
    development, block_sizes = assign_expanding_fold_roles(development, config=config)
    return StationSplit(development, test, block_sizes, config)


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


def _fold_metadata(split: StationSplit, horizon_hours: int) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    development_size = len(split.train)
    initial_size = development_size - sum(split.validation_block_sizes)
    block_start = initial_size
    for fold_number, block_size in enumerate(split.validation_block_sizes, start=1):
        role_column = f"fold_{fold_number:02d}_role"
        block_end = block_start + block_size
        train_end = block_start - horizon_hours
        validation_end = block_end - horizon_hours
        role_values = split.train[role_column]
        role_counts = {
            role: int((role_values == role).sum()) for role in ROLE_CATEGORIES
        }
        eligible_counts = {
            role: int(((role_values == role) & split.train["target_valid"]).sum())
            for role in ROLE_CATEGORIES
        }

        def boundary(start: int, end: int) -> dict[str, Any]:
            return {
                "start_row": start,
                "end_row_inclusive": end - 1,
                "start_utc": _utc_text(split.train["timestamp"].iloc[start]),
                "end_utc": _utc_text(split.train["timestamp"].iloc[end - 1]),
            }

        folds.append(
            {
                "fold": fold_number,
                "role_column": role_column,
                "nominal_validation_rows": block_size,
                "boundaries": {
                    "train": boundary(0, train_end),
                    "embargo": boundary(train_end, block_start),
                    "validation_anchors": boundary(block_start, validation_end),
                    "nominal_validation_block": boundary(block_start, block_end),
                },
                "role_counts": role_counts,
                "eligible_counts": eligible_counts,
            }
        )
        block_start = block_end
    return folds


def write_split_artifacts(
    split: StationSplit,
    *,
    station_id: str,
    source_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write both split Parquets followed by their shared metadata manifest.

    Existing artifacts are deliberately regenerated on every call.

    Args:
        split: Validated station split to persist.
        station_id: Station identifier used in artifact names and metadata.
        source_path: Interim Parquet from which ``split`` was produced.
        output_dir: Destination directory for all three artifacts.

    Returns:
        The manifest dictionary written to JSON.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        ValueError: If the supplied split does not match ``station_id`` or ``n_folds``.
    """
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if len(split.role_columns) != split.config.n_folds:
        raise ValueError("split fold count does not match its configuration")
    if set(split.train["station_id"].unique()) != {station_id} or set(
        split.test["station_id"].unique()
    ) != {station_id}:
        raise ValueError("split station identifiers do not match station_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{station_id}_train_splits.parquet"
    test_path = output_dir / f"{station_id}_test.parquet"
    metadata_path = output_dir / f"{station_id}_split_metadata.json"

    split.train.to_parquet(train_path, index=False)
    split.test.to_parquet(test_path, index=False)

    source_frame = pd.read_parquet(source_path)
    generator_path = Path(__file__)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "station_id": station_id,
        "configuration": {
            "forecast_horizon_hours": split.config.horizon_hours,
            "initial_train_fraction_of_development": split.config.initial_train_fraction,
            "n_folds": split.config.n_folds,
            "test_fraction": split.config.test_fraction,
        },
        "semantics": {
            "physical_split": "First floor((1-test_fraction)*N) rows are development; the remainder is sealed test data.",
            "target": "water_level at issue time t+1 through t+forecast_horizon_hours",
            "target_valid": "True only when every future target exists, is non-null, and is not imputed within the same physical artifact.",
            "roles": {
                "train": "Available to fit the fold model.",
                "embargo": "Issue times withheld immediately before validation.",
                "validation": "Eligible validation anchors whose targets remain inside the nominal block.",
                "future": "Unavailable to this fold, including the nominal block's final horizon rows.",
            },
        },
        "source": _frame_profile(source_frame, source_path),
        "artifacts": {
            "train_splits": _frame_profile(split.train, train_path),
            "test": _frame_profile(split.test, test_path),
        },
        "folds": _fold_metadata(split, split.config.horizon_hours),
        "generator": {
            "module": _relative_path(generator_path),
            "sha256": _sha256(generator_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pandas": pd.__version__,
            "pyarrow": version("pyarrow"),
            "platform": sys.platform,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
