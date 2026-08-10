"""Clean raw water-level series and merge them with weather data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import TEST_FRACTION


def clean_water_level(raw: pd.DataFrame, *, max_gap_hours: int) -> pd.DataFrame:
    """Reindex a station's water-level history to a strict hourly UTC grid.

    Gaps of at most ``max_gap_hours`` consecutive missing hours are linearly
    interpolated and flagged via the ``imputed`` column; longer gaps are left
    as ``NaN`` and not flagged.

    Args:
        raw: Raw PegelAlarm history with ``sourceDate``, ``value``, and
            ``station_id`` columns.
        max_gap_hours: Maximum length, in hours, of a gap that gets
            interpolated.

    Returns:
        DataFrame indexed by hourly UTC ``timestamp`` with ``water_level``,
        ``imputed``, and ``station_id`` columns.
    """
    station_id = raw["station_id"].iloc[0]
    series = raw.set_index("sourceDate")["value"].sort_index()
    grid = pd.date_range(
        series.index.min(), series.index.max(), freq="h", name="timestamp"
    )
    series = series.reindex(grid)

    missing = series.isna()
    gap_id = (missing != missing.shift()).cumsum()
    gap_length = missing.groupby(gap_id).transform("size").where(missing, 0)
    interpolated = series.interpolate(
        method="time", limit_area="inside", limit_direction="forward"
    )
    filled = series.where(gap_length > max_gap_hours, interpolated)
    imputed = missing & filled.notna()

    return pd.DataFrame(
        {
            "water_level": filled,
            "imputed": imputed,
            "station_id": station_id,
        }
    )


def merge_weather(
    water: pd.DataFrame, weather: pd.DataFrame, *, variables: Sequence[str]
) -> pd.DataFrame:
    """Left-join weather variables onto a water-level hourly grid.

    Args:
        water: Output of :func:`clean_water_level`, indexed by ``timestamp``.
        weather: Raw GeoSphere INCA weather history with a ``time`` column plus
            ``variables`` and other metadata columns.
        variables: Weather columns to keep.

    Returns:
        ``water`` with the requested weather columns appended.
    """
    trimmed = weather.set_index("time")[list(variables)]
    trimmed.index = trimmed.index.rename("timestamp")
    return water.join(trimmed, how="left")


def preprocess_station(
    station_id: str,
    *,
    raw_dir: Path,
    max_gap_hours: int,
    weather_variables: Sequence[str],
) -> pd.DataFrame:
    """Load a station's raw parquet files and build its analysis-ready frame.

    Args:
        station_id: PegelAlarm station identifier.
        raw_dir: Directory containing the raw parquet files from
            `01_fetch_data.ipynb`.
        max_gap_hours: Maximum length, in hours, of a water-level gap that
            gets interpolated.
        weather_variables: Weather columns to keep.

    Returns:
        Merged, hourly, analysis-ready DataFrame for the station.
    """
    water_raw = pd.read_parquet(
        raw_dir / f"pegelalarm_{station_id}_height_hour.parquet"
    )
    weather_raw = pd.read_parquet(raw_dir / f"geosphere_inca_{station_id}_hour.parquet")

    water = clean_water_level(water_raw, max_gap_hours=max_gap_hours)
    return merge_weather(water, weather_raw, variables=weather_variables).reset_index()


def _validate_hourly_station_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy after validating one contiguous hourly UTC station."""
    missing = sorted({"timestamp", "station_id"}.difference(frame.columns))
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
    if len(station_ids) != 1 or pd.isna(station_ids[0]):
        raise ValueError(
            f"frame must contain exactly one non-null station; got {station_ids!r}"
        )
    return ordered


def split_train_test(
    frame: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one hourly station timeline into chronological train and test rows.

    Args:
        frame: Preprocessed observations for exactly one station.
        test_fraction: Fraction assigned to the sealed tail partition.

    Returns:
        Independent train and test DataFrames with fresh row indexes.

    Raises:
        TypeError: If the fraction is not numeric.
        ValueError: If the fraction or station timeline is invalid, or if either
            resulting partition would be empty.
    """
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise TypeError("test_fraction must be a finite number between 0 and 1")
    if not math.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    ordered = _validate_hourly_station_frame(frame)
    train_size = math.floor((1.0 - test_fraction) * len(ordered))
    if train_size == 0 or train_size == len(ordered):
        raise ValueError(
            "test_fraction must produce non-empty train and test partitions"
        )
    train = ordered.iloc[:train_size].reset_index(drop=True)
    test = ordered.iloc[train_size:].reset_index(drop=True)
    return train, test


def join_station_frames(
    station_frames: Mapping[str, pd.DataFrame], *, target_station_id: str
) -> pd.DataFrame:
    """Join station frames onto the target station's timestamp timeline.

    Every non-timestamp column is prefixed with its station identifier so that
    complete preprocessed frames can be represented in one wide DataFrame.
    The target station supplies the output timestamps; timestamps unavailable
    for an upstream station therefore remain present with missing upstream
    values.

    Args:
        station_frames: Mapping from station identifier to one preprocessed
            station frame.
        target_station_id: Station whose timestamps define the output rows.

    Returns:
        A wide DataFrame with one unprefixed ``timestamp`` column and all
        station columns prefixed as ``<station_id>__<column>``.

    Raises:
        ValueError: If the mapping is empty, the target is missing, a frame's
            station identifier does not match its mapping key, or a frame is
            not a valid hourly station timeline.
    """
    if not station_frames:
        raise ValueError("station_frames must contain at least one station")
    if target_station_id not in station_frames:
        raise ValueError(
            f"target station {target_station_id!r} is missing from station_frames"
        )

    prepared: dict[str, pd.DataFrame] = {}
    for station_id, frame in station_frames.items():
        if not frame.columns.is_unique:
            raise ValueError(f"frame for {station_id!r} has duplicate columns")

        ordered = _validate_hourly_station_frame(frame)
        frame_station_ids = ordered["station_id"].drop_duplicates().tolist()
        if frame_station_ids != [station_id]:
            raise ValueError(
                f"frame station identifier does not match mapping key {station_id!r}: "
                f"{frame_station_ids!r}"
            )

        renamed = ordered.rename(
            columns={
                column: f"{station_id}__{column}"
                for column in ordered.columns
                if column != "timestamp"
            }
        )
        prepared[station_id] = renamed

    station_order = [
        target_station_id,
        *(
            station_id
            for station_id in station_frames
            if station_id != target_station_id
        ),
    ]
    joined = prepared[target_station_id]
    for station_id in station_order[1:]:
        joined = joined.merge(
            prepared[station_id],
            on="timestamp",
            how="left",
            sort=False,
            validate="one_to_one",
        )
    return joined


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
    }


def write_preprocess_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    station_id: str,
    output_dir: Path,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, Any]:
    """Write chronological preprocessing artifacts and their lineage manifest.

    Args:
        train: Chronological leading partition returned by :func:`split_train_test`.
        test: Sealed trailing partition returned by :func:`split_train_test`.
        station_id: Identifier used in artifact names and metadata.
        output_dir: Destination directory for the three artifacts.
        test_fraction: Fraction used to create the supplied partitions.

    Returns:
        The metadata dictionary written to JSON.

    Raises:
        ValueError: If the partitions do not reconstruct a valid split for the
            station and fraction.
    """
    combined = pd.concat([train, test], ignore_index=True)
    expected_train, expected_test = split_train_test(combined, test_fraction)
    if not train.reset_index(drop=True).equals(expected_train):
        raise ValueError("train does not match the chronological split contract")
    if not test.reset_index(drop=True).equals(expected_test):
        raise ValueError("test does not match the chronological split contract")
    if expected_train["station_id"].iloc[0] != station_id:
        raise ValueError("partition station identifier does not match station_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{station_id}_train.parquet"
    test_path = output_dir / f"{station_id}_test.parquet"
    metadata_path = output_dir / f"{station_id}_preprocess_metadata.json"
    expected_train.to_parquet(train_path, index=False)
    expected_test.to_parquet(test_path, index=False)

    generator_path = Path(__file__)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "station_id": station_id,
        "configuration": {
            "test_fraction": test_fraction,
            "split_rule": "first floor((1-test_fraction)*N) rows are train; the remainder is sealed test data",
        },
        "rows": {
            "total": len(combined),
            "train": len(expected_train),
            "test": len(expected_test),
        },
        "artifacts": {
            "train": _frame_profile(expected_train, train_path),
            "test": _frame_profile(expected_test, test_path),
        },
        "generator": {
            "module": _relative_path(generator_path),
            "sha256": _sha256(generator_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
