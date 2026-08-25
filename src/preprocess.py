"""Clean raw water-level series and merge them with weather data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MIN_TARGET_RANGE_OVERLAP, MIN_VALID_WATER_LEVEL, TEST_FRACTION


def _validate_min_valid_water_level(value: object) -> float:
    """Validate and normalize a minimum water-level threshold."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("min_valid_water_level must be a finite real number")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("min_valid_water_level must be a finite real number")
    return threshold


def clean_water_level(
    raw: pd.DataFrame,
    *,
    max_gap_hours: int,
    min_valid_water_level: float = MIN_VALID_WATER_LEVEL,
) -> pd.DataFrame:
    """Reindex a station's water-level history to a strict hourly UTC grid.

    Raw values at or below ``min_valid_water_level`` are treated as missing.
    Interior gaps (bounded by a valid observation both before and after) of
    at most ``max_gap_hours`` consecutive missing hours are forward-filled
    from the last valid observation and flagged via the ``imputed`` column;
    longer, leading, and trailing gaps are left as ``NaN`` and not flagged.
    Forward-filling only ever reads values at or before the timestamp being
    filled, so the fill is causal with respect to issue time.

    Args:
        raw: Raw PegelAlarm history with ``sourceDate``, ``value``, and
            ``station_id`` columns.
        max_gap_hours: Maximum length, in hours, of a gap that gets
            forward-filled.
        min_valid_water_level: Raw values at or below this threshold are
            treated as missing before gap detection.

    Returns:
        DataFrame indexed by hourly UTC ``timestamp`` with ``water_level``,
        ``imputed``, and ``station_id`` columns.

    Raises:
        TypeError: If ``min_valid_water_level`` is not a real number.
        ValueError: If ``min_valid_water_level`` is not finite.
    """
    threshold = _validate_min_valid_water_level(min_valid_water_level)
    station_id = raw["station_id"].iloc[0]
    series = raw.set_index("sourceDate")["value"].sort_index()
    series = series.where(series > threshold)
    grid = pd.date_range(
        series.index.min(), series.index.max(), freq="h", name="timestamp"
    )
    series = series.reindex(grid)

    missing = series.isna()
    gap_id = (missing != missing.shift()).cumsum()
    gap_length = missing.groupby(gap_id).transform("size").where(missing, 0)
    forward_filled = series.ffill()
    is_interior_gap = missing & series.ffill().notna() & series.bfill().notna()
    fillable = is_interior_gap & (gap_length <= max_gap_hours)
    filled = series.where(~fillable, forward_filled)
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
    min_valid_water_level: float = MIN_VALID_WATER_LEVEL,
) -> pd.DataFrame:
    """Load a station's raw parquet files and build its analysis-ready frame.

    Args:
        station_id: PegelAlarm station identifier.
        raw_dir: Directory containing the raw parquet files from
            `01_fetch_data.ipynb`.
        max_gap_hours: Maximum length, in hours, of a water-level gap that
            gets forward-filled.
        weather_variables: Weather columns to keep.
        min_valid_water_level: Raw water-level values at or below this
            threshold are treated as missing before gap filling.

    Returns:
        Merged, hourly, analysis-ready DataFrame for the station.

    Raises:
        TypeError: If ``min_valid_water_level`` is not a real number.
        ValueError: If ``min_valid_water_level`` is not finite.
    """
    water_raw = pd.read_parquet(
        raw_dir / f"pegelalarm_{station_id}_height_hour.parquet"
    )
    weather_raw = pd.read_parquet(raw_dir / f"geosphere_inca_{station_id}_hour.parquet")

    water = clean_water_level(
        water_raw,
        max_gap_hours=max_gap_hours,
        min_valid_water_level=min_valid_water_level,
    )
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


def filter_station_frames_by_target_range_overlap(
    station_frames: Mapping[str, pd.DataFrame],
    *,
    target_station_id: str,
    min_overlap_fraction: float = MIN_TARGET_RANGE_OVERLAP,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Retain station timelines that overlap the target range sufficiently.

    The overlap is measured from inclusive hourly timestamp spans, using the
    complete target timeline as the denominator.  A station with timestamps
    from ``target_start`` through ``target_end`` therefore has
    ``target_hours`` possible overlapping hours.  The target station is always
    retained after its own timeline has been validated.

    Args:
        station_frames: Complete, pre-split station frames keyed by station ID.
        target_station_id: Station whose complete timeline defines the target
            range and denominator.
        min_overlap_fraction: Minimum positive fraction of the target range a
            non-target station must overlap. The exact threshold is retained.

    Returns:
        A mapping containing the validated, sorted frames that pass the filter,
        followed by JSON-serializable per-station coverage records.

    Raises:
        TypeError: If ``min_overlap_fraction`` is not a real number.
        ValueError: If the threshold or any station timeline is invalid, the
            target is missing, or a frame's station ID does not match its key.
    """
    if isinstance(min_overlap_fraction, bool) or not isinstance(
        min_overlap_fraction, Real
    ):
        raise TypeError("min_overlap_fraction must be a finite number between 0 and 1")
    threshold = float(min_overlap_fraction)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_overlap_fraction must be between 0 and 1")
    if not station_frames:
        raise ValueError("station_frames must contain at least one station")
    if target_station_id not in station_frames:
        raise ValueError(
            f"target station {target_station_id!r} is missing from station_frames"
        )

    validated: dict[str, pd.DataFrame] = {}
    for station_id, frame in station_frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"frame for {station_id!r} must be a pandas DataFrame")
        if not frame.columns.is_unique:
            raise ValueError(f"frame for {station_id!r} has duplicate columns")
        ordered = _validate_hourly_station_frame(frame)
        frame_station_ids = ordered["station_id"].drop_duplicates().tolist()
        if frame_station_ids != [station_id]:
            raise ValueError(
                f"frame station identifier does not match mapping key {station_id!r}: "
                f"{frame_station_ids!r}"
            )
        validated[station_id] = ordered

    target = validated[target_station_id]
    target_start = target["timestamp"].iloc[0]
    target_end = target["timestamp"].iloc[-1]
    target_hours = len(target)
    coverage_report: list[dict[str, Any]] = []
    retained: dict[str, pd.DataFrame] = {}

    for station_id, frame in validated.items():
        station_start = frame["timestamp"].iloc[0]
        station_end = frame["timestamp"].iloc[-1]
        overlap_start = max(station_start, target_start)
        overlap_end = min(station_end, target_end)
        if overlap_start <= overlap_end:
            overlap_hours = (
                int((overlap_end - overlap_start).total_seconds() // (60 * 60)) + 1
            )
        else:
            overlap_hours = 0
        overlap_fraction = overlap_hours / target_hours
        is_target = station_id == target_station_id
        is_retained = is_target or (overlap_hours > 0 and overlap_fraction >= threshold)
        record: dict[str, Any] = {
            "station_id": station_id,
            "station_start_utc": _utc_text(station_start),
            "station_end_utc": _utc_text(station_end),
            "target_start_utc": _utc_text(target_start),
            "target_end_utc": _utc_text(target_end),
            "overlap_start_utc": (_utc_text(overlap_start) if overlap_hours else None),
            "overlap_end_utc": _utc_text(overlap_end) if overlap_hours else None,
            "target_hours": target_hours,
            "overlap_hours": overlap_hours,
            "overlap_fraction": overlap_fraction,
            "retained": is_retained,
        }
        coverage_report.append(record)
        if is_retained:
            retained[station_id] = frame

    return retained, coverage_report


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


def _validate_joined_frame(
    combined: pd.DataFrame, *, station_ids: Sequence[str]
) -> None:
    """Validate the target's contiguous hourly timeline and station coverage."""
    if "timestamp" not in combined.columns:
        raise ValueError("missing required column: timestamp")
    if combined.empty:
        raise ValueError("joined timeline is empty")

    timestamp_dtype = combined["timestamp"].dtype
    if (
        not isinstance(timestamp_dtype, pd.DatetimeTZDtype)
        or str(timestamp_dtype.tz) != "UTC"
    ):
        raise ValueError("timestamp must be timezone-aware UTC")
    if combined["timestamp"].duplicated().any():
        raise ValueError("timestamps must be unique")
    expected_grid = pd.date_range(
        combined["timestamp"].iloc[0], periods=len(combined), freq="h"
    )
    if not pd.DatetimeIndex(combined["timestamp"]).equals(expected_grid):
        raise ValueError("timestamps must form a contiguous hourly grid")

    missing_station_columns = sorted(
        station_id
        for station_id in station_ids
        if f"{station_id}__station_id" not in combined.columns
    )
    if missing_station_columns:
        raise ValueError(
            f"joined frame is missing station_id columns for: {missing_station_columns}"
        )


def write_joined_preprocess_artifacts(
    joined_train: pd.DataFrame,
    joined_test: pd.DataFrame,
    *,
    station_ids: Sequence[str],
    target_station_id: str,
    output_dir: Path,
    test_fraction: float = TEST_FRACTION,
    min_valid_water_level: float = MIN_VALID_WATER_LEVEL,
    coverage_report: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the joined chronological artifacts and their lineage manifest.

    Args:
        joined_train: Chronological leading partition from :func:`join_station_frames`.
        joined_test: Sealed trailing partition from :func:`join_station_frames`.
        station_ids: Every station represented in the joined frames.
        target_station_id: Station whose timeline defines the joined timestamps.
        output_dir: Destination directory for the three artifacts.
        test_fraction: Fraction used to create the supplied partitions.
        min_valid_water_level: Raw water-level values at or below this
            threshold were treated as missing before gap filling.
        coverage_report: Optional per-station target-range overlap records from
            :func:`filter_station_frames_by_target_range_overlap`.

    Returns:
        The metadata dictionary written to JSON.

    Raises:
        TypeError: If ``min_valid_water_level`` is not a real number.
        ValueError: If ``min_valid_water_level`` is not finite, the target's
            timeline is not contiguous hourly UTC, a station is missing its
            prefixed ``station_id`` column, or ``target_station_id`` is not in
            ``station_ids``.
    """
    if target_station_id not in station_ids:
        raise ValueError(
            f"target station {target_station_id!r} is missing from station_ids"
        )
    threshold = _validate_min_valid_water_level(min_valid_water_level)
    combined = pd.concat([joined_train, joined_test], ignore_index=True)
    _validate_joined_frame(combined, station_ids=station_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "all_stations_train.parquet"
    test_path = output_dir / "all_stations_test.parquet"
    metadata_path = output_dir / "all_stations_preprocess_metadata.json"
    joined_train.to_parquet(train_path, index=False)
    joined_test.to_parquet(test_path, index=False)

    generator_path = Path(__file__)
    metadata: dict[str, Any] = {
        "schema_version": "1.1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "station_ids": list(station_ids),
        "target_station_id": target_station_id,
        "configuration": {
            "test_fraction": test_fraction,
            "min_valid_water_level": threshold,
            "min_target_range_overlap": MIN_TARGET_RANGE_OVERLAP,
            "split_rule": "first floor((1-test_fraction)*N) rows are train; the remainder is sealed test data",
        },
        "rows": {
            "total": len(combined),
            "train": len(joined_train),
            "test": len(joined_test),
        },
        "artifacts": {
            "train": _frame_profile(joined_train, train_path),
            "test": _frame_profile(joined_test, test_path),
        },
        "generator": {
            "module": _relative_path(generator_path),
            "sha256": _sha256(generator_path),
        },
    }
    if coverage_report is not None:
        metadata["coverage"] = [dict(record) for record in coverage_report]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def write_preprocess_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    station_id: str,
    output_dir: Path,
    test_fraction: float = TEST_FRACTION,
    min_valid_water_level: float = MIN_VALID_WATER_LEVEL,
) -> dict[str, Any]:
    """Write chronological preprocessing artifacts and their lineage manifest.

    Args:
        train: Chronological leading partition returned by :func:`split_train_test`.
        test: Sealed trailing partition returned by :func:`split_train_test`.
        station_id: Identifier used in artifact names and metadata.
        output_dir: Destination directory for the three artifacts.
        test_fraction: Fraction used to create the supplied partitions.
        min_valid_water_level: Raw water-level values at or below this
            threshold were treated as missing before gap filling.

    Returns:
        The metadata dictionary written to JSON.

    Raises:
        TypeError: If ``min_valid_water_level`` is not a real number.
        ValueError: If ``min_valid_water_level`` is not finite or the partitions
            do not reconstruct a valid split for the station and fraction.
    """
    threshold = _validate_min_valid_water_level(min_valid_water_level)
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
            "min_valid_water_level": threshold,
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
