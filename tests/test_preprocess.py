from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    clean_water_level,
    merge_weather,
    split_train_test,
    write_preprocess_artifacts,
)


def _raw_water(values: list[float | None]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "sourceDate": index,
            "value": values,
            "station_id": "station-at",
        }
    )


def test_clean_water_level_reindexes_to_a_strict_hourly_grid() -> None:
    raw = _raw_water([1.0, 2.0, 3.0])
    raw = raw.drop(index=1)  # drop the middle timestamp entirely

    result = clean_water_level(raw, max_gap_hours=6)

    assert (
        result.index.tolist()
        == pd.date_range("2024-01-01T00:00", periods=3, freq="h", tz="UTC").tolist()
    )


def test_clean_water_level_interpolates_short_gaps_and_flags_them() -> None:
    values: list[float | None] = [1.0, np.nan, np.nan, np.nan, 5.0, 6.0]
    raw = _raw_water(values)

    result = clean_water_level(raw, max_gap_hours=3)

    assert result["water_level"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["imputed"].tolist() == [False, True, True, True, False, False]


def test_clean_water_level_leaves_long_gaps_as_nan_and_unflagged() -> None:
    values: list[float | None] = [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]
    raw = _raw_water(values)

    result = clean_water_level(raw, max_gap_hours=3)

    water_level = result["water_level"].tolist()
    assert water_level[0] == 1.0
    assert water_level[-1] == 6.0
    assert all(np.isnan(value) for value in water_level[1:-1])
    assert result["imputed"].tolist() == [False, False, False, False, False, False]
    assert result["station_id"].unique().tolist() == ["station-at"]


def test_merge_weather_keeps_only_requested_variables_and_left_joins() -> None:
    water_index = pd.date_range("2024-01-01T00:00", periods=3, freq="h", tz="UTC")
    water = pd.DataFrame(
        {"water_level": [1.0, 2.0, 3.0], "imputed": False, "station_id": "station-at"},
        index=water_index,
    )
    water.index = water.index.rename("timestamp")

    weather_index = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T02:00Z"], utc=True)
    weather = pd.DataFrame(
        {
            "time": weather_index,
            "temperature_2m": [10.0, 12.0],
            "precipitation": [0.0, 1.0],
            "weather_model": "inca-v1-1h-1km",
            "requested_latitude": 47.0,
        }
    )

    result = merge_weather(
        water, weather, variables=["temperature_2m", "precipitation"]
    )

    assert list(result.columns) == [
        "water_level",
        "imputed",
        "station_id",
        "temperature_2m",
        "precipitation",
    ]
    assert np.allclose(
        result["temperature_2m"].tolist(), [10.0, np.nan, 12.0], equal_nan=True
    )
    assert np.allclose(
        result["precipitation"].tolist(), [0.0, np.nan, 1.0], equal_nan=True
    )


def _station_frame(rows: int, *, station_id: str = "station-at") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "water_level": range(rows),
            "imputed": False,
            "station_id": station_id,
        }
    )


def test_split_reconstructs_source_with_exact_chronological_partition() -> None:
    source = _station_frame(503).sample(frac=1, random_state=42)

    train, test = split_train_test(source, 0.20)

    assert len(train) == 402
    assert len(test) == 101
    assert train["timestamp"].max() < test["timestamp"].min()
    pd.testing.assert_frame_equal(
        pd.concat([train, test], ignore_index=True),
        source.sort_values("timestamp").reset_index(drop=True),
    )


@pytest.mark.parametrize(
    ("rows", "test_fraction", "expected_train_rows"),
    [(503, 0.20, 402), (10, 0.25, 7), (3, 0.50, 1)],
)
def test_split_uses_deterministic_floor_rounding(
    rows: int, test_fraction: float, expected_train_rows: int
) -> None:
    train, test = split_train_test(_station_frame(rows), test_fraction)

    assert len(train) == expected_train_rows
    assert len(test) == rows - expected_train_rows


@pytest.mark.parametrize("test_fraction", [0.0, 1.0, -0.1, 1.1, np.nan, True])
def test_split_rejects_invalid_test_fraction(test_fraction: float) -> None:
    with pytest.raises((TypeError, ValueError), match="test_fraction"):
        split_train_test(_station_frame(10), test_fraction)


def test_split_rejects_empty_partitions() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        split_train_test(_station_frame(1), 0.20)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="timestamp"), "required columns"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "unique",
        ),
        (lambda frame: frame.drop(index=10), "contiguous hourly"),
        (
            lambda frame: frame.assign(
                timestamp=frame["timestamp"].dt.tz_localize(None)
            ),
            "UTC",
        ),
        (
            lambda frame: frame.assign(
                station_id=[
                    "other" if index == 0 else "station-at" for index in frame.index
                ]
            ),
            "exactly one",
        ),
    ],
)
def test_split_rejects_invalid_station_timelines(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_train_test(mutate(_station_frame(100)), 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_write_artifacts_records_hashes_schemas_rows_and_boundaries(
    tmp_path: Path,
) -> None:
    station_id = "station-at"
    output_dir = tmp_path / "processed"
    train, test = split_train_test(_station_frame(503), 0.20)

    metadata = write_preprocess_artifacts(
        train,
        test,
        station_id=station_id,
        output_dir=output_dir,
        test_fraction=0.20,
    )

    train_path = output_dir / f"{station_id}_train.parquet"
    test_path = output_dir / f"{station_id}_test.parquet"
    metadata_path = output_dir / f"{station_id}_preprocess_metadata.json"
    persisted_train = pd.read_parquet(train_path)
    persisted_test = pd.read_parquet(test_path)
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata["configuration"]["test_fraction"] == 0.20
    assert metadata["rows"] == {"total": 503, "train": 402, "test": 101}
    assert metadata["artifacts"]["train"]["sha256"] == _sha256(train_path)
    assert metadata["artifacts"]["test"]["sha256"] == _sha256(test_path)
    assert metadata["generator"]["sha256"] == _sha256(Path("src/preprocess.py"))
    assert metadata["artifacts"]["train"]["schema"] == {
        column: str(dtype) for column, dtype in persisted_train.dtypes.items()
    }
    assert metadata["artifacts"]["test"]["timestamp_range"] == {
        "start_utc": persisted_test["timestamp"]
        .iloc[0]
        .isoformat()
        .replace("+00:00", "Z"),
        "end_utc": persisted_test["timestamp"]
        .iloc[-1]
        .isoformat()
        .replace("+00:00", "Z"),
    }
    assert not Path(metadata["artifacts"]["train"]["path"]).is_absolute()
    assert metadata_path.stat().st_mtime_ns >= train_path.stat().st_mtime_ns
    assert metadata_path.stat().st_mtime_ns >= test_path.stat().st_mtime_ns


def test_writer_rejects_partitions_that_do_not_match_the_contract(
    tmp_path: Path,
) -> None:
    train, test = split_train_test(_station_frame(100), 0.20)

    with pytest.raises(ValueError, match="train does not match"):
        write_preprocess_artifacts(
            train.iloc[:-1],
            pd.concat([train.iloc[[-1]], test], ignore_index=True),
            station_id="station-at",
            output_dir=tmp_path,
            test_fraction=0.20,
        )
