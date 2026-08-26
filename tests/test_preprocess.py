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
    filter_station_frames_by_target_range_overlap,
    join_station_frames,
    merge_weather,
    preprocess_station,
    split_station_frames_at_target_boundary,
    split_train_test,
    write_joined_preprocess_artifacts,
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


def test_clean_water_level_forward_fills_short_gaps_and_flags_them() -> None:
    values: list[float | None] = [1.0, np.nan, np.nan, np.nan, 5.0, 6.0]
    raw = _raw_water(values)

    result = clean_water_level(raw, max_gap_hours=3)

    assert result["water_level"].tolist() == [1.0, 1.0, 1.0, 1.0, 5.0, 6.0]
    assert result["imputed"].tolist() == [False, True, True, True, False, False]


def test_clean_water_level_masks_zero_and_negative_values_before_forward_filling() -> (
    None
):
    raw = _raw_water([2.0, 0.0, -1.0, 5.0])

    result = clean_water_level(raw, max_gap_hours=2)

    assert result["water_level"].tolist() == [2.0, 2.0, 2.0, 5.0]
    assert result["imputed"].tolist() == [False, True, True, False]
    assert result.index.equals(
        pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    )


def test_clean_water_level_exact_threshold_is_invalid() -> None:
    raw = _raw_water([2.0, 1.0, 4.0])

    result = clean_water_level(raw, max_gap_hours=1, min_valid_water_level=1.0)

    assert result["water_level"].tolist() == [2.0, 2.0, 4.0]
    assert result["imputed"].tolist() == [False, True, False]


def test_clean_water_level_forward_fill_does_not_leak_future_values() -> None:
    values: list[float | None] = [1.0, np.nan, np.nan, np.nan, 5.0, 6.0]
    raw = _raw_water(values)

    result = clean_water_level(raw, max_gap_hours=3)

    filled = result["water_level"].tolist()[1:4]
    assert filled == [1.0, 1.0, 1.0]
    assert all(value == 1.0 for value in filled), (
        "filled values must equal the last observation before the gap, "
        "not a blend with the value after it"
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, -1.0, 2.0, 3.0],
        [2.0, 3.0, 0.0, -1.0],
        [2.0, 0.0, -1.0, 0.0, -1.0, 3.0],
    ],
)
def test_clean_water_level_leading_trailing_and_long_invalid_gaps_remain_null(
    values: list[float | None],
) -> None:
    result = clean_water_level(_raw_water(values), max_gap_hours=2)

    invalid_positions = [
        index for index, value in enumerate(values) if value is not None and value <= 0
    ]
    assert result.loc[result.index[invalid_positions], "water_level"].isna().all()
    assert result.loc[result.index[invalid_positions], "imputed"].eq(False).all()


@pytest.mark.parametrize("threshold", [True, np.nan, np.inf, -np.inf, "0.0"])
def test_clean_water_level_rejects_invalid_minimum_threshold(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError), match="min_valid_water_level"):
        clean_water_level(
            _raw_water([1.0, 2.0]),
            max_gap_hours=1,
            min_valid_water_level=threshold,  # type: ignore[arg-type]
        )


def test_preprocess_station_preserves_weather_and_hourly_timeline(
    tmp_path: Path,
) -> None:
    station_id = "station-at"
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    _raw_water([2.0, 0.0, -1.0, 5.0]).to_parquet(
        tmp_path / f"pegelalarm_{station_id}_height_hour.parquet", index=False
    )
    weather = pd.DataFrame(
        {
            "time": timestamps,
            "temperature_2m": [10.0, -2.0, 0.0, 12.0],
            "precipitation": [0.0, 1.5, -0.5, 2.0],
        }
    )
    weather.to_parquet(
        tmp_path / f"geosphere_inca_{station_id}_hour.parquet", index=False
    )

    result = preprocess_station(
        station_id,
        raw_dir=tmp_path,
        max_gap_hours=2,
        weather_variables=["temperature_2m", "precipitation"],
    )

    assert result["timestamp"].equals(timestamps.to_series(index=result.index))
    assert result["water_level"].tolist() == [2.0, 2.0, 2.0, 5.0]
    pd.testing.assert_frame_equal(
        result[["temperature_2m", "precipitation"]], weather.drop(columns="time")
    )


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


def test_join_station_frames_uses_target_timeline_and_prefixes_columns() -> None:
    target = _station_frame(4, station_id="target-at")
    upstream = _station_frame(2, station_id="upstream-at")
    upstream["timestamp"] = pd.date_range(
        "2024-01-01T01:00", periods=2, freq="h", tz="UTC"
    )

    result = join_station_frames(
        {"target-at": target, "upstream-at": upstream},
        target_station_id="target-at",
    )

    assert result["timestamp"].tolist() == target["timestamp"].tolist()
    assert list(result.columns) == [
        "timestamp",
        "target-at__water_level",
        "target-at__imputed",
        "target-at__station_id",
        "upstream-at__water_level",
        "upstream-at__imputed",
        "upstream-at__station_id",
    ]
    assert result["target-at__water_level"].tolist() == [0, 1, 2, 3]
    assert np.allclose(
        result["upstream-at__water_level"], [np.nan, 0, 1, np.nan], equal_nan=True
    )
    assert pd.isna(result["upstream-at__station_id"].iloc[0])
    assert result["upstream-at__station_id"].iloc[1:3].tolist() == [
        "upstream-at",
        "upstream-at",
    ]
    assert pd.isna(result["upstream-at__station_id"].iloc[3])


def test_join_station_frames_rejects_missing_target_and_mismatched_station() -> None:
    upstream = _station_frame(3, station_id="upstream-at")

    with pytest.raises(ValueError, match="target station"):
        join_station_frames({"upstream-at": upstream}, target_station_id="target-at")

    with pytest.raises(ValueError, match="does not match mapping key"):
        join_station_frames({"target-at": upstream}, target_station_id="target-at")


def test_filter_station_frames_retains_exact_threshold_and_target() -> None:
    target = _station_frame(10, station_id="target-at")
    exact = _station_frame(8, station_id="exact-at")
    exact["timestamp"] = pd.date_range(
        "2024-01-01T02:00", periods=8, freq="h", tz="UTC"
    )
    below = _station_frame(7, station_id="below-at")
    below["timestamp"] = pd.date_range(
        "2024-01-01T03:00", periods=7, freq="h", tz="UTC"
    )
    no_overlap = _station_frame(4, station_id="none-at")
    no_overlap["timestamp"] = pd.date_range("2024-01-02", periods=4, freq="h", tz="UTC")

    retained, report = filter_station_frames_by_target_range_overlap(
        {
            "target-at": target,
            "exact-at": exact,
            "below-at": below,
            "none-at": no_overlap,
        },
        target_station_id="target-at",
        min_overlap_fraction=0.80,
    )

    assert list(retained) == ["target-at", "exact-at"]
    report_by_station = {record["station_id"]: record for record in report}
    assert report_by_station["target-at"]["overlap_hours"] == 10
    assert report_by_station["exact-at"]["overlap_hours"] == 8
    assert report_by_station["exact-at"]["overlap_fraction"] == pytest.approx(0.80)
    assert report_by_station["exact-at"]["retained"] is True
    assert report_by_station["below-at"]["retained"] is False
    assert report_by_station["none-at"]["overlap_hours"] == 0
    assert report_by_station["none-at"]["retained"] is False


@pytest.mark.parametrize("threshold", [True, np.nan, -0.1, 1.1, "0.9"])
def test_filter_station_frames_rejects_invalid_thresholds(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError), match="min_overlap_fraction"):
        filter_station_frames_by_target_range_overlap(
            {"target-at": _station_frame(10)},
            target_station_id="target-at",
            min_overlap_fraction=threshold,  # type: ignore[arg-type]
        )


def test_filter_station_frames_rejects_missing_and_mismatched_frames() -> None:
    with pytest.raises(ValueError, match="target station"):
        filter_station_frames_by_target_range_overlap(
            {"other-at": _station_frame(10, station_id="other-at")},
            target_station_id="target-at",
        )

    with pytest.raises(ValueError, match="does not match mapping key"):
        filter_station_frames_by_target_range_overlap(
            {"target-at": _station_frame(10, station_id="other-at")},
            target_station_id="target-at",
        )


def test_joined_artifacts_can_persist_coverage_metadata(
    tmp_path: Path,
) -> None:
    target = _station_frame(20, station_id="target-at")
    upstream = _station_frame(20, station_id="upstream-at")
    retained, coverage_report = filter_station_frames_by_target_range_overlap(
        {"target-at": target, "upstream-at": upstream},
        target_station_id="target-at",
    )
    train_by_station, test_by_station, split_boundary = (
        split_station_frames_at_target_boundary(
            retained, target_station_id="target-at", test_fraction=0.20
        )
    )
    joined_train = join_station_frames(train_by_station, target_station_id="target-at")
    joined_test = join_station_frames(test_by_station, target_station_id="target-at")

    metadata = write_joined_preprocess_artifacts(
        joined_train,
        joined_test,
        station_ids=list(retained),
        target_station_id="target-at",
        output_dir=tmp_path,
        split_boundary_utc=split_boundary,
        test_fraction=0.20,
        coverage_report=coverage_report,
    )

    assert metadata["schema_version"] == "2.0"
    assert metadata["configuration"]["min_valid_water_level"] == 0.0
    assert metadata["coverage"] == coverage_report
    assert (
        json.loads((tmp_path / "all_stations_preprocess_metadata.json").read_text())[
            "coverage"
        ]
        == coverage_report
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
    split_boundary = train["timestamp"].iloc[-1]

    metadata = write_preprocess_artifacts(
        train,
        test,
        station_id=station_id,
        output_dir=output_dir,
        split_boundary_utc=split_boundary,
        test_fraction=0.20,
    )

    train_path = output_dir / f"{station_id}_train.parquet"
    test_path = output_dir / f"{station_id}_test.parquet"
    metadata_path = output_dir / f"{station_id}_preprocess_metadata.json"
    persisted_train = pd.read_parquet(train_path)
    persisted_test = pd.read_parquet(test_path)
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata["configuration"]["test_fraction"] == 0.20
    assert metadata["configuration"]["min_valid_water_level"] == 0.0
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
    split_boundary = train["timestamp"].iloc[-1]

    with pytest.raises(ValueError, match="train does not match"):
        write_preprocess_artifacts(
            train.iloc[:-1],
            pd.concat([train.iloc[[-1]], test], ignore_index=True),
            station_id="station-at",
            output_dir=tmp_path,
            split_boundary_utc=split_boundary,
            test_fraction=0.20,
        )


def _joined_partitions(
    *, rows: int = 500, test_fraction: float = 0.20, upstream_start: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    upstream = _station_frame(rows, station_id="upstream-at")
    if upstream_start is not None:
        upstream["timestamp"] = pd.date_range(
            upstream_start, periods=rows, freq="h", tz="UTC"
        )
    train_by_station, test_by_station, boundary = (
        split_station_frames_at_target_boundary(
            {
                "target-at": _station_frame(rows, station_id="target-at"),
                "upstream-at": upstream,
            },
            target_station_id="target-at",
            test_fraction=test_fraction,
        )
    )
    joined_train = join_station_frames(train_by_station, target_station_id="target-at")
    joined_test = join_station_frames(test_by_station, target_station_id="target-at")
    return joined_train, joined_test, boundary


def test_write_joined_artifacts_records_hashes_schemas_and_rows(
    tmp_path: Path,
) -> None:
    joined_train, joined_test, split_boundary = _joined_partitions()
    output_dir = tmp_path / "joined"

    metadata = write_joined_preprocess_artifacts(
        joined_train,
        joined_test,
        station_ids=["target-at", "upstream-at"],
        target_station_id="target-at",
        output_dir=output_dir,
        split_boundary_utc=split_boundary,
        test_fraction=0.20,
    )

    train_path = output_dir / "all_stations_train.parquet"
    test_path = output_dir / "all_stations_test.parquet"
    metadata_path = output_dir / "all_stations_preprocess_metadata.json"
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata["station_ids"] == ["target-at", "upstream-at"]
    assert metadata["target_station_id"] == "target-at"
    assert metadata["configuration"]["min_valid_water_level"] == 0.0
    assert metadata["rows"] == {"total": 500, "train": 400, "test": 100}
    assert metadata["artifacts"]["train"]["sha256"] == _sha256(train_path)
    assert metadata["artifacts"]["test"]["sha256"] == _sha256(test_path)
    assert metadata["generator"]["sha256"] == _sha256(Path("src/preprocess.py"))
    assert not Path(metadata["artifacts"]["train"]["path"]).is_absolute()


def test_write_joined_artifacts_rejects_target_missing_from_station_ids(
    tmp_path: Path,
) -> None:
    joined_train, joined_test, split_boundary = _joined_partitions()

    with pytest.raises(ValueError, match="target station"):
        write_joined_preprocess_artifacts(
            joined_train,
            joined_test,
            station_ids=["upstream-at"],
            target_station_id="target-at",
            output_dir=tmp_path,
            split_boundary_utc=split_boundary,
        )


def test_write_joined_artifacts_rejects_station_missing_from_joined_frame(
    tmp_path: Path,
) -> None:
    joined_train, joined_test, split_boundary = _joined_partitions()

    with pytest.raises(ValueError, match="missing station_id columns"):
        write_joined_preprocess_artifacts(
            joined_train,
            joined_test,
            station_ids=["target-at", "upstream-at", "other-at"],
            target_station_id="target-at",
            output_dir=tmp_path,
            split_boundary_utc=split_boundary,
        )


def _late_starting_station(*, rows: int, start: str, station_id: str) -> pd.DataFrame:
    frame = _station_frame(rows, station_id=station_id)
    frame["timestamp"] = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    return frame


def test_shared_boundary_splits_every_station_at_the_target_timestamp() -> None:
    target = _station_frame(500, station_id="target-at")
    upstream = _late_starting_station(
        rows=404, start="2024-01-05", station_id="upstream-at"
    )

    train_by_station, test_by_station, boundary = (
        split_station_frames_at_target_boundary(
            {"target-at": target, "upstream-at": upstream},
            target_station_id="target-at",
            test_fraction=0.20,
        )
    )

    assert boundary == target["timestamp"].iloc[399]
    for station_id in ("target-at", "upstream-at"):
        assert train_by_station[station_id]["timestamp"].max() == boundary
        assert test_by_station[station_id]["timestamp"].min() > boundary
    # The upstream station's own floor(0.8*N) would fall at a different hour.
    own_train, _own_test = split_train_test(upstream, 0.20)
    assert own_train["timestamp"].iloc[-1] != boundary


def test_shared_boundary_keeps_misaligned_stations_joinable() -> None:
    """The joined sealed test must not lose hours to split misalignment."""
    target = _station_frame(500, station_id="target-at")
    upstream = _late_starting_station(
        rows=404, start="2024-01-05", station_id="upstream-at"
    )
    frames = {"target-at": target, "upstream-at": upstream}

    _train_by_station, test_by_station, _boundary = (
        split_station_frames_at_target_boundary(
            frames, target_station_id="target-at", test_fraction=0.20
        )
    )
    joined_test = join_station_frames(test_by_station, target_station_id="target-at")

    assert joined_test["upstream-at__water_level"].notna().all()

    # Splitting each station at its own fraction strands upstream hours.
    independent_test = join_station_frames(
        {
            station_id: split_train_test(frame, 0.20)[1]
            for station_id, frame in frames.items()
        },
        target_station_id="target-at",
    )
    assert independent_test["upstream-at__water_level"].isna().any()


def test_shared_boundary_rejects_a_station_that_never_reaches_the_test_side() -> None:
    target = _station_frame(500, station_id="target-at")
    retired = _late_starting_station(
        rows=100, start="2024-01-01", station_id="retired-at"
    )

    with pytest.raises(ValueError, match="does not span the shared split boundary"):
        split_station_frames_at_target_boundary(
            {"target-at": target, "retired-at": retired},
            target_station_id="target-at",
            test_fraction=0.20,
        )


def test_shared_boundary_rejects_missing_target_and_mismatched_station() -> None:
    target = _station_frame(500, station_id="target-at")

    with pytest.raises(ValueError, match="target station"):
        split_station_frames_at_target_boundary(
            {"target-at": target}, target_station_id="absent-at"
        )
    with pytest.raises(ValueError, match="does not match mapping key"):
        split_station_frames_at_target_boundary(
            {"target-at": target, "upstream-at": target},
            target_station_id="target-at",
        )


def test_joined_writer_rejects_partitions_that_straddle_the_boundary(
    tmp_path: Path,
) -> None:
    joined_train, joined_test, split_boundary = _joined_partitions()

    with pytest.raises(ValueError, match="joined train rows must end at"):
        write_joined_preprocess_artifacts(
            pd.concat([joined_train, joined_test.iloc[[0]]], ignore_index=True),
            joined_test.iloc[1:].reset_index(drop=True),
            station_ids=["target-at", "upstream-at"],
            target_station_id="target-at",
            output_dir=tmp_path,
            split_boundary_utc=split_boundary,
        )
