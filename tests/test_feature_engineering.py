from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import WEATHER_VARIABLES
from src.feature_engineering import (
    DEFAULT_FEATURE_CONFIG,
    FeatureConfig,
    build_feature_frame,
    calculate_target_eligibility,
    extract_station_frame,
    feature_column_names,
    is_log1p_eligible_base_name,
    log1p_eligible_columns,
    target_column_names,
    write_feature_artifacts,
    write_joined_feature_artifacts,
)
from src.preprocess import join_station_frames


def _station_frame(
    rows: int = 220,
    *,
    station_id: str = "station-at",
    start: str = "2024-01-01",
    water_offset: float = 0.0,
) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=rows, freq="h", tz="UTC"),
            "water_level": values + water_offset,
            "imputed": False,
            "station_id": station_id,
            "precipitation": values.astype("float32"),
            "temperature_2m": (280.0 + values).astype("float32"),
        }
    )
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_column_helpers_define_the_complete_stable_contract() -> None:
    predictors = feature_column_names()
    targets = target_column_names()

    assert predictors[:4] == ("water_level", "imputed", *WEATHER_VARIABLES)
    assert predictors[-6:] == (
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_day_of_week_sin",
        "utc_day_of_week_cos",
        "utc_day_of_year_sin",
        "utc_day_of_year_cos",
    )
    assert len(predictors) == len(set(predictors)) == 45
    assert targets == tuple(f"target_t_plus_{hour:02d}" for hour in range(1, 25))


def test_builds_exact_lag_change_rolling_weather_and_calendar_values() -> None:
    frame = _station_frame()
    frame.loc[160, "imputed"] = True

    result = build_feature_frame(frame, station_id="station-at")
    row = result.iloc[180]

    assert row["water_level_lag_1h"] == 179.0
    assert row["water_level_lag_72h"] == 108.0
    assert row["water_level_change_24h"] == 24.0
    assert row["water_level_rolling_mean_6h"] == pytest.approx(177.5)
    assert row["water_level_rolling_std_6h"] == pytest.approx(
        np.std(np.arange(175.0, 181.0), ddof=0)
    )
    assert row["water_level_rolling_min_24h"] == 157.0
    assert row["water_level_rolling_max_24h"] == 180.0
    assert row["imputed_count_24h"] == 1.0
    assert row["precipitation_rolling_sum_6h"] == sum(range(175, 181))
    assert row["temperature_2m_rolling_mean_6h"] == pytest.approx(457.5)
    assert row["temperature_2m_rolling_min_24h"] == 437.0
    assert row["temperature_2m_rolling_max_24h"] == 460.0

    midnight_monday = result.iloc[0]
    assert midnight_monday["utc_hour_sin"] == pytest.approx(0.0)
    assert midnight_monday["utc_hour_cos"] == pytest.approx(1.0)
    assert midnight_monday["utc_day_of_week_sin"] == pytest.approx(0.0)
    assert midnight_monday["utc_day_of_week_cos"] == pytest.approx(1.0)
    assert result.iloc[24]["utc_day_of_year_sin"] == pytest.approx(
        np.sin(2 * np.pi / 366)
    )


def test_targets_point_forward_and_invalid_anchors_have_blank_vectors() -> None:
    frame = _station_frame()

    result = build_feature_frame(frame, station_id="station-at")
    targets = list(target_column_names())

    assert result.loc[[10], targets].to_numpy().ravel().tolist() == list(
        np.arange(11.0, 35.0)
    )
    assert result.loc[[195], targets].to_numpy().ravel().tolist() == list(
        np.arange(196.0, 220.0)
    )
    assert result.loc[196:, targets].isna().all(axis=None)


def test_target_eligibility_requires_observed_non_imputed_future_values() -> None:
    frame = _station_frame(rows=80)
    frame.loc[10, "water_level"] = np.nan
    frame.loc[30, "imputed"] = True

    eligible = calculate_target_eligibility(frame)
    result = build_feature_frame(frame, station_id="station-at")

    assert not eligible.iloc[:10].any()
    assert not eligible.iloc[6:30].any()
    assert eligible.iloc[30]
    assert eligible.iloc[40:56].all()
    assert not eligible.iloc[-24:].any()
    pd.testing.assert_series_equal(result["target_valid"], eligible)


def test_incomplete_or_missing_lookbacks_remain_nan_without_dropping_rows() -> None:
    frame = _station_frame()
    frame.loc[10, "water_level"] = np.nan
    frame.loc[20, "precipitation"] = np.nan

    result = build_feature_frame(frame, station_id="station-at")

    assert len(result) == len(frame)
    assert result.index.equals(frame.index)
    assert result.loc[:71, "water_level_lag_72h"].isna().all()
    assert np.isnan(float(result["water_level_rolling_mean_6h"].iloc[12]))
    assert np.isnan(float(result["precipitation_rolling_sum_6h"].iloc[25]))
    assert not np.isnan(float(result["precipitation_rolling_sum_6h"].iloc[26]))


def test_physical_artifacts_are_transformed_independently() -> None:
    train = _station_frame(start="2024-01-01", water_offset=0.0)
    test = _station_frame(start="2024-01-10 04:00", water_offset=10_000.0)

    train_features = build_feature_frame(train, station_id="station-at")
    test_features = build_feature_frame(test, station_id="station-at")

    assert np.isnan(test_features.iloc[0]["water_level_lag_1h"])
    assert test_features.iloc[:5]["water_level_rolling_mean_6h"].isna().all()
    assert test_features.iloc[5]["water_level_rolling_mean_6h"] == pytest.approx(
        10_002.5
    )
    assert train_features.iloc[-1]["water_level"] == 219.0
    assert not train_features["target_valid"].iloc[-24:].any()
    assert not test_features["target_valid"].iloc[-24:].any()


def test_original_columns_dtypes_rows_and_index_are_preserved() -> None:
    frame = _station_frame()
    frame.index = pd.Index(range(1000, 1220), name="source_row")
    original = frame.copy(deep=True)

    result = build_feature_frame(frame, station_id="station-at")

    assert result.index.equals(original.index)
    assert result.columns[: len(original.columns)].tolist() == original.columns.tolist()
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.columns[len(original.columns)] == "target_valid"
    assert "features_valid" not in result


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="precipitation"), "missing required columns"),
        (
            lambda frame: frame.assign(
                timestamp=frame["timestamp"].dt.tz_localize(None)
            ),
            "UTC",
        ),
        (lambda frame: frame.drop(index=10), "contiguous hourly"),
        (lambda frame: frame.iloc[::-1], "contiguous hourly"),
        (
            lambda frame: frame.assign(
                station_id=np.where(frame.index == 0, "other", "station-at")
            ),
            "one station",
        ),
    ],
)
def test_rejects_invalid_physical_artifacts(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_feature_frame(mutate(_station_frame()), station_id="station-at")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FeatureConfig(horizon_hours=0),
        lambda: FeatureConfig(calendar_timezone="Europe/Vienna"),
        lambda: FeatureConfig(lag_hours=(3, 1)),
        lambda: FeatureConfig(rolling_windows=()),
    ],
)
def test_feature_config_rejects_invalid_values(
    factory: Callable[[], FeatureConfig],
) -> None:
    with pytest.raises(ValueError):
        factory()


def test_write_artifacts_replaces_files_and_records_complete_lineage(
    tmp_path: Path,
) -> None:
    station_id = "station-at"
    train_source_path = tmp_path / f"{station_id}_train.parquet"
    test_source_path = tmp_path / f"{station_id}_test.parquet"
    output_dir = tmp_path / "processed"
    train = _station_frame(start="2024-01-01")
    test = _station_frame(start="2024-02-01")
    train.to_parquet(train_source_path, index=False)
    test.to_parquet(test_source_path, index=False)
    train_features = build_feature_frame(train, station_id=station_id)
    test_features = build_feature_frame(test, station_id=station_id)
    output_dir.mkdir()
    train_path = output_dir / f"{station_id}_train_features.parquet"
    test_path = output_dir / f"{station_id}_test_features.parquet"
    train_path.write_bytes(b"obsolete")
    test_path.write_bytes(b"obsolete")

    metadata = write_feature_artifacts(
        train_features,
        test_features,
        station_id=station_id,
        train_source_path=train_source_path,
        test_source_path=test_source_path,
        output_dir=output_dir,
    )
    first_hashes = (_sha256(train_path), _sha256(test_path))
    repeated = write_feature_artifacts(
        train_features,
        test_features,
        station_id=station_id,
        train_source_path=train_source_path,
        test_source_path=test_source_path,
        output_dir=output_dir,
    )

    metadata_path = output_dir / f"{station_id}_feature_metadata.json"
    persisted_metadata = json.loads(metadata_path.read_text())
    assert persisted_metadata == repeated
    assert metadata["schema_version"] == "1.0"
    assert first_hashes == (_sha256(train_path), _sha256(test_path))
    assert metadata["inputs"]["train"]["sha256"] == _sha256(train_source_path)
    assert metadata["inputs"]["test"]["sha256"] == _sha256(test_source_path)
    assert metadata["artifacts"]["train_features"]["sha256"] == _sha256(train_path)
    assert metadata["artifacts"]["test_features"]["sha256"] == _sha256(test_path)
    assert metadata["predictor_columns"] == list(feature_column_names())
    assert metadata["target_columns"] == list(target_column_names())
    assert "fold_role_columns" not in metadata
    assert metadata["artifacts"]["train_features"]["rows"] == len(train)
    assert metadata["artifacts"]["test_features"]["null_counts"] == {
        column: int(count)
        for column, count in pd.read_parquet(test_path).isna().sum().items()
    }
    assert metadata["generator"]["sha256"] == _sha256(
        Path("src/feature_engineering.py")
    )
    assert not Path(metadata["inputs"]["train"]["path"]).is_absolute()


def test_writer_rejects_features_that_change_the_source_contract(
    tmp_path: Path,
) -> None:
    station_id = "station-at"
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train = _station_frame()
    test = _station_frame(start="2024-02-01")
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    train_features = build_feature_frame(train, station_id=station_id)
    test_features = build_feature_frame(test, station_id=station_id)
    train_features.loc[0, "station_id"] = "changed"

    with pytest.raises(ValueError, match="changed original column"):
        write_feature_artifacts(
            train_features,
            test_features,
            station_id=station_id,
            train_source_path=train_path,
            test_source_path=test_path,
            output_dir=tmp_path / "output",
        )


def test_writer_requires_both_source_artifacts(tmp_path: Path) -> None:
    frame = _station_frame()
    features = build_feature_frame(frame, station_id="station-at")

    with pytest.raises(FileNotFoundError):
        write_feature_artifacts(
            features,
            features,
            station_id="station-at",
            train_source_path=tmp_path / "missing-train.parquet",
            test_source_path=tmp_path / "missing-test.parquet",
            output_dir=tmp_path,
            config=DEFAULT_FEATURE_CONFIG,
        )


def test_extract_station_frame_selects_unprefixed_usable_rows() -> None:
    target = _station_frame(4, station_id="target-at")
    upstream = _station_frame(2, station_id="upstream-at", start="2024-01-01T01:00")
    joined = join_station_frames(
        {"target-at": target, "upstream-at": upstream}, target_station_id="target-at"
    )

    result = extract_station_frame(joined, "upstream-at")

    assert list(result.columns) == [
        "timestamp",
        "water_level",
        "imputed",
        "station_id",
        "precipitation",
        "temperature_2m",
    ]
    assert result["timestamp"].tolist() == upstream["timestamp"].tolist()
    assert result["station_id"].tolist() == ["upstream-at", "upstream-at"]
    assert result["water_level"].tolist() == [0.0, 1.0]
    assert result["imputed"].dtype == bool


def test_extract_station_frame_rejects_unknown_station() -> None:
    target = _station_frame(3, station_id="target-at")
    joined = join_station_frames({"target-at": target}, target_station_id="target-at")

    with pytest.raises(ValueError, match="missing joined column"):
        extract_station_frame(joined, "other-at")


def test_write_joined_feature_artifacts_records_manifest_and_hashes(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
            "target-at__water_level": range(5),
            "target-at__station_id": "target-at",
        }
    )
    train_source_path = tmp_path / "all_stations_train.parquet"
    test_source_path = tmp_path / "all_stations_test.parquet"
    source.to_parquet(train_source_path, index=False)
    source.to_parquet(test_source_path, index=False)
    features = source.copy()
    features["target-at__water_level_lag_1h"] = features[
        "target-at__water_level"
    ].shift(1)

    metadata = write_joined_feature_artifacts(
        features,
        features,
        station_ids=["target-at"],
        engineered_station_ids=["target-at"],
        train_source_path=train_source_path,
        test_source_path=test_source_path,
        output_dir=tmp_path / "output",
    )

    train_path = tmp_path / "output" / "all_stations_train_features.parquet"
    test_path = tmp_path / "output" / "all_stations_test_features.parquet"
    metadata_path = tmp_path / "output" / "all_stations_feature_metadata.json"
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata["station_ids"] == ["target-at"]
    assert metadata["engineered_station_ids"] == ["target-at"]
    assert metadata["predictor_columns"][:2] == [
        "target-at__water_level",
        "target-at__imputed",
    ]
    assert metadata["target_columns"][0] == "target-at__target_t_plus_01"
    assert metadata["artifacts"]["train"]["sha256"] == _sha256(train_path)
    assert metadata["artifacts"]["test"]["sha256"] == _sha256(test_path)
    assert metadata["generator"]["sha256"] == _sha256(
        Path("src/feature_engineering.py")
    )
    assert not Path(metadata["inputs"]["train"]["path"]).is_absolute()


def test_target_only_joined_schema_preserves_upstream_raw_columns(
    tmp_path: Path,
) -> None:
    target_id = "target-at"
    upstream_id = "upstream-at"
    target = _station_frame(220, station_id=target_id)
    upstream = _station_frame(220, station_id=upstream_id, water_offset=1000.0)
    source = join_station_frames(
        {target_id: target, upstream_id: upstream}, target_station_id=target_id
    )
    target_source = extract_station_frame(source, target_id)
    target_features = build_feature_frame(target_source, station_id=target_id)
    features = source.copy()
    for column in target_features.columns:
        if column not in target_source.columns:
            features[f"{target_id}__{column}"] = target_features[column].to_numpy()

    train_source_path = tmp_path / "train.parquet"
    test_source_path = tmp_path / "test.parquet"
    source.to_parquet(train_source_path, index=False)
    source.to_parquet(test_source_path, index=False)

    metadata = write_joined_feature_artifacts(
        features,
        features,
        station_ids=[target_id, upstream_id],
        engineered_station_ids=[target_id],
        train_source_path=train_source_path,
        test_source_path=test_source_path,
        output_dir=tmp_path / "output",
    )

    raw_predictors = ["water_level", "imputed", *WEATHER_VARIABLES]
    expected_predictors = [
        f"{target_id}__{column}" for column in feature_column_names()
    ] + [f"{upstream_id}__{column}" for column in raw_predictors]
    expected_targets = [f"{target_id}__{column}" for column in target_column_names()]
    assert metadata["station_ids"] == [target_id, upstream_id]
    assert metadata["engineered_station_ids"] == [target_id]
    assert metadata["predictor_columns"] == expected_predictors
    assert metadata["target_columns"] == expected_targets
    assert f"{upstream_id}__station_id" not in metadata["predictor_columns"]
    assert list(features.columns[: len(source.columns)]) == list(source.columns)
    for column in source.columns:
        assert features[column].equals(source[column])

    upstream_derived = {
        f"{upstream_id}__{column}"
        for column in set(feature_column_names()).difference(raw_predictors)
        | {"target_valid", *target_column_names()}
    }
    assert not upstream_derived.intersection(features.columns)


def test_write_joined_feature_artifacts_rejects_missing_source_files(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")}
    )

    with pytest.raises(FileNotFoundError):
        write_joined_feature_artifacts(
            frame,
            frame,
            station_ids=["target-at"],
            engineered_station_ids=["target-at"],
            train_source_path=tmp_path / "missing-train.parquet",
            test_source_path=tmp_path / "missing-test.parquet",
            output_dir=tmp_path,
        )


def test_write_joined_feature_artifacts_rejects_changed_source_columns(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "target-at__water_level": range(3),
        }
    )
    train_source_path = tmp_path / "train.parquet"
    test_source_path = tmp_path / "test.parquet"
    source.to_parquet(train_source_path, index=False)
    source.to_parquet(test_source_path, index=False)
    features = source.rename(columns={"target-at__water_level": "target-at__level"})

    with pytest.raises(ValueError, match="do not preserve source columns"):
        write_joined_feature_artifacts(
            features,
            features,
            station_ids=["target-at"],
            engineered_station_ids=["target-at"],
            train_source_path=train_source_path,
            test_source_path=test_source_path,
            output_dir=tmp_path / "output",
        )


def test_is_log1p_eligible_base_name_matches_expected_water_level_columns() -> None:
    predictors = feature_column_names()
    expected_eligible = {
        name
        for name in predictors
        if name == "water_level"
        or name.startswith(
            (
                "water_level_lag_",
                "water_level_rolling_mean_",
                "water_level_rolling_min_",
                "water_level_rolling_max_",
            )
        )
    }
    for name in predictors:
        assert is_log1p_eligible_base_name(name) == (name in expected_eligible)

    excluded = set(predictors) - expected_eligible
    assert "water_level_change_24h" in excluded
    assert "water_level_rolling_std_24h" in excluded
    assert "imputed" in excluded
    assert "water_level" in expected_eligible
    assert "water_level_lag_1h" in expected_eligible
    assert "water_level_rolling_mean_6h" in expected_eligible
    assert "water_level_rolling_min_24h" in expected_eligible
    assert "water_level_rolling_max_72h" in expected_eligible


def test_log1p_eligible_columns_strips_station_prefix_and_preserves_order() -> None:
    columns = [
        "station-a__imputed",
        "station-a__water_level_change_24h",
        "station-a__water_level",
        "station-b__water_level_lag_3h",
        "station-a__water_level_rolling_std_6h",
        "water_level_rolling_mean_6h",
    ]

    assert log1p_eligible_columns(columns) == [
        "station-a__water_level",
        "station-b__water_level_lag_3h",
        "water_level_rolling_mean_6h",
    ]
