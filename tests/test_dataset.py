import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_STATION_ID
from src.dataset import JoinedFeatureContract, load_joined_dataset, time_series_splits
from src.feature_engineering import feature_column_names

WEATHER_VARIABLES = ("precipitation", "temperature_2m")


def _predictor_columns(station_ids: Sequence[str]) -> list[str]:
    """Return one contract-shaped predictor per station and base name."""
    return [
        f"{station_id}__{base_name}"
        for station_id in station_ids
        for base_name in (
            "water_level",
            "imputed",
            "precipitation",
            "temperature_2m",
            "water_level_lag_24h",
            "imputed_count_24h",
            "utc_hour_sin",
        )
    ]


def _write_artifacts(
    root: Path,
    *,
    station_id: str = "station-a",
    horizon_hours: int = 2,
    predictor_columns: Sequence[str] | None = None,
    metadata_targets: Sequence[str] | None = None,
    engineered_station_ids: Sequence[str] | None = None,
    metadata_horizon_hours: int | None = None,
    train_rows: int = 24,
    test_rows: int = 6,
    write_train: bool = True,
    drop_train_columns: Sequence[str] = (),
    null_target_on_valid_row: bool = False,
    target_valid: bool = True,
) -> tuple[Path, Path, Path]:
    """Write a synthetic joined metadata/train/test artifact triple."""
    if predictor_columns is None:
        predictor_columns = _predictor_columns([station_id, "station-b"])
    target_columns = [
        f"{station_id}__target_t_plus_{horizon:02d}"
        for horizon in range(1, horizon_hours + 1)
    ]
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": list(
                    engineered_station_ids
                    if engineered_station_ids is not None
                    else [station_id]
                ),
                "configuration": {
                    "horizon_hours": (
                        horizon_hours
                        if metadata_horizon_hours is None
                        else metadata_horizon_hours
                    )
                },
                "predictor_columns": list(predictor_columns),
                "target_columns": list(
                    target_columns if metadata_targets is None else metadata_targets
                ),
            }
        ),
        encoding="utf-8",
    )

    def frame(row_count: int, start: str) -> pd.DataFrame:
        built = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start, periods=row_count, freq="h", tz="UTC"
                ),
                f"{station_id}__target_valid": target_valid,
            }
        )
        for predictor_number, column in enumerate(predictor_columns, start=1):
            built[column] = predictor_number + np.arange(row_count, dtype=float)
            if column.endswith("__imputed"):
                built[column] = False
        for horizon, column in enumerate(target_columns, start=1):
            built[column] = 10.0 * np.arange(row_count, dtype=float) + horizon
        if null_target_on_valid_row:
            built.loc[0, target_columns[0]] = float("nan")
        return built

    train_path = root / "train.parquet"
    test_path = root / "test.parquet"
    if write_train:
        train_frame = frame(train_rows, "2024-01-01")
        # Reverse the rows so chronological ordering is observable in the cohort.
        train_frame = train_frame.iloc[::-1].reset_index(drop=True)
        train_frame.drop(columns=list(drop_train_columns)).to_parquet(
            train_path, index=False
        )
    frame(test_rows, "2024-02-01").to_parquet(test_path, index=False)
    return metadata_path, train_path, test_path


def _load(
    paths: tuple[Path, Path, Path],
    *,
    station_id: str = "station-a",
    forecast_horizon_hours: int = 2,
    initial_train_fraction: float = 0.5,
    n_validation_folds: int = 2,
    embargo_rows: int = 2,
):
    metadata_path, train_path, test_path = paths
    return load_joined_dataset(
        metadata_path,
        train_path,
        test_path,
        station_id=station_id,
        forecast_horizon_hours=forecast_horizon_hours,
        weather_variables=WEATHER_VARIABLES,
        initial_train_fraction=initial_train_fraction,
        n_validation_folds=n_validation_folds,
        embargo_rows=embargo_rows,
    )


def test_load_joined_dataset_builds_the_contract_cohorts_and_provenance(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, train_rows=20, test_rows=6)

    dataset = _load(paths)

    assert dataset.contract == JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=tuple(_predictor_columns(["station-a", "station-b"])),
        target_columns=(
            "station-a__target_t_plus_01",
            "station-a__target_t_plus_02",
        ),
    )
    assert len(dataset.train_rows) == 20
    assert len(dataset.test_rows) == 6
    assert dataset.raw_row_counts == {"train": 20, "test": 6}
    assert dataset.target_water_level_quartile_cutoffs_cm == pytest.approx(
        (5.75, 10.5, 15.25)
    )
    assert dataset.target_water_level_quartile_reference_count == 20
    # The train artifact is written newest-first, so ordering is not incidental.
    assert dataset.train_rows["timestamp"].is_monotonic_increasing
    assert dataset.train_rows.index.tolist() == list(range(20))
    assert dataset.input_hashes == {
        "train_input_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        "test_input_sha256": hashlib.sha256(paths[2].read_bytes()).hexdigest(),
    }


def test_load_joined_dataset_excludes_rows_with_an_incomplete_full_contract(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, train_rows=20)
    train = pd.read_parquet(paths[1])
    # One missing neighbour-station predictor disqualifies the whole issue time.
    train.loc[0, "station-b__precipitation"] = float("nan")
    train.to_parquet(paths[1], index=False)

    dataset = _load(paths)

    assert len(dataset.train_rows) == 19
    assert dataset.raw_row_counts["train"] == 20


def test_load_joined_dataset_builds_expanding_folds_with_an_embargo(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, train_rows=20)

    dataset = _load(paths)

    assert dataset.validation_test_size == 4
    assert [
        (train_indices.tolist(), validation_indices.tolist())
        for train_indices, validation_indices in dataset.folds
    ] == [
        (list(range(10)), list(range(12, 16))),
        (list(range(14)), list(range(16, 20))),
    ]


def test_load_joined_dataset_rejects_too_few_rows_for_the_cv_policy(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, train_rows=6)

    with pytest.raises(ValueError, match="Not enough eligible training rows"):
        _load(paths)


def test_load_joined_dataset_builds_six_contract_ordered_feature_subsets(
    tmp_path: Path,
) -> None:
    predictor_columns = _predictor_columns(["station-a", "station-b"])
    paths = _write_artifacts(tmp_path, predictor_columns=predictor_columns)

    subsets = _load(paths).feature_subsets

    assert list(subsets) == [
        "full",
        "all_station_hydrology_quality_time",
        "raw_all_stations",
        "target_station_full",
        "target_station_hydrology_quality_time",
        "current_water_levels_all_stations",
    ]
    assert subsets == {
        "full": predictor_columns,
        "all_station_hydrology_quality_time": [
            predictor_columns[index] for index in (0, 1, 4, 5, 6, 7, 8, 11, 12, 13)
        ],
        "raw_all_stations": [
            predictor_columns[index] for index in (0, 1, 2, 3, 7, 8, 9, 10)
        ],
        "target_station_full": predictor_columns[:7],
        "target_station_hydrology_quality_time": [
            predictor_columns[index] for index in (0, 1, 4, 5, 6)
        ],
        "current_water_levels_all_stations": [
            predictor_columns[0],
            predictor_columns[7],
        ],
    }


def test_load_joined_dataset_preserves_the_project_feature_contract(
    tmp_path: Path,
) -> None:
    neighbor_station_ids = (
        "207019-at",
        "207027-at",
        "207340-at",
        "207068-at",
        "Ennshafen1.Rivermeter-at",
        "207084-at",
        "207357-at",
    )
    predictor_columns = [
        f"{TARGET_STATION_ID}__{column}" for column in feature_column_names()
    ] + [
        f"{station_id}__{column}"
        for station_id in neighbor_station_ids
        for column in ("water_level", "imputed", "precipitation", "temperature_2m")
    ]
    paths = _write_artifacts(
        tmp_path,
        station_id=TARGET_STATION_ID,
        predictor_columns=predictor_columns,
    )

    subsets = _load(paths, station_id=TARGET_STATION_ID).feature_subsets

    assert [len(columns) for columns in subsets.values()] == [73, 49, 32, 45, 35, 8]
    assert len({tuple(columns) for columns in subsets.values()}) == 6
    for columns in subsets.values():
        assert len(columns) == len(set(columns))
        assert columns == [column for column in predictor_columns if column in columns]


def test_load_joined_dataset_builds_the_target_context_series(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, train_rows=20, test_rows=6)

    context = _load(paths).target_context_series

    assert context.columns.tolist() == [
        "station-a__water_level",
        "station-a__imputed",
    ]
    assert context.index.is_monotonic_increasing
    assert context.index.tz is not None
    # Both artifacts contribute, and every timestamp appears exactly once.
    assert len(context) == 26
    assert not context.index.has_duplicates


def test_load_joined_dataset_quartiles_use_unfiltered_finite_non_imputed_train_values(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, train_rows=8, test_rows=4)
    train = pd.read_parquet(paths[1])
    water_level = "station-a__water_level"
    imputed = "station-a__imputed"
    train[water_level] = [100.0, 200.0, np.nan, np.inf, 300.0, 400.0, 500.0, 600.0]
    train.loc[4, imputed] = True
    # The null/inf and imputed rows remain in the raw artifact but are excluded
    # from the training-reference population.
    train.to_parquet(paths[1], index=False)

    dataset = _load(paths, initial_train_fraction=0.5, n_validation_folds=2)

    assert dataset.target_water_level_quartile_reference_count == 5
    assert dataset.target_water_level_quartile_cutoffs_cm == pytest.approx(
        (200.0, 400.0, 500.0)
    )


def test_load_joined_dataset_rejects_metadata_horizon_mismatch(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, metadata_horizon_hours=1)

    with pytest.raises(ValueError, match="horizon"):
        _load(paths)


def test_load_joined_dataset_rejects_unengineered_target_station(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, engineered_station_ids=["station-b"])

    with pytest.raises(ValueError, match="not engineered"):
        _load(paths)


def test_load_joined_dataset_rejects_out_of_order_metadata_targets(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(
        tmp_path,
        metadata_targets=[
            "station-a__target_t_plus_02",
            "station-a__target_t_plus_01",
        ],
    )

    with pytest.raises(ValueError, match="target columns"):
        _load(paths)


def test_load_joined_dataset_rejects_empty_predictor_contract(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, predictor_columns=[])

    with pytest.raises(ValueError, match="predictor contract is empty"):
        _load(paths)


def test_load_joined_dataset_reports_missing_artifact_path(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, write_train=False)

    with pytest.raises(
        FileNotFoundError, match=f"Missing joined feature artifact: {paths[1]}"
    ):
        _load(paths)


def test_load_joined_dataset_rejects_frame_missing_contract_column(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, drop_train_columns=["station-a__target_valid"])

    with pytest.raises(
        ValueError, match="train artifact is missing required columns.*target_valid"
    ):
        _load(paths)


def test_load_joined_dataset_rejects_null_target_on_target_valid_row(
    tmp_path: Path,
) -> None:
    paths = _write_artifacts(tmp_path, null_target_on_valid_row=True)

    with pytest.raises(ValueError, match="null targets in target-valid rows"):
        _load(paths)


def test_load_joined_dataset_rejects_empty_eligible_cohort(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, target_valid=False)

    with pytest.raises(ValueError, match="train artifact has no eligible model rows"):
        _load(paths)


def test_load_joined_dataset_rejects_duplicate_predictors(tmp_path: Path) -> None:
    duplicated = _predictor_columns(["station-a", "station-b"])
    paths = _write_artifacts(tmp_path, predictor_columns=[*duplicated, duplicated[0]])

    with pytest.raises(ValueError, match="predictor contract contains duplicates"):
        _load(paths)


def test_load_joined_dataset_rejects_malformed_predictor_names(tmp_path: Path) -> None:
    paths = _write_artifacts(
        tmp_path,
        predictor_columns=[*_predictor_columns(["station-a"]), "water_level"],
    )

    with pytest.raises(ValueError, match="must use '<station>__<feature>' format"):
        _load(paths)


def test_load_joined_dataset_rejects_empty_feature_subset(tmp_path: Path) -> None:
    paths = _write_artifacts(
        tmp_path, predictor_columns=_predictor_columns(["station-b"])
    )

    with pytest.raises(ValueError, match="target_station_full.*empty"):
        _load(paths)


def test_time_series_splits_builds_expanding_folds_with_an_embargo() -> None:
    folds, validation_test_size = time_series_splits(
        20, initial_train_fraction=0.5, n_validation_folds=2, embargo_rows=2
    )

    assert validation_test_size == 4
    assert [
        (train_indices.tolist(), validation_indices.tolist())
        for train_indices, validation_indices in folds
    ] == [
        (list(range(10)), list(range(12, 16))),
        (list(range(14)), list(range(16, 20))),
    ]


def test_time_series_splits_rejects_too_few_rows_for_the_cv_policy() -> None:
    with pytest.raises(ValueError, match="Not enough eligible training rows"):
        time_series_splits(
            6, initial_train_fraction=0.5, n_validation_folds=2, embargo_rows=2
        )


def test_load_joined_dataset_requires_target_station_context_columns(
    tmp_path: Path,
) -> None:
    predictor_columns = [
        column
        for column in _predictor_columns(["station-a", "station-b"])
        if column != "station-a__imputed"
    ]
    paths = _write_artifacts(tmp_path, predictor_columns=predictor_columns)

    with pytest.raises(ValueError, match="missing target-station context columns"):
        _load(paths)
