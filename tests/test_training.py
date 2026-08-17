import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import TARGET_STATION_ID
from src.feature_engineering import feature_column_names
from src.training import (
    JoinedFeatureContract,
    build_feature_subsets,
    load_joined_training_data,
    mlflow_finite_float,
    mlflow_run_series,
    numeric_predictors,
    prediction_preview,
    prepare_model_rows,
    sha256_file,
    summarize_cv_metrics,
    time_series_splits,
    validate_predictions,
)


def test_load_joined_training_data_reads_contract_and_parquet_frames(
    tmp_path: Path,
) -> None:
    station_id = "station-a"
    predictors = ("station-a__water_level", "station-b__water_level")
    targets = ("station-a__target_t_plus_01", "station-a__target_t_plus_02")
    metadata_path = tmp_path / "metadata.json"
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": [station_id],
                "configuration": {"horizon_hours": 2},
                "predictor_columns": list(predictors),
                "target_columns": list(targets),
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "station-a__target_valid": [True, True],
            predictors[0]: [1.0, 2.0],
            predictors[1]: [3.0, 4.0],
            targets[0]: [2.0, 3.0],
            targets[1]: [3.0, 4.0],
        }
    )
    frame.to_parquet(train_path, index=False)
    frame.iloc[[1]].to_parquet(test_path, index=False)

    contract, train, test = load_joined_training_data(
        metadata_path,
        train_path,
        test_path,
        station_id=station_id,
        forecast_horizon_hours=2,
    )

    assert contract == JoinedFeatureContract(
        station_id=station_id,
        target_valid_column="station-a__target_valid",
        predictor_columns=predictors,
        target_columns=targets,
    )
    pd.testing.assert_frame_equal(train, frame)
    pd.testing.assert_frame_equal(test, frame.iloc[[1]].reset_index(drop=True))


def test_load_joined_training_data_rejects_metadata_horizon_mismatch(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": ["station-a"],
                "configuration": {"horizon_hours": 1},
                "predictor_columns": ["station-a__water_level"],
                "target_columns": ["station-a__target_t_plus_01"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="horizon"):
        load_joined_training_data(
            metadata_path,
            tmp_path / "train.parquet",
            tmp_path / "test.parquet",
            station_id="station-a",
            forecast_horizon_hours=2,
        )


def test_load_joined_training_data_reports_missing_artifact_path(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    train_path = tmp_path / "missing-train.parquet"
    test_path = tmp_path / "test.parquet"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": ["station-a"],
                "configuration": {"horizon_hours": 1},
                "predictor_columns": ["station-a__water_level"],
                "target_columns": ["station-a__target_t_plus_01"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"placeholder": [1]}).to_parquet(test_path, index=False)

    with pytest.raises(
        FileNotFoundError,
        match=f"Missing joined feature artifact: {train_path}",
    ):
        load_joined_training_data(
            metadata_path,
            train_path,
            test_path,
            station_id="station-a",
            forecast_horizon_hours=1,
        )


def test_load_joined_training_data_rejects_unengineered_target_station(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": ["station-b"],
                "configuration": {"horizon_hours": 1},
                "predictor_columns": ["station-a__water_level"],
                "target_columns": ["station-a__target_t_plus_01"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not engineered"):
        load_joined_training_data(
            metadata_path,
            tmp_path / "train.parquet",
            tmp_path / "test.parquet",
            station_id="station-a",
            forecast_horizon_hours=1,
        )


def test_load_joined_training_data_rejects_out_of_order_metadata_targets(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": ["station-a"],
                "configuration": {"horizon_hours": 2},
                "predictor_columns": ["station-a__water_level"],
                "target_columns": [
                    "station-a__target_t_plus_02",
                    "station-a__target_t_plus_01",
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target columns"):
        load_joined_training_data(
            metadata_path,
            tmp_path / "train.parquet",
            tmp_path / "test.parquet",
            station_id="station-a",
            forecast_horizon_hours=2,
        )


def test_load_joined_training_data_rejects_empty_predictor_contract(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": ["station-a"],
                "configuration": {"horizon_hours": 1},
                "predictor_columns": [],
                "target_columns": ["station-a__target_t_plus_01"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predictor contract is empty"):
        load_joined_training_data(
            metadata_path,
            tmp_path / "train.parquet",
            tmp_path / "test.parquet",
            station_id="station-a",
            forecast_horizon_hours=1,
        )


def test_load_joined_training_data_rejects_frame_missing_contract_column(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": ["station-a"],
                "configuration": {"horizon_hours": 1},
                "predictor_columns": ["station-a__water_level"],
                "target_columns": ["station-a__target_t_plus_01"],
            }
        ),
        encoding="utf-8",
    )
    incomplete = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, tz="UTC"),
            "station-a__water_level": [1.0],
            "station-a__target_t_plus_01": [2.0],
        }
    )
    complete = incomplete.assign(**{"station-a__target_valid": True})
    incomplete.to_parquet(train_path, index=False)
    complete.to_parquet(test_path, index=False)

    with pytest.raises(
        ValueError,
        match="train artifact is missing required columns.*target_valid",
    ):
        load_joined_training_data(
            metadata_path,
            train_path,
            test_path,
            station_id="station-a",
            forecast_horizon_hours=1,
        )


def test_prepare_model_rows_uses_full_contract_and_sorts_chronologically() -> None:
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=("station-a__water_level", "station-b__water_level"),
        target_columns=("station-a__target_t_plus_01",),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"], utc=True
            ),
            contract.target_valid_column: [True, True, True],
            contract.predictor_columns[0]: [3.0, 1.0, 2.0],
            contract.predictor_columns[1]: [3.0, 1.0, float("nan")],
            contract.target_columns[0]: [4.0, 2.0, 3.0],
        }
    )

    rows = prepare_model_rows(frame, contract, artifact_name="train")

    assert rows["timestamp"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)
    )
    assert rows.index.tolist() == [0, 1]


def test_prepare_model_rows_rejects_null_target_on_target_valid_row() -> None:
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=("station-a__water_level",),
        target_columns=("station-a__target_t_plus_01",),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            contract.target_valid_column: [True],
            contract.predictor_columns[0]: [1.0],
            contract.target_columns[0]: [float("nan")],
        }
    )

    with pytest.raises(ValueError, match="null targets in target-valid rows"):
        prepare_model_rows(frame, contract, artifact_name="train")


def test_prepare_model_rows_rejects_empty_eligible_cohort() -> None:
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=("station-a__water_level",),
        target_columns=("station-a__target_t_plus_01",),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            contract.target_valid_column: [False],
            contract.predictor_columns[0]: [1.0],
            contract.target_columns[0]: [float("nan")],
        }
    )

    with pytest.raises(ValueError, match="train artifact has no eligible model rows"):
        prepare_model_rows(frame, contract, artifact_name="train")


def test_build_feature_subsets_returns_six_contract_ordered_candidates() -> None:
    predictors = (
        "station-a__water_level",
        "station-a__imputed",
        "station-a__precipitation",
        "station-a__temperature_2m",
        "station-a__water_level_lag_24h",
        "station-a__imputed_count_24h",
        "station-a__utc_hour_sin",
        "station-b__water_level",
        "station-b__imputed",
        "station-b__precipitation",
        "station-b__temperature_2m",
    )
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=predictors,
        target_columns=("station-a__target_t_plus_01",),
    )

    subsets = build_feature_subsets(
        contract,
        weather_variables=("precipitation", "temperature_2m"),
    )

    assert list(subsets) == [
        "full",
        "all_station_hydrology_quality_time",
        "raw_all_stations",
        "target_station_full",
        "target_station_hydrology_quality_time",
        "current_water_levels_all_stations",
    ]
    assert subsets == {
        "full": list(predictors),
        "all_station_hydrology_quality_time": [
            predictors[index] for index in (0, 1, 4, 5, 6, 7, 8)
        ],
        "raw_all_stations": [predictors[index] for index in (0, 1, 2, 3, 7, 8, 9, 10)],
        "target_station_full": list(predictors[:7]),
        "target_station_hydrology_quality_time": [
            predictors[index] for index in (0, 1, 4, 5, 6)
        ],
        "current_water_levels_all_stations": [predictors[0], predictors[7]],
    }


def test_build_feature_subsets_preserves_the_project_feature_contract() -> None:
    target_station_id = TARGET_STATION_ID
    raw_columns = ("water_level", "imputed", "precipitation", "temperature_2m")
    neighbor_station_ids = (
        "207019-at",
        "207027-at",
        "207340-at",
        "207068-at",
        "Ennshafen1.Rivermeter-at",
        "207084-at",
        "207357-at",
    )
    predictors = tuple(
        [f"{target_station_id}__{column}" for column in feature_column_names()]
        + [
            f"{station_id}__{column}"
            for station_id in neighbor_station_ids
            for column in raw_columns
        ]
    )
    contract = JoinedFeatureContract(
        station_id=target_station_id,
        target_valid_column=f"{target_station_id}__target_valid",
        predictor_columns=predictors,
        target_columns=(f"{target_station_id}__target_t_plus_01",),
    )

    subsets = build_feature_subsets(
        contract,
        weather_variables=("precipitation", "temperature_2m"),
    )

    assert list(subsets) == [
        "full",
        "all_station_hydrology_quality_time",
        "raw_all_stations",
        "target_station_full",
        "target_station_hydrology_quality_time",
        "current_water_levels_all_stations",
    ]
    assert [len(columns) for columns in subsets.values()] == [81, 55, 32, 53, 41, 8]
    assert len({tuple(columns) for columns in subsets.values()}) == 6
    for columns in subsets.values():
        assert len(columns) == len(set(columns))
        assert columns == [column for column in predictors if column in columns]


def test_build_feature_subsets_rejects_duplicate_predictors() -> None:
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=(
            "station-a__water_level",
            "station-a__water_level",
        ),
        target_columns=("station-a__target_t_plus_01",),
    )

    with pytest.raises(ValueError, match="predictor contract contains duplicates"):
        build_feature_subsets(contract, weather_variables=("precipitation",))


def test_build_feature_subsets_rejects_malformed_predictor_names() -> None:
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=("water_level",),
        target_columns=("station-a__target_t_plus_01",),
    )

    with pytest.raises(ValueError, match="must use '<station>__<feature>' format"):
        build_feature_subsets(contract, weather_variables=("precipitation",))


def test_build_feature_subsets_rejects_empty_candidate() -> None:
    contract = JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=("station-b__water_level",),
        target_columns=("station-a__target_t_plus_01",),
    )

    with pytest.raises(ValueError, match="target_station_full.*empty"):
        build_feature_subsets(contract, weather_variables=("precipitation",))


def test_numeric_predictors_converts_selected_columns_to_float() -> None:
    frame = pd.DataFrame(
        {
            "water": ["1.5", "2.5"],
            "imputed": [False, True],
            "ignored": ["not numeric", "not numeric"],
        }
    )

    numeric = numeric_predictors(frame, ["water", "imputed"])

    assert numeric.to_dict(orient="list") == {
        "water": [1.5, 2.5],
        "imputed": [0.0, 1.0],
    }
    assert all(dtype == float for dtype in numeric.dtypes)


def test_time_series_splits_builds_expanding_folds_with_embargo() -> None:
    splitter, splits, test_size = time_series_splits(
        20,
        initial_train_fraction=0.5,
        n_validation_folds=2,
        embargo_rows=2,
    )

    assert splitter.n_splits == 2
    assert test_size == 4
    assert [
        (train_indices.tolist(), validation_indices.tolist())
        for train_indices, validation_indices in splits
    ] == [
        (list(range(10)), list(range(12, 16))),
        (list(range(14)), list(range(16, 20))),
    ]


def test_validate_predictions_returns_finite_expected_shape() -> None:
    predictions = [[1, 2], [3, 4]]

    validated = validate_predictions(
        predictions,
        expected_rows=2,
        target_columns=("target-1", "target-2"),
        artifact_name="validation fold",
    )

    assert validated.dtype == float
    assert validated.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_summarize_cv_metrics_calculates_literal_fold_statistics() -> None:
    fold_aggregate_metrics = pd.DataFrame(
        {"mae": [1.0, 3.0], "rmse": [2.0, 4.0], "me": [1.0, 3.0], "r2": [2.0, 4.0]}
    )
    fold_horizon_metrics = pd.DataFrame(
        {
            "horizon_hours": [1, 2, 1, 2],
            "mae": [1.0, 2.0, 3.0, 4.0],
            "rmse": [2.0, 3.0, 4.0, 5.0],
            "me": [1.0, 2.0, 3.0, 4.0],
            "r2": [2.0, 3.0, 4.0, 5.0],
        }
    )

    summary = summarize_cv_metrics(
        fold_aggregate_metrics,
        fold_horizon_metrics,
    )

    assert summary == {
        "cv_mae_mean": 2.0,
        "cv_mae_std": 1.0,
        "cv_rmse_mean": 3.0,
        "cv_rmse_std": 1.0,
        "cv_me_mean": 2.0,
        "cv_me_std": 1.0,
        "cv_r2_mean": 3.0,
        "cv_r2_std": 1.0,
        "cv_mae_horizon_01_mean": 2.0,
        "cv_mae_horizon_01_std": 1.0,
        "cv_rmse_horizon_01_mean": 3.0,
        "cv_rmse_horizon_01_std": 1.0,
        "cv_me_horizon_01_mean": 2.0,
        "cv_me_horizon_01_std": 1.0,
        "cv_r2_horizon_01_mean": 3.0,
        "cv_r2_horizon_01_std": 1.0,
        "cv_mae_horizon_02_mean": 3.0,
        "cv_mae_horizon_02_std": 1.0,
        "cv_rmse_horizon_02_mean": 4.0,
        "cv_rmse_horizon_02_std": 1.0,
        "cv_me_horizon_02_mean": 3.0,
        "cv_me_horizon_02_std": 1.0,
        "cv_r2_horizon_02_mean": 4.0,
        "cv_r2_horizon_02_std": 1.0,
    }


def test_prediction_preview_combines_issue_time_actuals_and_predictions() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "target-1": [1.0, 2.0],
            "target-2": [3.0, 4.0],
            "unused": [5.0, 6.0],
        }
    )

    preview = prediction_preview(
        frame,
        [[1.5, 3.5], [2.5, 4.5]],
        target_columns=("target-1", "target-2"),
    )

    assert preview.columns.tolist() == [
        "timestamp",
        "target-1",
        "target-2",
        "prediction_target-1",
        "prediction_target-2",
    ]
    assert preview[
        ["prediction_target-1", "prediction_target-2"]
    ].to_numpy().tolist() == [
        [1.5, 3.5],
        [2.5, 4.5],
    ]


def test_sha256_file_hashes_file_contents(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"hello world")

    assert sha256_file(file_path) == hashlib.sha256(b"hello world").hexdigest()


def test_mlflow_run_series_returns_column_or_null_series() -> None:
    runs = pd.DataFrame({"status": ["FINISHED", "RUNNING"]})

    present = mlflow_run_series(runs, "status")
    missing = mlflow_run_series(runs, "tags.execution_uuid")

    pd.testing.assert_series_equal(present, runs["status"])
    assert missing.isna().all()
    assert missing.index.equals(runs.index)


def test_mlflow_finite_float_converts_or_returns_none() -> None:
    assert mlflow_finite_float("1.5") == 1.5
    assert mlflow_finite_float(2) == 2.0
    assert mlflow_finite_float(None) is None
    assert mlflow_finite_float(pd.NA) is None
    assert mlflow_finite_float(float("nan")) is None
    assert mlflow_finite_float(float("inf")) is None
    assert mlflow_finite_float("not-a-number") is None
