import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor  # type: ignore[import-untyped]

from src import xgboost_model
from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID, WEATHER_VARIABLES
from src.dataset import JoinedDataset, load_joined_dataset
from src.xgboost_model import (
    build_xgboost_estimator,
    load_xgboost_manifest,
    save_xgboost_manifest,
    score_saved_model,
    select_candidate,
)


def test_select_candidate_ranks_by_metric_then_ties() -> None:
    winner = select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Wide",
                    "feature_count": 10,
                    "max_depth": 3,
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 1.0,
                    "mae_mean": 2.0,
                    "rmse_mean": 2.0,
                },
                {
                    "subset": "Lean",
                    "feature_count": 2,
                    "max_depth": 4,
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 10.0,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    )
    assert winner == ("Lean", 4, 200, 0.05, 0.8, 0.8, 10.0)


class _FakeSavedXgboost:
    def __init__(self) -> None:
        self.predictor_values: pd.DataFrame | None = None

    def predict(self, predictors: pd.DataFrame) -> np.ndarray:
        self.predictor_values = predictors.copy()
        row_numbers = np.arange(len(predictors), dtype=float)[:, None]
        horizon_numbers = np.arange(1, FORECAST_HORIZON_HOURS + 1, dtype=float)[None, :]
        return row_numbers * 10.0 + horizon_numbers + 100.0


def _write_feature_artifacts(root: Path) -> JoinedDataset:
    """Write a synthetic joined artifact triple and load it as a dataset."""
    processed_dir = root / "data" / "processed" / "joined"
    processed_dir.mkdir(parents=True)
    station_id = TARGET_STATION_ID
    predictor_columns = [
        f"{station_id}__water_level",
        f"{station_id}__imputed",
        f"{station_id}__precipitation",
        f"{station_id}__temperature_2m",
        "station-b__water_level",
    ]
    target_columns = [
        f"{station_id}__target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    metadata_path = processed_dir / "all_stations_feature_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": [station_id],
                "configuration": {"horizon_hours": FORECAST_HORIZON_HOURS},
                "predictor_columns": predictor_columns,
                "target_columns": target_columns,
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03 00:00", "2024-01-01 00:00", "2024-01-02 00:00"],
                utc=True,
            ),
            f"{station_id}__target_valid": True,
        }
    )
    for predictor_number, column in enumerate(predictor_columns, start=1):
        frame[column] = predictor_number + np.arange(len(frame), dtype=float)
    frame[f"{station_id}__imputed"] = False
    for horizon, column in enumerate(target_columns, start=1):
        frame[column] = 10.0 * np.arange(len(frame), dtype=float) + horizon
    train_path = processed_dir / "all_stations_train_features.parquet"
    test_path = processed_dir / "all_stations_test_features.parquet"
    frame.to_parquet(train_path, index=False)
    frame.to_parquet(test_path, index=False)
    return load_joined_dataset(
        metadata_path,
        train_path,
        test_path,
        station_id=station_id,
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
        weather_variables=WEATHER_VARIABLES,
        initial_train_fraction=0.5,
        n_validation_folds=2,
        embargo_rows=0,
    )


def _cv_results(selected_subset: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subset": selected_subset,
                "feature_count": 2,
                "max_depth": 3,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "mae_mean": 1.0,
                "rmse_mean": 1.5,
            },
            {
                "subset": "full",
                "feature_count": 5,
                "max_depth": 6,
                "n_estimators": 300,
                "learning_rate": 0.2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 0.1,
                "mae_mean": 2.0,
                "rmse_mean": 2.5,
            },
        ]
    )


def _save(
    dataset: JoinedDataset,
    manifest_path: Path,
    *,
    selected_subset: str = "current_water_levels_all_stations",
) -> None:
    save_xgboost_manifest(
        manifest_path,
        model_path=manifest_path.with_suffix(".joblib"),
        execution_uuid="execution-1",
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
        selected_subset=selected_subset,
        selected_max_depth=3,
        selected_n_estimators=100,
        selected_learning_rate=0.1,
        selected_subsample=0.8,
        selected_colsample_bytree=0.8,
        selected_reg_lambda=1.0,
    )


def test_xgboost_manifest_round_trips_the_model_contract(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "xgboost.json"

    _save(dataset, manifest_path)
    manifest = load_xgboost_manifest(
        manifest_path,
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
    )

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "3.0"
    assert not {
        "selection_metric",
        "tie_breaking",
        "full_feature_columns",
        "feature_subsets",
        "cv_results",
        "sealed_test_metrics",
        "horizon_metrics",
        "regime_definition",
        "regime_metrics",
        "common_cohort_eligibility",
        "training",
    }.intersection(stored)
    assert stored["selected_max_depth"] == 3
    assert stored["selected_n_estimators"] == 100
    assert manifest.execution_uuid == "execution-1"
    assert manifest.selected_subset == "current_water_levels_all_stations"
    assert manifest.selected_max_depth == 3
    assert manifest.selected_n_estimators == 100
    assert manifest.selected_learning_rate == 0.1
    assert manifest.selected_subsample == 0.8
    assert manifest.selected_colsample_bytree == 0.8
    assert manifest.selected_reg_lambda == 1.0
    assert manifest.selected_feature_columns == tuple(
        dataset.feature_subsets["current_water_levels_all_stations"]
    )
    assert manifest.target_columns == dataset.contract.target_columns


def test_load_xgboost_manifest_reports_missing_manifest(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Run 04_04_train_xgboost.ipynb through its sealed-test cell",
    ):
        load_xgboost_manifest(
            tmp_path / "absent.json",
            contract=dataset.contract,
            feature_subsets=dataset.feature_subsets,
        )


def test_load_xgboost_manifest_rejects_an_older_schema_version(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "xgboost.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["schema_version"] = "0.9"
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        load_xgboost_manifest(
            manifest_path,
            contract=dataset.contract,
            feature_subsets=dataset.feature_subsets,
        )


def test_load_xgboost_manifest_rejects_contract_mismatches(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "xgboost.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))

    for field, value, message in (
        ("station_id", "other-station", "station_id"),
        ("forecast_horizon_hours", FORECAST_HORIZON_HOURS + 1, "forecast horizon"),
        ("target_columns", ["only-one"], "target contract"),
        ("selected_subset", "unknown-subset", "selected subset"),
        ("selected_feature_columns", ["only-one"], "selected features"),
    ):
        manifest_path.write_text(json.dumps({**stored, field: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_xgboost_manifest(
                manifest_path,
                contract=dataset.contract,
                feature_subsets=dataset.feature_subsets,
            )


def test_save_xgboost_manifest_rejects_an_unknown_selected_subset(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)

    with pytest.raises(ValueError, match="is not a known subset"):
        _save(dataset, tmp_path / "xgboost.json", selected_subset="unknown-subset")


def test_score_saved_model_scores_the_cohort_with_the_manifest_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "xgboost.json"
    _save(dataset, manifest_path)
    manifest = load_xgboost_manifest(
        manifest_path,
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
    )
    fake_model = _FakeSavedXgboost()
    monkeypatch.setattr(xgboost_model, "load_joblib", lambda _path: fake_model)

    predictions = score_saved_model(
        manifest, manifest_path.with_suffix(".joblib"), dataset.test_rows
    )

    assert predictions.shape == (len(dataset.test_rows), FORECAST_HORIZON_HOURS)
    assert fake_model.predictor_values is not None
    assert list(fake_model.predictor_values.columns) == list(
        manifest.selected_feature_columns
    )


def _synthetic_predictors(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "station-at__water_level": rng.uniform(1.0, 50.0, rows),
            "station-at__water_level_change_24h": rng.uniform(-5.0, 5.0, rows),
            "station-at__temperature_2m": rng.uniform(-10.0, 30.0, rows),
        }
    )


def test_build_xgboost_estimator_fits_and_predicts_the_right_shape() -> None:
    predictors = _synthetic_predictors()
    targets = pd.DataFrame(
        {
            "target_1": 2.0 * predictors["station-at__water_level"] + 1.0,
            "target_2": 3.0 * predictors["station-at__water_level"] + 2.0,
        }
    )
    estimator = build_xgboost_estimator(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=10,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
    )
    estimator.fit(predictors, targets)

    assert isinstance(estimator, XGBRegressor)
    assert estimator.get_params()["multi_strategy"] == "multi_output_tree"
    predictions = estimator.predict(predictors)

    assert predictions.shape == (len(predictors), 2)
    assert np.isfinite(predictions).all()


def test_build_xgboost_estimator_is_reproducible_with_a_fixed_random_state() -> None:
    predictors = _synthetic_predictors()
    targets = pd.DataFrame(
        {"target_1": 2.0 * predictors["station-at__water_level"] + 1.0}
    )
    first = build_xgboost_estimator(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=7,
    )
    first.fit(predictors, targets)
    second = build_xgboost_estimator(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=7,
    )
    second.fit(predictors, targets)

    np.testing.assert_allclose(first.predict(predictors), second.predict(predictors))
