import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import ridge
from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID, WEATHER_VARIABLES
from src.dataset import JoinedDataset, load_joined_dataset
from src.ridge import (
    load_ridge_manifest,
    save_ridge_manifest,
    score_saved_model,
    select_candidate,
)


def test_select_candidate_applies_all_tie_breakers() -> None:
    assert select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Wide",
                    "alpha": 0.1,
                    "feature_count": 10,
                    "mae_mean": 2.0,
                    "rmse_mean": 2.0,
                },
                {
                    "subset": "Lean",
                    "alpha": 0.1,
                    "feature_count": 2,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    ) == ("Lean", 0.1)
    assert select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Wide",
                    "alpha": 0.01,
                    "feature_count": 10,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
                {
                    "subset": "Lean",
                    "alpha": 0.1,
                    "feature_count": 2,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    ) == ("Lean", 0.1)
    assert select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Same",
                    "alpha": 1.0,
                    "feature_count": 2,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
                {
                    "subset": "Same",
                    "alpha": 0.1,
                    "feature_count": 2,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    ) == ("Same", 0.1)
    assert select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Zulu",
                    "alpha": 0.1,
                    "feature_count": 2,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
                {
                    "subset": "Alpha",
                    "alpha": 0.1,
                    "feature_count": 2,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    ) == ("Alpha", 0.1)
    different_metrics = pd.DataFrame(
        [
            {
                "subset": "Low MAE",
                "alpha": 0.1,
                "feature_count": 2,
                "mae_mean": 1.0,
                "rmse_mean": 4.0,
            },
            {
                "subset": "Low RMSE",
                "alpha": 0.1,
                "feature_count": 2,
                "mae_mean": 2.0,
                "rmse_mean": 1.0,
            },
        ]
    )
    assert select_candidate(different_metrics) == ("Low RMSE", 0.1)
    assert select_candidate(different_metrics, metric="mae") == ("Low MAE", 0.1)


class _FakeSavedRidge:
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
                "alpha": 0.1,
                "mae_mean": 1.0,
                "rmse_mean": 1.5,
            },
            {
                "subset": "full",
                "feature_count": 5,
                "alpha": 1.0,
                "mae_mean": 2.0,
                "rmse_mean": 2.5,
            },
        ]
    )


def _per_horizon_metrics(horizons: int = FORECAST_HORIZON_HOURS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "station_id": TARGET_STATION_ID,
                "horizon_hours": horizon,
                "target": f"target_{horizon:02d}",
                "mae": 1.0 + horizon,
                "rmse": 2.0 + horizon,
                "me": 0.25,
                "r2": 0.75,
            }
            for horizon in range(1, horizons + 1)
        ]
    )


def _save(
    dataset: JoinedDataset,
    manifest_path: Path,
    *,
    selected_subset: str = "current_water_levels_all_stations",
    per_horizon_metrics: pd.DataFrame | None = None,
    sealed_test_metrics: dict[str, float] | None = None,
) -> None:
    save_ridge_manifest(
        manifest_path,
        model_path=manifest_path.with_suffix(".joblib"),
        execution_uuid="execution-1",
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
        selected_subset=selected_subset,
        selected_alpha=0.1,
        selection_metric="rmse",
        cv_results=_cv_results(selected_subset),
        sealed_test_metrics=sealed_test_metrics
        or {"test_mae": 1.0, "test_rmse": 2.0, "test_me": 0.25, "test_r2": 0.75},
        per_horizon_metrics=(
            _per_horizon_metrics()
            if per_horizon_metrics is None
            else per_horizon_metrics
        ),
        cohort={"train_eligible_rows": len(dataset.train_rows)},
        training={"source_artifact": "train.parquet"},
    )


def test_ridge_manifest_round_trips_the_selection_and_sealed_test_record(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "ridge.json"

    _save(dataset, manifest_path)
    manifest = load_ridge_manifest(
        manifest_path,
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
    )

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "2.0"
    assert stored["tie_breaking"][0] == "lowest aggregate CV RMSE"
    assert manifest.execution_uuid == "execution-1"
    assert manifest.selected_subset == "current_water_levels_all_stations"
    assert manifest.selected_alpha == 0.1
    assert manifest.selection_metric == "rmse"
    assert manifest.selected_feature_columns == tuple(
        dataset.feature_subsets["current_water_levels_all_stations"]
    )
    assert manifest.target_columns == dataset.contract.target_columns
    assert manifest.cv_results["subset"].tolist() == [
        "current_water_levels_all_stations",
        "full",
    ]
    assert manifest.sealed_test_metrics == {
        "test_mae": 1.0,
        "test_rmse": 2.0,
        "test_me": 0.25,
        "test_r2": 0.75,
    }
    assert manifest.horizon_metrics["horizon_hours"].tolist() == list(
        range(1, FORECAST_HORIZON_HOURS + 1)
    )
    assert manifest.horizon_metrics["test_mae"].iloc[0] == 2.0
    assert (
        manifest.horizon_metrics["test_rmse"].iloc[-1] == 2.0 + FORECAST_HORIZON_HOURS
    )


def test_load_ridge_manifest_reports_missing_manifest(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)

    with pytest.raises(
        FileNotFoundError, match="Run 04_train_ridge.ipynb through its sealed-test cell"
    ):
        load_ridge_manifest(
            tmp_path / "absent.json",
            contract=dataset.contract,
            feature_subsets=dataset.feature_subsets,
        )


def test_load_ridge_manifest_rejects_an_older_schema_version(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "ridge.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["schema_version"] = "1.1"
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        load_ridge_manifest(
            manifest_path,
            contract=dataset.contract,
            feature_subsets=dataset.feature_subsets,
        )


def test_load_ridge_manifest_rejects_contract_mismatches(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "ridge.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))

    for field, value, message in (
        ("station_id", "other-station", "station_id"),
        ("forecast_horizon_hours", FORECAST_HORIZON_HOURS + 1, "forecast horizon"),
        ("full_feature_columns", ["only-one"], "full feature contract"),
        ("target_columns", ["only-one"], "target contract"),
        ("selected_subset", "unknown-subset", "selected subset"),
        ("selected_feature_columns", ["only-one"], "selected features"),
    ):
        manifest_path.write_text(json.dumps({**stored, field: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_ridge_manifest(
                manifest_path,
                contract=dataset.contract,
                feature_subsets=dataset.feature_subsets,
            )


def test_save_ridge_manifest_rejects_an_incomplete_sealed_test_record(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "ridge.json"

    with pytest.raises(ValueError, match="Sealed-test metrics are missing"):
        _save(
            dataset,
            manifest_path,
            sealed_test_metrics={"test_mae": 1.0, "test_rmse": 2.0},
        )
    with pytest.raises(ValueError, match="non-finite"):
        _save(
            dataset,
            manifest_path,
            sealed_test_metrics={
                "test_mae": float("nan"),
                "test_rmse": 2.0,
                "test_me": 0.25,
                "test_r2": 0.75,
            },
        )
    with pytest.raises(ValueError, match="do not match the forecast contract"):
        _save(
            dataset,
            manifest_path,
            per_horizon_metrics=_per_horizon_metrics(FORECAST_HORIZON_HOURS - 1),
        )
    # Nothing is written until every check passes.
    assert not manifest_path.exists()


def test_save_ridge_manifest_rejects_an_unknown_selected_subset(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)

    with pytest.raises(ValueError, match="is not a known subset"):
        _save(dataset, tmp_path / "ridge.json", selected_subset="unknown-subset")


def test_score_saved_model_scores_the_cohort_with_the_manifest_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "ridge.json"
    _save(dataset, manifest_path)
    manifest = load_ridge_manifest(
        manifest_path,
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
    )
    fake_model = _FakeSavedRidge()
    monkeypatch.setattr(ridge, "load_joblib", lambda _path: fake_model)

    predictions = score_saved_model(
        manifest, manifest_path.with_suffix(".joblib"), dataset.test_rows
    )

    assert predictions.shape == (len(dataset.test_rows), FORECAST_HORIZON_HOURS)
    assert fake_model.predictor_values is not None
    assert list(fake_model.predictor_values.columns) == list(
        manifest.selected_feature_columns
    )
