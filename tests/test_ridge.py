import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from src import ridge
from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID, WEATHER_VARIABLES
from src.dataset import JoinedDataset, load_joined_dataset
from src.ridge import (
    build_ridge_estimator,
    load_ridge_manifest,
    save_ridge_manifest,
    score_saved_model,
    select_candidate,
)


def test_select_candidate_returns_ridge_triple_with_ridges_exact_tie_break_set() -> (
    None
):
    """Regression test: the shared model_selection tie-break ordering/columns
    generically is covered by tests/test_model_selection.py. This confirms only
    that Ridge's wrapper still returns the (subset, alpha, log1p) triple using
    Ridge's exact tie-break columns (feature_count, log1p, alpha, subset).
    """
    assert select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Wide",
                    "alpha": 0.1,
                    "feature_count": 10,
                    "log1p": False,
                    "mae_mean": 2.0,
                    "rmse_mean": 2.0,
                },
                {
                    "subset": "Lean",
                    "alpha": 0.1,
                    "feature_count": 2,
                    "log1p": False,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    ) == ("Lean", 0.1, False)


def test_select_candidate_prefers_log1p_false_on_tie() -> None:
    assert select_candidate(
        pd.DataFrame(
            [
                {
                    "subset": "Same",
                    "alpha": 10.0,
                    "feature_count": 2,
                    "log1p": False,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
                {
                    "subset": "Same",
                    "alpha": 0.01,
                    "feature_count": 2,
                    "log1p": True,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    ) == ("Same", 10.0, False)


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
                "alpha": 0.1,
                "log1p": False,
                "mae_mean": 1.0,
                "rmse_mean": 1.5,
            },
            {
                "subset": "full",
                "feature_count": 5,
                "alpha": 1.0,
                "log1p": False,
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
    selected_log1p: bool = False,
) -> None:
    save_ridge_manifest(
        manifest_path,
        model_path=manifest_path.with_suffix(".joblib"),
        execution_uuid="execution-1",
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
        selected_subset=selected_subset,
        selected_alpha=0.1,
        selected_log1p=selected_log1p,
    )


def test_ridge_manifest_round_trips_the_model_contract(
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
    assert stored["schema_version"] == "5.0"
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
    assert stored["selected_log1p"] is False
    assert manifest.execution_uuid == "execution-1"
    assert manifest.selected_subset == "current_water_levels_all_stations"
    assert manifest.selected_alpha == 0.1
    assert manifest.selected_log1p is False
    assert manifest.selected_feature_columns == tuple(
        dataset.feature_subsets["current_water_levels_all_stations"]
    )
    assert manifest.target_columns == dataset.contract.target_columns


def test_load_ridge_manifest_reports_missing_manifest(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Run 04_02_train_ridge.ipynb through its sealed-test cell",
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
    stored["schema_version"] = "2.0"
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


def _synthetic_predictors(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "station-at__water_level": rng.uniform(1.0, 50.0, rows),
            "station-at__water_level_change_24h": rng.uniform(-5.0, 5.0, rows),
            "station-at__temperature_2m": rng.uniform(-10.0, 30.0, rows),
        }
    )


def test_build_ridge_estimator_log1p_round_trips_to_raw_scale() -> None:
    predictors = _synthetic_predictors()
    targets = pd.DataFrame(
        {
            "target_1": 2.0 * predictors["station-at__water_level"] + 1.0,
            "target_2": 3.0 * predictors["station-at__water_level"] + 2.0,
        }
    )
    feature_columns = list(predictors.columns)

    estimator = build_ridge_estimator(feature_columns, alpha=1.0, log1p=True)
    estimator.fit(predictors, targets)
    predictions = estimator.predict(predictors)

    assert predictions.shape == (len(predictors), 2)
    assert np.isfinite(predictions).all()


def test_build_ridge_estimator_log1p_tolerates_negative_ineligible_columns() -> None:
    predictors = _synthetic_predictors()
    predictors["station-at__water_level_change_24h"] = np.linspace(
        -20.0, -1.0, len(predictors)
    )
    targets = pd.DataFrame({"target_1": predictors["station-at__water_level"] + 5.0})
    feature_columns = list(predictors.columns)

    estimator = build_ridge_estimator(feature_columns, alpha=1.0, log1p=True)
    estimator.fit(predictors, targets)
    predictions = estimator.predict(predictors)

    assert np.isfinite(predictions).all()


def test_build_ridge_estimator_without_log1p_matches_hand_built_pipeline() -> None:
    predictors = _synthetic_predictors()
    targets = pd.DataFrame(
        {"target_1": 2.0 * predictors["station-at__water_level"] + 1.0}
    )
    feature_columns = list(predictors.columns)

    estimator = build_ridge_estimator(feature_columns, alpha=1.0, log1p=False)
    estimator.fit(predictors, targets)
    predictions = estimator.predict(predictors)

    reference = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    reference.fit(predictors[feature_columns], targets)
    reference_predictions = reference.predict(predictors[feature_columns])

    np.testing.assert_allclose(predictions, reference_predictions)
