import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor  # type: ignore[import-untyped]

from src import extra_trees
from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID, WEATHER_VARIABLES
from src.dataset import JoinedDataset, load_joined_dataset
from src.extra_trees import (
    build_extra_trees_estimator,
    load_extra_trees_manifest,
    normalize_candidate_key,
    save_extra_trees_manifest,
    score_saved_model,
    select_candidate,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subset": "wide",
                "feature_count": 5,
                "max_depth": None,
                "n_estimators": 800,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "mae_mean": 1.0,
                "rmse_mean": 1.0,
            },
            {
                "subset": "lean",
                "feature_count": 2,
                "max_depth": 32,
                "n_estimators": 400,
                "min_samples_leaf": 8,
                "max_features": "sqrt",
                "mae_mean": 1.0,
                "rmse_mean": 1.0,
            },
        ]
    )


def test_select_candidate_uses_simplicity_tie_breaking() -> None:
    assert select_candidate(_candidate_rows()) == ("lean", 32, 400, 8, "sqrt")


def test_select_candidate_preserves_unbounded_depth_when_it_wins() -> None:
    rows = _candidate_rows()
    rows.loc[1, "rmse_mean"] = 2.0
    assert select_candidate(rows) == ("wide", None, 800, 1, 1.0)


@pytest.mark.parametrize("max_depth", [None, np.nan, "None", "nan", ""])
def test_normalize_candidate_key_round_trips_parameter_sources(
    max_depth: object,
) -> None:
    assert normalize_candidate_key(
        subset="full",
        max_depth=max_depth,
        n_estimators="200",
        min_samples_leaf="4.0",
        max_features="0.5",
    ) == ("full", None, 200, 4, 0.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subset", ""),
        ("max_depth", "invalid"),
        ("n_estimators", "2.5"),
        ("min_samples_leaf", 0),
        ("max_features", "invalid"),
    ],
)
def test_normalize_candidate_key_rejects_invalid_values(
    field: str, value: object
) -> None:
    parameters: dict[str, object] = {
        "subset": "full",
        "max_depth": 16,
        "n_estimators": 200,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    }
    parameters[field] = value
    with pytest.raises(ValueError):
        normalize_candidate_key(**parameters)


@pytest.mark.parametrize(
    ("column", "first", "second", "expected"),
    [
        ("feature_count", 2, 3, "first"),
        ("min_samples_leaf", 8, 1, "first"),
        ("max_depth", 8, 16, "first"),
        ("n_estimators", 200, 400, "first"),
        ("max_features", "sqrt", 0.5, "first"),
        ("max_features", 0.5, 1.0, "first"),
        ("subset", "a", "b", "first"),
    ],
)
def test_select_candidate_breaks_each_tie_axis(
    column: str, first: object, second: object, expected: str
) -> None:
    rows = []
    for subset, value in (("a", first), ("b", second)):
        row = {
            "subset": subset,
            "feature_count": 4,
            "max_depth": 16,
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "max_features": 0.5,
            "mae_mean": 1.0,
            "rmse_mean": 1.0,
        }
        if column == "subset":
            row["subset"] = value
        else:
            row[column] = value
        rows.append(row)
    winner = select_candidate(pd.DataFrame(rows))
    assert winner[0] == ("a" if expected == "first" else "b")


def test_select_candidate_rejects_invalid_or_nonfinite_results() -> None:
    with pytest.raises(ValueError, match="either"):
        select_candidate(_candidate_rows(), metric="r2")
    with pytest.raises(ValueError, match="missing columns"):
        select_candidate(_candidate_rows().drop(columns="max_features"))
    invalid = _candidate_rows()
    invalid.loc[0, "rmse_mean"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        select_candidate(invalid)


def _synthetic_predictors(rows: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "level": rng.uniform(1.0, 50.0, rows),
            "weather": rng.uniform(-10.0, 30.0, rows),
            "quality": rng.uniform(0.0, 1.0, rows),
        }
    )


def test_build_extra_trees_estimator_fits_multioutput_and_is_reproducible() -> None:
    predictors = _synthetic_predictors()
    targets = pd.DataFrame(
        {
            f"target_{horizon:02d}": (horizon + 1.0) * predictors["level"] + horizon
            for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
        }
    )
    first = build_extra_trees_estimator(
        n_estimators=12,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        n_jobs=1,
    )
    second = build_extra_trees_estimator(
        n_estimators=12,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        n_jobs=1,
    )
    assert isinstance(first, ExtraTreesRegressor)
    assert first.bootstrap is False
    first.fit(predictors, targets)
    second.fit(predictors, targets)
    assert first.predict(predictors).shape == (
        len(predictors),
        FORECAST_HORIZON_HOURS,
    )
    np.testing.assert_allclose(first.predict(predictors), second.predict(predictors))


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "n_estimators": 0,
            "max_depth": 8,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
        },
        {
            "n_estimators": 10,
            "max_depth": 0,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
        },
        {
            "n_estimators": 10,
            "max_depth": None,
            "min_samples_leaf": 0,
            "max_features": "sqrt",
            "bootstrap": False,
        },
        {
            "n_estimators": 10,
            "max_depth": 8,
            "min_samples_leaf": 1,
            "max_features": 0.25,
            "bootstrap": False,
        },
        {
            "n_estimators": 10,
            "max_depth": 8,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": "false",
        },
    ],
)
def test_build_extra_trees_estimator_rejects_invalid_parameters(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_extra_trees_estimator(**kwargs)  # type: ignore[arg-type]


def _write_feature_artifacts(root: Path) -> JoinedDataset:
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
    for number, column in enumerate(predictor_columns, start=1):
        frame[column] = number + np.arange(len(frame), dtype=float)
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
                "max_depth": None,
                "n_estimators": 200,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "mae_mean": 1.0,
                "rmse_mean": 1.5,
            },
            {
                "subset": "full",
                "feature_count": 5,
                "max_depth": 16,
                "n_estimators": 400,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "mae_mean": 2.0,
                "rmse_mean": 2.5,
            },
        ]
    )


def _per_horizon_metrics() -> pd.DataFrame:
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
            for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
        ]
    )


def _save(
    dataset: JoinedDataset,
    manifest_path: Path,
    *,
    selected_subset: str = "current_water_levels_all_stations",
    sealed_test_metrics: dict[str, float] | None = None,
    per_horizon_metrics: pd.DataFrame | None = None,
    bootstrap: bool = False,
) -> None:
    save_extra_trees_manifest(
        manifest_path,
        model_path=manifest_path.with_suffix(".joblib"),
        execution_uuid="execution-1",
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
        selected_subset=selected_subset,
        selected_max_depth=None,
        selected_n_estimators=200,
        selected_min_samples_leaf=4,
        selected_max_features="sqrt",
        bootstrap=bootstrap,
        selection_metric="rmse",
        cv_results=_cv_results("current_water_levels_all_stations"),
        sealed_test_metrics=(
            {
                "test_mae": 1.0,
                "test_rmse": 2.0,
                "test_me": 0.25,
                "test_r2": 0.75,
            }
            if sealed_test_metrics is None
            else sealed_test_metrics
        ),
        per_horizon_metrics=(
            _per_horizon_metrics()
            if per_horizon_metrics is None
            else per_horizon_metrics
        ),
        cohort={"train_eligible_rows": len(dataset.train_rows)},
        training={"source_artifact": "train.parquet"},
    )


def test_extra_trees_manifest_round_trips_nullable_parameters(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "extra_trees.json"
    _save(dataset, manifest_path)

    manifest = load_extra_trees_manifest(
        manifest_path,
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
    )

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "1.0"
    assert stored["selected_max_depth"] is None
    assert manifest.selected_max_depth is None
    assert manifest.selected_n_estimators == 200
    assert manifest.selected_min_samples_leaf == 4
    assert manifest.selected_max_features == "sqrt"
    assert stored["bootstrap"] is False
    assert manifest.bootstrap is False
    assert manifest.horizon_metrics["test_mae"].iloc[0] == 2.0


def test_load_extra_trees_manifest_reports_missing_manifest(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    with pytest.raises(FileNotFoundError, match="Run 04_06_train_extra_trees"):
        load_extra_trees_manifest(
            tmp_path / "absent.json",
            contract=dataset.contract,
            feature_subsets=dataset.feature_subsets,
        )


def test_load_extra_trees_manifest_rejects_an_older_schema_version(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "extra_trees.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["schema_version"] = "0.9"
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        load_extra_trees_manifest(
            manifest_path,
            contract=dataset.contract,
            feature_subsets=dataset.feature_subsets,
        )


def test_save_extra_trees_manifest_rejects_invalid_test_records(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "extra_trees.json"
    with pytest.raises(ValueError, match="Sealed-test metrics are missing"):
        _save(dataset, manifest_path, sealed_test_metrics={"test_mae": 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        _save(
            dataset,
            manifest_path,
            sealed_test_metrics={
                "test_mae": np.nan,
                "test_rmse": 2.0,
                "test_me": 0.25,
                "test_r2": 0.75,
            },
        )
    with pytest.raises(ValueError, match="do not match the forecast contract"):
        _save(
            dataset,
            manifest_path,
            per_horizon_metrics=_per_horizon_metrics().iloc[:-1],
        )
    with pytest.raises(ValueError, match="missing columns"):
        _save(
            dataset,
            manifest_path,
            per_horizon_metrics=_per_horizon_metrics().drop(columns="r2"),
        )
    assert not manifest_path.exists()


def test_save_extra_trees_manifest_rejects_unknown_selected_subset(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    with pytest.raises(ValueError, match="is not a known subset"):
        _save(dataset, tmp_path / "extra_trees.json", selected_subset="unknown")


def test_load_extra_trees_manifest_rejects_invalid_bootstrap_and_random_state(
    tmp_path: Path,
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "extra_trees.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, value, message, error_type in (
        ("bootstrap", "false", "bootstrap", TypeError),
        ("random_state", 1, "random_state", ValueError),
    ):
        manifest_path.write_text(json.dumps({**stored, field: value}), encoding="utf-8")
        with pytest.raises(error_type, match=message):
            load_extra_trees_manifest(
                manifest_path,
                contract=dataset.contract,
                feature_subsets=dataset.feature_subsets,
            )


def test_extra_trees_manifest_rejects_contract_mismatches(tmp_path: Path) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "extra_trees.json"
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
            load_extra_trees_manifest(
                manifest_path,
                contract=dataset.contract,
                feature_subsets=dataset.feature_subsets,
            )


class _FakeSavedExtraTrees:
    def __init__(self) -> None:
        self.predictor_values: pd.DataFrame | None = None

    def predict(self, predictors: pd.DataFrame) -> np.ndarray:
        self.predictor_values = predictors.copy()
        return np.ones((len(predictors), FORECAST_HORIZON_HOURS))


def test_score_saved_model_uses_manifest_feature_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "extra_trees.json"
    _save(dataset, manifest_path)
    manifest = load_extra_trees_manifest(
        manifest_path,
        contract=dataset.contract,
        feature_subsets=dataset.feature_subsets,
    )
    fake_model = _FakeSavedExtraTrees()
    monkeypatch.setattr(extra_trees, "load_joblib", lambda _path: fake_model)

    predictions = score_saved_model(
        manifest, manifest_path.with_suffix(".joblib"), dataset.test_rows
    )

    assert predictions.shape == (len(dataset.test_rows), FORECAST_HORIZON_HOURS)
    assert fake_model.predictor_values is not None
    assert list(fake_model.predictor_values.columns) == list(
        manifest.selected_feature_columns
    )
