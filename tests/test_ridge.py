import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src import ridge
from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID, WEATHER_VARIABLES
from src.ridge import (
    load_and_score_saved_model,
    select_candidate,
    select_execution,
    validate_execution,
)
from src.training import (
    build_feature_subsets,
    load_joined_training_data,
    prepare_model_rows,
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


def _comparison_runs(
    execution_uuid: str,
    end_time: str,
    *,
    selected_subset: str = "lean",
    selected_alpha: str = "0.1",
    selected_test_mae: float = 7.5,
    duplicate_candidate: bool = False,
    lean_cv_mae: float = 1.0,
) -> pd.DataFrame:
    rows = []
    candidates = [
        ("lean", "0.1", 2, lean_cv_mae),
        ("wide", "1.0", 5, lean_cv_mae + 1.0),
    ]
    for candidate_number, (subset, alpha, feature_count, cv_mae) in enumerate(
        candidates, start=1
    ):
        rows.append(
            {
                "run_id": f"{execution_uuid}-candidate-{candidate_number}",
                "status": "FINISHED",
                "end_time": end_time,
                "tags.run_type": "candidate_parent",
                "tags.execution_uuid": execution_uuid,
                "tags.subset": subset,
                "params.alpha": alpha,
                "params.feature_count": str(feature_count),
                "metrics.cv_mae_mean": cv_mae,
                "metrics.cv_mae_std": 0.1,
                "metrics.cv_rmse_mean": cv_mae + 0.2,
                "metrics.cv_rmse_std": 0.2,
                "metrics.cv_me_mean": 0.1,
                "metrics.cv_me_std": 0.05,
                "metrics.cv_r2_mean": 0.5,
                "metrics.cv_r2_std": 0.1,
            }
        )
    if duplicate_candidate:
        rows.append(rows[0].copy())

    sealed_test_row = {
        "run_id": f"{execution_uuid}-sealed-test",
        "status": "FINISHED",
        "end_time": end_time,
        "tags.run_type": "sealed_test",
        "tags.execution_uuid": execution_uuid,
        "tags.subset": selected_subset,
        "params.alpha": selected_alpha,
        "metrics.test_mae": selected_test_mae,
        "metrics.test_rmse": selected_test_mae + 1.0,
        "metrics.test_me": 0.25,
        "metrics.test_r2": 0.75,
    }
    for horizon in range(1, FORECAST_HORIZON_HOURS + 1):
        sealed_test_row[f"metrics.test_mae_horizon_{horizon:02d}"] = (
            selected_test_mae + horizon
        )
        sealed_test_row[f"metrics.test_rmse_horizon_{horizon:02d}"] = (
            selected_test_mae + horizon + 1.0
        )
        sealed_test_row[f"metrics.test_me_horizon_{horizon:02d}"] = 0.25
        sealed_test_row[f"metrics.test_r2_horizon_{horizon:02d}"] = 0.75
    rows.append(sealed_test_row)
    return pd.DataFrame(rows)


def test_select_execution_skips_newer_invalid_execution() -> None:
    runs = pd.concat(
        [
            _comparison_runs(
                "new-execution",
                "2025-02-01T00:00:00Z",
                duplicate_candidate=True,
            ),
            _comparison_runs("old-execution", "2025-01-01T00:00:00Z"),
        ],
        ignore_index=True,
    )

    selected_execution = select_execution(runs)

    assert selected_execution["execution_uuid"] == "old-execution"
    assert len(cast(pd.DataFrame, selected_execution["candidate_table"])) == 2


def test_select_execution_reports_missing_mlflow_execution() -> None:
    with pytest.raises(ValueError, match="Run 04_train_ridge.ipynb"):
        select_execution(pd.DataFrame())


def test_select_execution_uses_cv_for_candidates_and_test_for_selected_only() -> None:
    runs = _comparison_runs(
        "valid-execution",
        "2025-01-01T00:00:00Z",
        selected_subset="wide",
        selected_alpha="1.0",
        selected_test_mae=7.5,
        lean_cv_mae=1.0,
    )

    selected_execution = select_execution(runs)
    candidate_table = cast(pd.DataFrame, selected_execution["candidate_table"])
    selected_candidate = cast(pd.Series, selected_execution["selected_candidate"])
    sealed_test_metrics = cast(dict, selected_execution["sealed_test_metrics"])

    assert candidate_table["subset"].tolist() == ["lean", "wide"]
    assert candidate_table["cv_mae_mean"].tolist() == [1.0, 2.0]
    assert "test_mae" not in candidate_table.columns
    assert selected_candidate["subset"] == "wide"
    assert sealed_test_metrics["test_mae"] == 7.5


def test_select_execution_ranks_by_configured_rmse_metric() -> None:
    runs = _comparison_runs("rmse-ranked-execution", "2025-01-01T00:00:00Z")
    runs.loc[runs["tags.subset"].eq("lean"), "metrics.cv_rmse_mean"] = 5.0
    runs.loc[runs["tags.subset"].eq("wide"), "metrics.cv_rmse_mean"] = 1.0

    selected_execution = select_execution(runs)
    candidate_table = cast(pd.DataFrame, selected_execution["candidate_table"])

    assert candidate_table["subset"].tolist() == ["wide", "lean"]


def test_select_execution_skips_legacy_metric_schema() -> None:
    legacy_runs = _comparison_runs("legacy-execution", "2025-01-01T00:00:00Z")
    legacy_metric_columns = [
        column
        for column in legacy_runs.columns
        if any(
            metric in str(column) for metric in ("cv_me", "cv_r2", "test_me", "test_r2")
        )
    ]
    legacy_runs = legacy_runs.drop(columns=legacy_metric_columns)

    with pytest.raises(ValueError, match="No valid Ridge MLflow execution exists"):
        select_execution(legacy_runs)


def test_validate_execution_rejects_sealed_test_without_candidate_match() -> None:
    runs = _comparison_runs("mismatched-execution", "2025-01-01T00:00:00Z")
    sealed_test_run = runs.loc[runs["tags.run_type"].eq("sealed_test")].iloc[0].copy()
    sealed_test_run["tags.subset"] = "unknown-subset"

    with pytest.raises(ValueError, match="does not match a candidate-parent row"):
        validate_execution(sealed_test_run, runs)


class _FakeSavedRidge:
    def __init__(self) -> None:
        self.predictor_values: pd.DataFrame | None = None

    def predict(self, predictors: pd.DataFrame) -> np.ndarray:
        self.predictor_values = predictors.copy()
        row_numbers = np.arange(len(predictors), dtype=float)[:, None]
        horizon_numbers = np.arange(1, FORECAST_HORIZON_HOURS + 1, dtype=float)[None, :]
        return row_numbers * 10.0 + horizon_numbers + 100.0


def _write_comparison_artifacts(
    root: Path,
    *,
    execution_uuid: str,
    manifest_execution_uuid: str | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    processed_dir = root / "data" / "processed" / "joined"
    processed_dir.mkdir(parents=True)
    model_dir = root / "models"
    model_dir.mkdir()
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

    selected_feature_columns = predictor_columns[:2]
    manifest = {
        "schema_version": "1.1",
        "execution_uuid": manifest_execution_uuid or execution_uuid,
        "station_id": station_id,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "full_feature_columns": predictor_columns,
        "selected_subset": "lean",
        "feature_subset": "lean",
        "selected_feature_columns": selected_feature_columns,
        "target_columns": target_columns,
        "selected_alpha": 0.1,
    }
    manifest_path = model_dir / f"ridge_{station_id}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    model_path = model_dir / f"ridge_{station_id}.joblib"
    model_path.touch()
    return metadata_path, train_path, test_path, model_path, manifest_path


def test_load_and_score_saved_model_rejects_manifest_execution_provenance_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path, train_path, test_path, model_path, manifest_path = (
        _write_comparison_artifacts(
            tmp_path,
            execution_uuid="matching-execution",
            manifest_execution_uuid="different-execution",
        )
    )
    contract, _train_features, test_features = load_joined_training_data(
        metadata_path,
        train_path,
        test_path,
        station_id=TARGET_STATION_ID,
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
    )
    feature_subsets = build_feature_subsets(
        contract, weather_variables=WEATHER_VARIABLES
    )
    test_rows = prepare_model_rows(
        test_features, contract, artifact_name="comparison test"
    )
    selected_candidate = pd.Series({"subset": "lean", "alpha": 0.1})
    monkeypatch.setattr(ridge, "load_joblib", lambda _path: _FakeSavedRidge())

    with pytest.raises(ValueError, match="execution_uuid"):
        load_and_score_saved_model(
            model_path,
            manifest_path,
            contract,
            feature_subsets,
            test_rows,
            selected_candidate,
            "matching-execution",
            forecast_horizon_hours=FORECAST_HORIZON_HOURS,
            target_station_id=TARGET_STATION_ID,
        )


def test_load_and_score_saved_model_scores_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path, train_path, test_path, model_path, manifest_path = (
        _write_comparison_artifacts(tmp_path, execution_uuid="matching-execution")
    )
    contract, _train_features, test_features = load_joined_training_data(
        metadata_path,
        train_path,
        test_path,
        station_id=TARGET_STATION_ID,
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
    )
    feature_subsets = build_feature_subsets(
        contract, weather_variables=WEATHER_VARIABLES
    )
    test_rows = prepare_model_rows(
        test_features, contract, artifact_name="comparison test"
    )
    selected_candidate = pd.Series({"subset": "lean", "alpha": 0.1})
    fake_model = _FakeSavedRidge()
    monkeypatch.setattr(ridge, "load_joblib", lambda _path: fake_model)

    predictions = load_and_score_saved_model(
        model_path,
        manifest_path,
        contract,
        feature_subsets,
        test_rows,
        selected_candidate,
        "matching-execution",
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
        target_station_id=TARGET_STATION_ID,
    )

    assert predictions.shape == (len(test_rows), FORECAST_HORIZON_HOURS)
    assert fake_model.predictor_values is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(fake_model.predictor_values.columns) == list(
        manifest["selected_feature_columns"]
    )
