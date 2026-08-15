import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID

NOTEBOOK_PATH = Path("04_train_ridge.ipynb")


def _cell_source(notebook: dict, tag: str) -> str:
    matching_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and tag in cell.get("metadata", {}).get("tags", [])
    ]
    if len(matching_cells) != 1:
        raise AssertionError(
            f"Expected exactly one code cell tagged {tag!r}, "
            f"found {len(matching_cells)}"
        )
    source = matching_cells[0]["source"]
    return "".join(source) if isinstance(source, list) else source


def _ridge_model_comparison_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    comparison_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and any(
            tag == "ridge-model-comparison" or tag.startswith("ridge-model-comparison-")
            for tag in cell.get("metadata", {}).get("tags", [])
        )
    ]
    return "".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in comparison_cells
    )


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


class _FakeMlflow(types.ModuleType):
    def __init__(self, runs: pd.DataFrame) -> None:
        super().__init__("mlflow")
        self.runs = runs
        self.tracking_uri: str | None = None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def get_experiment_by_name(self, name: str) -> SimpleNamespace:
        assert name == "ridge"
        return SimpleNamespace(experiment_id="ridge-experiment")

    def search_runs(self, *, experiment_ids: list[str]) -> pd.DataFrame:
        assert experiment_ids == ["ridge-experiment"]
        return self.runs.copy()


def _execute_ridge_model_selection(
    runs: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> dict:
    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow(runs))
    namespace: dict = {}
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    excluded_tags = {
        "ridge-model-comparison-model",
        "ridge-model-comparison-predictions",
        "ridge-model-comparison-error-charts",
    }
    selection_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and any(
            tag == "ridge-model-comparison" or tag.startswith("ridge-model-comparison-")
            for tag in cell.get("metadata", {}).get("tags", [])
        )
        and not excluded_tags.intersection(cell.get("metadata", {}).get("tags", []))
    ]
    exec(  # noqa: S102
        "".join("".join(cell["source"]) for cell in selection_cells),
        namespace,
    )
    return namespace


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
) -> _FakeSavedRidge:
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
    metadata = {
        "engineered_station_ids": [station_id],
        "configuration": {"horizon_hours": FORECAST_HORIZON_HOURS},
        "predictor_columns": predictor_columns,
        "target_columns": target_columns,
    }
    (processed_dir / "all_stations_feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
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
    frame.to_parquet(processed_dir / "all_stations_train_features.parquet", index=False)
    frame.to_parquet(processed_dir / "all_stations_test_features.parquet", index=False)

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
    (model_dir / f"ridge_{station_id}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (model_dir / f"ridge_{station_id}.joblib").touch()
    return _FakeSavedRidge()


def _execute_ridge_model_comparison_with_artifacts(
    runs: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_execution_uuid: str | None = None,
) -> tuple[dict, _FakeSavedRidge]:
    model = _write_comparison_artifacts(
        tmp_path,
        execution_uuid=str(
            runs.loc[
                runs["tags.run_type"].eq("sealed_test"), "tags.execution_uuid"
            ].iloc[0]
        ),
        manifest_execution_uuid=manifest_execution_uuid,
    )
    import joblib

    source = _ridge_model_comparison_source()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(joblib, "load", lambda _path: model)
    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow(runs))
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102
    return namespace, model


def test_ridge_notebook_wires_shared_loading_subsets_and_cohorts(
    tmp_path: Path,
) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    processed_dir = tmp_path / "joined"
    processed_dir.mkdir()
    metadata_path = processed_dir / "all_stations_feature_metadata.json"
    train_path = processed_dir / "all_stations_train_features.parquet"
    test_path = processed_dir / "all_stations_test_features.parquet"
    station_id = TARGET_STATION_ID
    target_columns = [
        f"{station_id}__target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    predictor_columns = [
        f"{station_id}__water_level",
        f"{station_id}__imputed",
        f"{station_id}__precipitation",
        f"{station_id}__temperature_2m",
        f"{station_id}__water_level_lag_24h",
        f"{station_id}__imputed_count_24h",
        f"{station_id}__utc_hour_sin",
        "station-b__water_level",
        "station-b__imputed",
        "station-b__precipitation",
        "station-b__temperature_2m",
    ]
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
                ["2024-01-03", "2024-01-01", "2024-01-02"], utc=True
            ),
            f"{station_id}__target_valid": [True, True, False],
        }
    )
    for column in predictor_columns:
        frame[column] = False if column.endswith("__imputed") else 1.0
    for target_number, column in enumerate(target_columns, start=1):
        frame[column] = np.arange(len(frame), dtype=float) + target_number
    frame.loc[2, target_columns] = np.nan
    frame.to_parquet(train_path, index=False)
    frame.iloc[[1]].to_parquet(test_path, index=False)

    setup_source = _cell_source(notebook, "ridge-setup").replace(
        'PROCESSED_DIR = Path("data/processed/joined")',
        f"PROCESSED_DIR = Path({str(processed_dir)!r})",
    )
    namespace: dict = {}
    exec(setup_source, namespace)  # noqa: S102
    exec(_cell_source(notebook, "ridge-load-artifacts"), namespace)  # noqa: S102
    exec(_cell_source(notebook, "ridge-prepare-cohort"), namespace)  # noqa: S102

    assert namespace["contract"].station_id == station_id
    assert namespace["FULL_FEATURE_COLUMNS"] == predictor_columns
    assert list(namespace["FEATURE_SUBSETS"]) == [
        "full",
        "all_station_hydrology_quality_time",
        "raw_all_stations",
        "target_station_full",
        "target_station_hydrology_quality_time",
        "current_water_levels_all_stations",
    ]
    assert namespace["train_rows"]["timestamp"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)
    )
    assert len(namespace["test_rows"]) == 1
    assert namespace["INPUT_PARQUET_SHA256_PARAMS"] == {
        "train_input_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_input_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }
    assert "absolute_error_boxplot_payload(" in _cell_source(
        notebook, "ridge-evaluation"
    )


def test_ridge_notebook_logs_input_hashes_on_every_mlflow_run() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cv_source = _cell_source(notebook, "ridge-cross-validation")
    test_source = _cell_source(notebook, "ridge-evaluation")

    assert cv_source.count("**INPUT_PARQUET_SHA256_PARAMS") == 2
    assert test_source.count("**INPUT_PARQUET_SHA256_PARAMS") == 1


def test_ridge_candidate_selection_applies_all_tie_breakers() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict = {"pd": pd}
    exec(_cell_source(notebook, "ridge-selection"), namespace)  # noqa: S102
    select_candidate = namespace["select_candidate"]

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


def test_ridge_model_comparison_is_standalone_read_only_and_plotly_based() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = _ridge_model_comparison_source()
    section_start = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "markdown"
        and cell.get("source", [""])[0] == "# Ridge MLflow candidate comparison\n"
    )
    section_cells = notebook["cells"][section_start:]

    assert len([cell for cell in section_cells if cell["cell_type"] == "code"]) == 12
    assert len([cell for cell in section_cells if cell["cell_type"] == "markdown"]) == 8
    assert (
        sum(
            "## " in "".join(cell.get("source", []))
            for cell in section_cells
            if cell["cell_type"] == "markdown"
        )
        == 7
    )
    assert "import mlflow" in source
    assert "CV_SELECTION_METRIC," in source
    assert "MLFLOW_TRACKING_URI," in source
    assert "import pandas as pd" in source
    assert "import plotly.graph_objects as go" in source
    assert "from IPython.display import display" in source
    assert "mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)" in source
    assert "mlflow.get_experiment_by_name" in source
    assert "mlflow.search_runs(" in source
    assert "from joblib import load as load_joblib" in source
    assert "load_joined_training_data(" in source
    assert "prepare_model_rows(" in source
    assert "validate_predictions(" in source
    assert "go.Scattergl" in source
    assert "go.Box" in source
    assert "boxpoints=False" in source
    assert "signed_errors =" in source
    assert "issue_time" in source
    assert "sliders" in source
    assert "mlflow.start_run" not in source
    assert "write_text" not in source
    assert "log_metrics" not in source


def test_ridge_model_comparison_skips_newer_invalid_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    namespace = _execute_ridge_model_selection(runs, monkeypatch)

    assert namespace["selected_execution_uuid"] == "old-execution"
    assert len(namespace["candidate_comparison_table"]) == 2


def test_ridge_model_comparison_reports_missing_mlflow_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Run 04_train_ridge.ipynb"):
        _execute_ridge_model_selection(pd.DataFrame(), monkeypatch)


def test_ridge_model_comparison_uses_cv_for_candidates_and_test_for_selected_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _comparison_runs(
        "valid-execution",
        "2025-01-01T00:00:00Z",
        selected_subset="wide",
        selected_alpha="1.0",
        selected_test_mae=7.5,
        lean_cv_mae=1.0,
    )

    namespace = _execute_ridge_model_selection(runs, monkeypatch)
    candidate_table = namespace["candidate_comparison_table"]
    selected_summary = namespace["selected_candidate_sealed_test_summary"]

    assert candidate_table["subset"].tolist() == ["lean", "wide"]
    assert candidate_table["cv_mae_mean"].tolist() == [1.0, 2.0]
    assert "test_mae" not in candidate_table.columns
    assert len(selected_summary) == 1
    assert selected_summary.loc[0, "subset"] == "wide"
    assert selected_summary.loc[0, "test_mae"] == 7.5


def test_ridge_model_comparison_ranks_by_configured_rmse_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _comparison_runs("rmse-ranked-execution", "2025-01-01T00:00:00Z")
    runs.loc[runs["tags.subset"].eq("lean"), "metrics.cv_rmse_mean"] = 5.0
    runs.loc[runs["tags.subset"].eq("wide"), "metrics.cv_rmse_mean"] = 1.0

    namespace = _execute_ridge_model_selection(runs, monkeypatch)

    assert namespace["candidate_comparison_table"]["subset"].tolist() == [
        "wide",
        "lean",
    ]


def test_ridge_model_comparison_skips_legacy_metric_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        _execute_ridge_model_selection(legacy_runs, monkeypatch)


def test_ridge_saved_model_scores_all_horizons_and_aligns_h_plus_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _comparison_runs("saved-model-execution", "2025-01-01T00:00:00Z")

    namespace, model = _execute_ridge_model_comparison_with_artifacts(
        runs, tmp_path, monkeypatch
    )

    predictions = namespace["comparison_prediction_values"]
    assert predictions.shape == (3, FORECAST_HORIZON_HOURS)
    assert model.predictor_values is not None
    assert model.predictor_values.shape == (3, 2)
    assert namespace["comparison_prediction_table"].columns.tolist() == [
        "issue_time",
        *namespace["COMPARISON_TARGET_COLUMNS"],
        *namespace["comparison_prediction_columns"],
    ]
    assert namespace["comparison_prediction_table"]["issue_time"].tolist() == list(
        pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00"],
            utc=True,
        )
    )

    horizon_two_frame = namespace["comparison_time_series_frames"][1]
    table = namespace["comparison_prediction_table"]
    expected_valid_times = table["issue_time"] + pd.to_timedelta(2, unit="h")
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(pd.to_datetime(horizon_two_frame.data[0].x, utc=True)),
        pd.DatetimeIndex(expected_valid_times).rename(None),
    )
    assert list(horizon_two_frame.data[0].y) == list(
        table[namespace["COMPARISON_TARGET_COLUMNS"][1]]
    )
    assert list(horizon_two_frame.data[1].y) == list(
        table[namespace["comparison_prediction_columns"][1]]
    )
    assert all(trace.mode == "lines+markers" for trace in horizon_two_frame.data)
    assert all(trace.type == "scattergl" for trace in horizon_two_frame.data)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(
            pd.to_datetime(horizon_two_frame.data[0].customdata, utc=True)
        ),
        pd.DatetimeIndex(table["issue_time"]).rename(None),
    )


def test_ridge_saved_model_error_figures_use_signed_errors_and_all_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _comparison_runs("error-figure-execution", "2025-01-01T00:00:00Z")

    namespace, _model = _execute_ridge_model_comparison_with_artifacts(
        runs, tmp_path, monkeypatch
    )

    actual = namespace["comparison_actual_values"]
    predictions = namespace["comparison_prediction_values"]
    signed_errors = predictions - actual
    signed_figure = namespace["signed_error_boxplot_figure"]
    absolute_figure = namespace["absolute_error_boxplot_figure"]
    expected_horizons = list(range(1, FORECAST_HORIZON_HOURS + 1))
    expected_horizon_labels = [f"H+{horizon:02d}" for horizon in expected_horizons]
    signed_box_traces = [trace for trace in signed_figure.data if trace.type == "box"]
    absolute_box_traces = [
        trace for trace in absolute_figure.data if trace.type == "box"
    ]

    assert len(signed_box_traces) == 1
    assert len(absolute_box_traces) == 1
    assert [trace.name for trace in signed_box_traces] == ["Boxplots"]
    assert [trace.name for trace in absolute_box_traces] == ["Boxplots"]
    for figure in (signed_figure, absolute_figure):
        assert figure.layout.xaxis.type == "category"
    np.testing.assert_allclose(signed_box_traces[0].y, signed_errors.reshape(-1))
    np.testing.assert_allclose(
        absolute_box_traces[0].y, np.abs(signed_errors).reshape(-1)
    )
    assert all(trace.boxpoints is False for trace in signed_box_traces)
    assert all(trace.boxpoints is False for trace in absolute_box_traces)
    expected_flattened_labels = expected_horizon_labels * len(signed_errors)
    assert list(signed_box_traces[0].x) == expected_flattened_labels
    assert list(absolute_box_traces[0].x) == expected_flattened_labels

    signed_mean_trace = next(
        trace for trace in signed_figure.data if trace.name == "Mean error"
    )
    assert list(signed_mean_trace.x) == expected_horizon_labels
    np.testing.assert_allclose(signed_mean_trace.y, signed_errors.mean(axis=0))
    absolute_marker_traces = {
        trace.name: trace for trace in absolute_figure.data if trace.type == "scatter"
    }
    assert list(absolute_marker_traces["MAE"].x) == expected_horizon_labels
    assert list(absolute_marker_traces["RMSE"].x) == expected_horizon_labels
    np.testing.assert_allclose(
        absolute_marker_traces["MAE"].y,
        [7.5 + horizon for horizon in range(1, FORECAST_HORIZON_HOURS + 1)],
    )
    np.testing.assert_allclose(
        absolute_marker_traces["RMSE"].y,
        [8.5 + horizon for horizon in range(1, FORECAST_HORIZON_HOURS + 1)],
    )
    assert signed_figure.layout.shapes[0].y0 == 0
    assert signed_figure.layout.shapes[0].y1 == 0


def test_ridge_saved_model_rejects_manifest_execution_provenance_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _comparison_runs("provenance-execution", "2025-01-01T00:00:00Z")

    with pytest.raises(ValueError, match="execution_uuid"):
        _execute_ridge_model_comparison_with_artifacts(
            runs,
            tmp_path,
            monkeypatch,
            manifest_execution_uuid="different-execution",
        )
