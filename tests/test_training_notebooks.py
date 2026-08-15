import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
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
    return "\n".join(
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
        "ridge-model-comparison-forecast-window-setup",
        "ridge-model-comparison-best-forecast-window",
        "ridge-model-comparison-worst-forecast-window",
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
        "\n".join("".join(cell["source"]) for cell in selection_cells),
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
    import joblib  # type: ignore[import-untyped]

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
    assert namespace["INPUT_PARQUET_SHA256_PARAMS"] == {
        "train_input_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_input_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }


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
    source = _ridge_model_comparison_source()
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


def test_ridge_forecast_window_cells_follow_heading_in_order() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    heading_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "markdown"
        and "## Inspect best and worst Ridge forecast windows"
        in "".join(cell.get("source", []))
    )
    best_indices = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "ridge-model-comparison-best-forecast-window"
        in cell.get("metadata", {}).get("tags", [])
    ]
    worst_indices = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "ridge-model-comparison-worst-forecast-window"
        in cell.get("metadata", {}).get("tags", [])
    ]

    assert len(best_indices) == 1
    assert len(worst_indices) == 1
    assert best_indices[0] > heading_index
    assert worst_indices[0] > best_indices[0]


def test_ridge_forecast_window_selection_and_figures() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    issue_times = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-02 00:00",
            "2024-01-02 01:00",
            "2024-01-05 00:00",
            "2024-01-06 00:00",
        ],
        utc=True,
    )
    target_columns = [
        f"{TARGET_STATION_ID}__target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    prediction_columns = [f"prediction_{column}" for column in target_columns]
    errors_by_issue = [1.0, 1.0, 5.0, 5.0, 0.0, 10.0]
    comparison_prediction_table = pd.DataFrame(
        {
            "issue_time": issue_times,
            **{column: np.zeros(len(issue_times)) for column in target_columns},
            **{column: np.asarray(errors_by_issue) for column in prediction_columns},
        }
    )

    context_times = pd.date_range(
        "2023-12-30 00:00",
        "2024-01-10 00:00",
        freq="h",
        tz="UTC",
    )
    water_level_column = f"{TARGET_STATION_ID}__water_level"
    imputed_column = f"{TARGET_STATION_ID}__imputed"
    context_source = pd.DataFrame(
        {
            "timestamp": context_times,
            water_level_column: np.arange(len(context_times), dtype=float),
            imputed_column: False,
        }
    )
    incomplete_time = pd.Timestamp("2024-01-07 00:00", tz="UTC")
    imputed_time = pd.Timestamp("2024-01-08 00:00", tz="UTC")
    context_source = context_source.loc[
        context_source["timestamp"] != incomplete_time
    ].copy()
    context_source.loc[context_source["timestamp"] == imputed_time, imputed_column] = (
        True
    )

    namespace = {
        "COMPARISON_TARGET_COLUMNS": target_columns,
        "TARGET_STATION_ID": TARGET_STATION_ID,
        "comparison_horizons": list(range(1, FORECAST_HORIZON_HOURS + 1)),
        "comparison_prediction_columns": prediction_columns,
        "comparison_prediction_table": comparison_prediction_table,
        "_comparison_train_features": context_source,
        "comparison_test_features": pd.DataFrame(
            {
                "timestamp": issue_times,
                water_level_column: 0.0,
                imputed_column: False,
            }
        ),
        "display": lambda _figure: None,
        "go": go,
        "np": np,
        "pd": pd,
    }
    exec(  # noqa: S102
        _cell_source(notebook, "ridge-model-comparison-forecast-window-setup"),
        namespace,
    )
    exec(  # noqa: S102
        _cell_source(notebook, "ridge-model-comparison-best-forecast-window"),
        namespace,
    )
    exec(  # noqa: S102
        _cell_source(notebook, "ridge-model-comparison-worst-forecast-window"),
        namespace,
    )

    assert namespace["best_forecast_window_row"]["issue_time"] == pd.Timestamp(
        "2024-01-01", tz="UTC"
    )
    assert namespace["worst_forecast_window_row"]["issue_time"] == pd.Timestamp(
        "2024-01-02", tz="UTC"
    )
    assert set(namespace["comparison_forecast_window_context_by_row"]) == {0, 1, 2, 3}
    assert np.isclose(namespace["best_forecast_window_row"]["issue_rmse"], 1.0)
    assert np.isclose(namespace["worst_forecast_window_row"]["issue_rmse"], 5.0)

    for figure_name, selected_issue_time in (
        ("best_ridge_forecast_window_figure", pd.Timestamp("2024-01-01", tz="UTC")),
        ("worst_ridge_forecast_window_figure", pd.Timestamp("2024-01-02", tz="UTC")),
    ):
        figure = namespace[figure_name]
        assert len(figure.data) == 2
        assert figure.data[0].name == "Ground truth"
        assert figure.data[1].name == "Prediction"
        assert figure.data[0].mode == "lines+markers"
        assert len(figure.frames) == len(figure.layout.sliders[0].steps) == 2
        assert 0 <= figure.layout.sliders[0].active < len(figure.frames)
        assert figure.layout.margin.b == 160
        assert figure.layout.sliders[0].y == pytest.approx(-0.28)
        assert figure.layout.sliders[0].pad.t == 12
        assert all(len(frame.data) == 1 for frame in figure.frames)
        assert all(list(frame.traces) == [1] for frame in figure.frames)
        assert all(frame.data[0].name == "Prediction" for frame in figure.frames)
        assert len(figure.frames[0].data[0].x) == FORECAST_HORIZON_HOURS
        assert len(figure.frames[1].data[0].x) == FORECAST_HORIZON_HOURS
        assert pd.Timestamp(
            figure.frames[0].data[0].x[0]
        ) == selected_issue_time + pd.to_timedelta(1, unit="h")
        assert pd.Timestamp(
            figure.frames[1].data[0].x[0]
        ) == selected_issue_time + pd.to_timedelta(2, unit="h")
        assert len(figure.data[0].x) == 97
        assert len(figure.data[0].y) == 97
        assert pd.Timestamp(
            figure.data[0].x[0]
        ) == selected_issue_time - pd.to_timedelta(48, unit="h")
        assert pd.Timestamp(
            figure.data[0].x[-1]
        ) == selected_issue_time + pd.to_timedelta(48, unit="h")
        assert len(figure.data[1].x) == FORECAST_HORIZON_HOURS
        assert len(figure.data[1].y) == FORECAST_HORIZON_HOURS
        assert pd.Timestamp(
            figure.data[1].x[0]
        ) == selected_issue_time + pd.to_timedelta(1, unit="h")
        assert pd.Timestamp(
            figure.data[1].x[-1]
        ) == selected_issue_time + pd.to_timedelta(FORECAST_HORIZON_HOURS, unit="h")
        assert figure.layout.xaxis.title.text == "Valid time"
        assert len(figure.layout.shapes) == 1
        assert any(
            annotation.text == "Issue time" for annotation in figure.layout.annotations
        )
        assert selected_issue_time.isoformat() in figure.layout.title.text
        assert "24-hour RMSE" in figure.layout.title.text


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
