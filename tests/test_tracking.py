from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src import tracking
from src.tracking import (
    CV_SUMMARY_COLUMNS,
    backdated_run,
    cv_result_row,
    cv_results_table,
    fold_window_params,
    log_evaluation_figures,
    metric_log_dict,
    run_cv_candidate,
    run_tags,
    sealed_test_tables,
    selected_cv_metric,
    verified_cv_results,
    verify_cv_runs,
)

matplotlib.use("Agg")


@pytest.fixture
def tracking_uri(tmp_path: Path) -> Iterator[str]:
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(previous)


def _metric_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = pd.DataFrame([{"mae": 1.0, "rmse": 2.0, "me": -0.5, "r2": 0.9}])
    per_horizon = pd.DataFrame(
        {
            "horizon_hours": [1, 2],
            "mae": [0.5, 1.5],
            "rmse": [1.0, 3.0],
            "me": [0.0, -1.0],
            "r2": [0.95, 0.85],
        }
    )
    return aggregate, per_horizon


def test_run_tags_stringifies_extras_and_keeps_execution_uuid() -> None:
    tags = run_tags("cv", "fold", "uuid-1", subset="full", log1p=True, fold=3)

    assert tags == {
        "phase": "cv",
        "run_type": "fold",
        "subset": "full",
        "log1p": "True",
        "fold": "3",
        "execution_uuid": "uuid-1",
    }


def test_metric_log_dict_flattens_aggregate_and_horizon_metrics() -> None:
    aggregate, per_horizon = _metric_tables()

    metrics = metric_log_dict("fold", aggregate, per_horizon)

    assert metrics == {
        "fold_mae": 1.0,
        "fold_rmse": 2.0,
        "fold_me": -0.5,
        "fold_r2": 0.9,
        "fold_mae_horizon_01": 0.5,
        "fold_rmse_horizon_01": 1.0,
        "fold_me_horizon_01": 0.0,
        "fold_r2_horizon_01": 0.95,
        "fold_mae_horizon_02": 1.5,
        "fold_rmse_horizon_02": 3.0,
        "fold_me_horizon_02": -1.0,
        "fold_r2_horizon_02": 0.85,
    }


def test_fold_window_params_describes_train_and_validation_windows() -> None:
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"))

    params = fold_window_params(
        timestamps,
        np.array([0, 1, 2, 3]),
        np.array([6, 7]),
        gap_rows=2,
        forecast_horizon_hours=24,
    )

    assert params == {
        "train_rows": 4,
        "validation_rows": 2,
        "gap_rows": 2,
        "forecast_horizon_hours": 24,
        "train_start": "2024-01-01T00:00:00+00:00",
        "train_end": "2024-01-01T03:00:00+00:00",
        "validation_start": "2024-01-01T06:00:00+00:00",
        "validation_end": "2024-01-01T07:00:00+00:00",
        "train_index_start": 0,
        "train_index_end": 3,
        "validation_index_start": 6,
        "validation_index_end": 7,
    }


def test_fold_window_params_omits_training_fields_without_train_indices() -> None:
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"))

    params = fold_window_params(
        timestamps, None, [8, 9], gap_rows=0, forecast_horizon_hours=1
    )

    assert not any(key.startswith("train_") for key in params)
    assert params["validation_rows"] == 2


def test_cv_result_row_renames_aggregates_and_optionally_keeps_horizons() -> None:
    parent_metrics = {
        **{f"cv_{column}": float(i) for i, column in enumerate(CV_SUMMARY_COLUMNS)},
        "cv_mae_horizon_01_mean": 9.0,
    }

    row = cv_result_row({"subset": "full"}, parent_metrics)
    compact = cv_result_row({"subset": "full"}, parent_metrics, include_horizons=False)

    assert row == {
        "subset": "full",
        **{column: float(i) for i, column in enumerate(CV_SUMMARY_COLUMNS)},
        "cv_mae_horizon_01_mean": 9.0,
    }
    assert "cv_mae_horizon_01_mean" not in compact


def test_cv_results_table_checks_candidate_coverage() -> None:
    rows = [{"subset": "a", "alpha": 1.0}, {"subset": "b", "alpha": 2.0}]

    table = cv_results_table(
        rows,
        expected_candidate_keys={("a", 1.0), ("b", 2.0)},
        key_columns=["subset", "alpha"],
    )

    assert table["subset"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="incomplete"):
        cv_results_table(
            rows[:1],
            expected_candidate_keys={("a", 1.0), ("b", 2.0)},
            key_columns=["subset", "alpha"],
        )
    with pytest.raises(ValueError, match="does not match"):
        cv_results_table(
            rows,
            expected_candidate_keys={("a", 1.0), ("c", 2.0)},
            key_columns=["subset", "alpha"],
        )


def test_selected_cv_metric_reads_the_matching_row() -> None:
    cv_results = pd.DataFrame(
        {
            "subset": ["a", "a", "b"],
            "alpha": [1.0, 2.0, 1.0],
            "rmse_mean": [3.0, 2.0, 1.0],
        }
    )

    value = selected_cv_metric(cv_results, {"subset": "a", "alpha": 2.0}, "rmse")

    assert value == 2.0


def _log_cv_runs(
    execution_uuid: str, candidates: list[str], n_folds: int, *, skip_fold: bool
) -> None:
    mlflow.set_experiment("demo")
    for subset in candidates:
        with mlflow.start_run(
            tags=run_tags("cv", "candidate_parent", execution_uuid, subset=subset)
        ):
            for fold in range(1, n_folds + 1):
                if skip_fold and subset == candidates[-1] and fold == n_folds:
                    continue
                with mlflow.start_run(
                    nested=True,
                    tags=run_tags(
                        "cv", "fold", execution_uuid, subset=subset, fold=fold
                    ),
                ):
                    pass


def test_verify_cv_runs_accepts_complete_executions(tracking_uri: str) -> None:
    _log_cv_runs("uuid-1", ["a", "b"], 2, skip_fold=False)
    _log_cv_runs("uuid-2", ["a"], 2, skip_fold=False)

    verify_cv_runs(
        experiment_name="demo",
        execution_uuid="uuid-1",
        expected_candidate_keys={("a",), ("b",)},
        n_validation_folds=2,
        run_key=lambda row: (str(row["tags.subset"]),),
    )


def test_verify_cv_runs_rejects_missing_candidates_and_folds(
    tracking_uri: str,
) -> None:
    _log_cv_runs("uuid-1", ["a", "b"], 2, skip_fold=True)

    with pytest.raises(ValueError, match="every expected candidate"):
        verify_cv_runs(
            experiment_name="demo",
            execution_uuid="uuid-1",
            expected_candidate_keys={("a",), ("b",), ("c",)},
            n_validation_folds=2,
            run_key=lambda row: (str(row["tags.subset"]),),
        )
    with pytest.raises(ValueError, match="nested fold runs"):
        verify_cv_runs(
            experiment_name="demo",
            execution_uuid="uuid-1",
            expected_candidate_keys={("a",), ("b",)},
            n_validation_folds=2,
            run_key=lambda row: (str(row["tags.subset"]),),
        )
    with pytest.raises(ValueError, match="was not found"):
        verify_cv_runs(
            experiment_name="absent",
            execution_uuid="uuid-1",
            expected_candidate_keys=set(),
            n_validation_folds=2,
            run_key=lambda row: (),
        )


def test_backdated_run_records_supplied_times(tracking_uri: str) -> None:
    client = MlflowClient()
    experiment_id = mlflow.set_experiment("demo").experiment_id

    with backdated_run(
        client,
        experiment_id=experiment_id,
        run_name="candidate",
        tags={"phase": "cv"},
        start_time_ms=1_000,
        end_time_ms=5_000,
    ) as run:
        mlflow.log_param("order", "(1, 0, 0)")

    logged = client.get_run(run.info.run_id)
    assert logged.info.start_time == 1_000
    assert logged.info.end_time == 5_000
    assert logged.info.status == "FINISHED"
    assert logged.data.params == {"order": "(1, 0, 0)"}
    assert logged.data.tags["phase"] == "cv"


def test_log_evaluation_figures_logs_three_titled_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: dict[str, str] = {}
    monkeypatch.setattr(
        tracking.mlflow,
        "log_figure",
        lambda figure, name: logged.update({name: figure.axes[0].get_title()}),
    )
    monkeypatch.setattr(tracking, "cv_error_boxplots_figure", _titled_figure)
    monkeypatch.setattr(tracking, "test_error_boxplots_figure", _titled_figure)
    monkeypatch.setattr(tracking, "predicted_vs_actual_figure", _titled_figure)
    test_rows = pd.DataFrame({"target": [1.0]})

    log_evaluation_figures(
        "Ridge",
        "full, alpha=1",
        cv_horizon_metrics=pd.DataFrame(),
        test_rows=test_rows,
        test_predictions=np.array([[1.0]]),
        per_horizon_metrics=pd.DataFrame(),
        target_columns=["target"],
        show=False,
    )

    assert logged == {
        "cv_rmse_mae_boxplots.png": "Ridge CV errors — full, alpha=1",
        "test_error_boxplots.png": "Ridge final-test errors — full, alpha=1",
        "test_predicted_vs_actual.png": "Ridge predicted vs actual — full, alpha=1",
    }


def _titled_figure(*args: object, title: str) -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.set_title(title)
    return figure


def test_sealed_test_tables_rejects_non_finite_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, per_horizon = _metric_tables()
    aggregate.loc[0, "rmse"] = np.nan
    monkeypatch.setattr(
        tracking, "metric_tables", lambda *args, **kwargs: (aggregate, per_horizon)
    )
    monkeypatch.setattr(
        tracking,
        "sealed_test_regime_tables",
        lambda *args, **kwargs: ({}, pd.DataFrame(), pd.DataFrame()),
    )
    dataset: Any = SimpleNamespace(
        target_water_level_quartile_cutoffs_cm=(1.0, 2.0, 3.0),
        target_water_level_quartile_reference_count=10,
    )

    def score(model_label: str | None) -> tuple[Any, ...]:
        return sealed_test_tables(
            pd.DataFrame(),
            np.array([[1.0]]),
            dataset=dataset,
            target_columns=["target"],
            station_id="station",
            model_label=model_label,
        )

    assert score(None)[0] is aggregate
    with pytest.raises(ValueError, match="Ridge reported non-finite aggregate"):
        score("Ridge")


def test_run_cv_candidate_logs_parent_and_fold_runs(tracking_uri: str) -> None:
    mlflow.set_experiment("demo")
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"))
    targets = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    folds = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([0, 1, 2]), np.array([4, 5])),
    ]
    calls: list[tuple[list[int], list[int]]] = []

    def fit_predict(
        train_indices: np.ndarray, validation_indices: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray]:
        calls.append((train_indices.tolist(), validation_indices.tolist()))
        actual = targets.iloc[validation_indices]
        return actual, actual.to_numpy() + 1.0

    result = run_cv_candidate(
        "demo_cv",
        fit_predict,
        folds=folds,
        execution_uuid="uuid-1",
        tags={"subset": "full"},
        parent_params={"alpha": 1.0},
        fold_params={"subset": "full"},
        target_columns=["target"],
        station_id="station",
        window_timestamps=timestamps,
        gap_rows=0,
        forecast_horizon_hours=1,
    )

    assert calls == [([0, 1], [2, 3]), ([0, 1, 2], [4, 5])]
    assert result.parent_metrics["cv_mae_mean"] == pytest.approx(1.0)
    assert len(result.fold_aggregate_metrics) == 2
    assert len(result.fold_horizon_rows) == 2
    runs = mlflow.search_runs(experiment_names=["demo"])
    assert isinstance(runs, pd.DataFrame)
    parent = runs[runs["tags.run_type"] == "candidate_parent"].iloc[0]
    folds_logged = runs[runs["tags.run_type"] == "fold"].sort_values("tags.fold")
    assert parent["tags.mlflow.runName"] == "demo_cv"
    assert parent["params.alpha"] == "1.0"
    assert parent["metrics.cv_mae_mean"] == pytest.approx(1.0)
    assert folds_logged["tags.mlflow.runName"].tolist() == [
        "demo_cv_fold_1",
        "demo_cv_fold_2",
    ]
    assert folds_logged["params.train_start"].tolist() == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
    ]
    assert folds_logged["metrics.fold_mae"].tolist() == [1.0, 1.0]
    assert set(folds_logged["tags.subset"]) == {"full"}


def test_run_cv_candidate_logs_only_counts_without_timestamps(
    tracking_uri: str,
) -> None:
    mlflow.set_experiment("demo")
    targets = pd.DataFrame({"target": [1.0, 2.0, 3.0]})

    run_cv_candidate(
        "demo_cv",
        lambda train, validation: (targets.iloc[validation], np.ones((1, 1))),
        folds=[(np.array([0, 1]), np.array([2]))],
        execution_uuid="uuid-1",
        tags={},
        parent_params={},
        fold_params={},
        target_columns=["target"],
        station_id="station",
        window_timestamps=None,
        gap_rows=3,
        forecast_horizon_hours=1,
    )

    fold = mlflow.search_runs(
        experiment_names=["demo"], filter_string="tags.run_type = 'fold'"
    )
    assert isinstance(fold, pd.DataFrame)
    params = {
        str(column).removeprefix("params."): value
        for column, value in fold.iloc[0].items()
        if str(column).startswith("params.")
    }
    assert params == {
        "phase": "cv",
        "run_type": "fold",
        "fold": "1",
        "train_rows": "2",
        "validation_rows": "1",
        "gap_rows": "3",
    }


def test_verified_cv_results_sorts_by_key_columns(tracking_uri: str) -> None:
    _log_cv_runs("uuid-1", ["b", "a"], 1, skip_fold=False)

    table = verified_cv_results(
        [{"subset": "b"}, {"subset": "a"}],
        experiment_name="demo",
        execution_uuid="uuid-1",
        expected_candidate_keys={("a",), ("b",)},
        n_validation_folds=1,
        run_key=lambda row: (str(row["tags.subset"]),),
        key_columns=["subset"],
    )

    assert table["subset"].tolist() == ["a", "b"]


def test_log_sealed_test_run_logs_params_metrics_and_extra_figures(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow.set_experiment("demo")
    aggregate, per_horizon = _metric_tables()
    sealed = tracking.SealedTestTables(
        aggregate, per_horizon, {"definition": 1}, pd.DataFrame(), pd.DataFrame()
    )
    monkeypatch.setattr(tracking, "regime_mlflow_params", lambda d: {"regime": "q"})
    monkeypatch.setattr(
        tracking, "regime_mlflow_metrics", lambda a, h: {"regime_q1_mae": 1.0}
    )
    monkeypatch.setattr(tracking, "log_evaluation_figures", lambda *a, **k: None)
    logged_figures: list[str] = []
    monkeypatch.setattr(
        tracking.mlflow, "log_figure", lambda figure, name: logged_figures.append(name)
    )

    tracking.log_sealed_test_run(
        "demo_test",
        sealed,
        execution_uuid="uuid-1",
        tags={"subset": "full"},
        params={"alpha": 1.0},
        model_label="Demo",
        title_detail=None,
        cv_horizon_metrics=pd.DataFrame(),
        test_rows=pd.DataFrame(),
        test_predictions=np.empty((0, 1)),
        target_columns=["target"],
        show_figures=False,
        extra_figures={"loss.png": lambda: _titled_figure(title="Loss")},
    )

    runs = mlflow.search_runs(experiment_names=["demo"])
    assert isinstance(runs, pd.DataFrame)
    run = runs.iloc[0]
    assert run["tags.run_type"] == "sealed_test"
    assert run["tags.subset"] == "full"
    assert run["params.phase"] == "test"
    assert run["params.alpha"] == "1.0"
    assert run["params.regime"] == "q"
    assert run["metrics.test_rmse_horizon_02"] == 3.0
    assert run["metrics.regime_q1_mae"] == 1.0
    assert logged_figures == ["loss.png"]
