import warnings
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.evaluation import (
    COMPARISON_METRICS_COLUMNS,
    FEATURE_SUBSET_CV_COLUMNS,
    load_latest_complete_comparison_metrics,
    load_latest_complete_feature_subset_cv_metrics,
)

METRICS = ("mae", "rmse", "me", "r2")


class FakeMlflowClient:
    def __init__(self, runs_by_experiment: dict[str, list[SimpleNamespace]]) -> None:
        self.runs_by_experiment = runs_by_experiment

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        if name not in self.runs_by_experiment:
            return None
        return SimpleNamespace(experiment_id=name)

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str,
        run_view_type: object = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> list[SimpleNamespace]:
        del run_view_type, max_results, order_by, page_token
        run_type = (
            "sealed_test" if "sealed_test" in filter_string else "candidate_parent"
        )
        return sorted(
            [
                run
                for run in self.runs_by_experiment[experiment_ids[0]]
                if run.data.tags["run_type"] == run_type
                and run.info.status == "FINISHED"
            ],
            key=lambda run: run.info.end_time,
            reverse=True,
        )


def _run(
    run_id: str,
    *,
    run_type: str,
    execution_uuid: str,
    end_time: int,
    params: dict[str, str],
    metrics: dict[str, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            status="FINISHED",
            end_time=end_time,
        ),
        data=SimpleNamespace(
            tags={"run_type": run_type, "execution_uuid": execution_uuid},
            params=params,
            metrics=metrics,
        ),
    )


def _cv_metrics(horizon_hours: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric_number, metric in enumerate(METRICS, start=1):
        result[f"cv_{metric}_mean"] = float(metric_number)
        result[f"cv_{metric}_std"] = metric_number / 10
        for horizon in range(1, horizon_hours + 1):
            result[f"cv_{metric}_horizon_{horizon:02d}_mean"] = (
                metric_number + horizon / 100
            )
            result[f"cv_{metric}_horizon_{horizon:02d}_std"] = (
                metric_number / 10 + horizon / 1000
            )
    return result


def _test_metrics(horizon_hours: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric_number, metric in enumerate(METRICS, start=1):
        result[f"test_{metric}"] = metric_number + 10.0
        for horizon in range(1, horizon_hours + 1):
            result[f"test_{metric}_horizon_{horizon:02d}"] = (
                metric_number + 10.0 + horizon / 100
            )
    return result


def _execution_runs(
    experiment_name: str,
    execution_uuid: str,
    *,
    horizon_hours: int = 2,
    end_time: int = 2_000,
    train_hash: str = "train-hash",
    test_hash: str = "test-hash",
    cohort_rows: int = 100,
    candidate_params: dict[str, str] | None = None,
) -> list[SimpleNamespace]:
    candidate_params = candidate_params or _candidate_params(experiment_name)
    provenance = {
        "train_input_sha256": train_hash,
        "test_input_sha256": test_hash,
        "forecast_horizon_hours": str(horizon_hours),
    }
    candidate = _run(
        f"{execution_uuid}-candidate",
        run_type="candidate_parent",
        execution_uuid=execution_uuid,
        end_time=end_time - 100,
        params={**provenance, **candidate_params},
        metrics=_cv_metrics(horizon_hours),
    )
    sealed = _run(
        f"{execution_uuid}-sealed",
        run_type="sealed_test",
        execution_uuid=execution_uuid,
        end_time=end_time,
        params={
            **provenance,
            **candidate_params,
            "scored_issue_times": str(cohort_rows),
        },
        metrics=_test_metrics(horizon_hours),
    )
    return [candidate, sealed]


def _candidate_params(experiment_name: str) -> dict[str, str]:
    if experiment_name == "persistence":
        return {"persistence_column": "station__water_level"}
    if experiment_name == "ridge":
        return {"subset": "full", "alpha": "0.1", "log1p": "False"}
    raise AssertionError(f"Missing test candidate parameters for {experiment_name}")


def _load(
    runs_by_experiment: dict[str, list[SimpleNamespace]],
    *,
    experiments: dict[str, str] | None = None,
) -> pd.DataFrame:
    return load_latest_complete_comparison_metrics(
        client=cast(MlflowClient, FakeMlflowClient(runs_by_experiment)),
        model_experiments=experiments or {"Persistence": "persistence"},
        forecast_horizon_hours=2,
    )


def test_load_latest_complete_falls_back_from_newest_incomplete_execution() -> None:
    older = _execution_runs("persistence", "older", end_time=2_000)
    newer = _execution_runs("persistence", "newer", end_time=3_000)
    newer[1].data.metrics.pop("test_r2_horizon_02")

    with pytest.warns(UserWarning, match="newer.*incomplete"):
        result = _load({"persistence": older + newer})

    assert set(result["execution_uuid"]) == {"older"}


def test_load_latest_complete_warns_and_skips_missing_experiment() -> None:
    runs = {"persistence": _execution_runs("persistence", "complete")}

    with pytest.warns(UserWarning, match="random_forest.*not found"):
        result = _load(
            runs,
            experiments={
                "Persistence": "persistence",
                "Random Forest": "random_forest",
            },
        )

    assert set(result["model"]) == {"Persistence"}


def test_load_latest_complete_matches_the_unique_selected_candidate() -> None:
    selected = _execution_runs("ridge", "ridge-execution")
    unselected = _run(
        "unselected-candidate",
        run_type="candidate_parent",
        execution_uuid="ridge-execution",
        end_time=1_800,
        params={
            **selected[0].data.params,
            "alpha": "10.0",
        },
        metrics=_cv_metrics(2),
    )

    result = _load(
        {"ridge": [*selected, unselected]},
        experiments={"Ridge": "ridge"},
    )

    cv_rows = result[result["phase"].eq("cross_validation")]
    assert set(cv_rows["run_id"]) == {"ridge-execution-candidate"}


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_load_latest_complete_rejects_non_finite_candidate_metrics(
    invalid_value: float,
) -> None:
    runs = _execution_runs("persistence", "malformed")
    runs[0].data.metrics["cv_mae_mean"] = invalid_value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = _load({"persistence": runs})

    warning_messages = [str(warning.message) for warning in caught_warnings]
    assert any("non-finite metrics" in message for message in warning_messages)
    assert any("no complete execution" in message for message in warning_messages)
    assert result.empty
    assert list(result.columns) == COMPARISON_METRICS_COLUMNS


def test_load_latest_complete_rejects_ambiguous_selected_candidate() -> None:
    runs = _execution_runs("ridge", "ambiguous")
    duplicate = _run(
        "duplicate-candidate",
        run_type="candidate_parent",
        execution_uuid="ambiguous",
        end_time=1_850,
        params=runs[0].data.params.copy(),
        metrics=_cv_metrics(2),
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = _load(
            {"ridge": [*runs, duplicate]},
            experiments={"Ridge": "ridge"},
        )

    warning_messages = [str(warning.message) for warning in caught_warnings]
    assert any(
        "matched 2 candidate_parent runs" in message for message in warning_messages
    )
    assert any("no complete execution" in message for message in warning_messages)
    assert result.empty


@pytest.mark.parametrize(
    ("parameter", "different_value"),
    [
        ("train_input_sha256", "different-train"),
        ("test_input_sha256", "different-test"),
        ("forecast_horizon_hours", "3"),
    ],
)
def test_load_latest_complete_fails_on_cross_model_provenance_mismatch(
    parameter: str,
    different_value: str,
) -> None:
    persistence = _execution_runs("persistence", "persistence-execution")
    ridge = _execution_runs("ridge", "ridge-execution")
    ridge[0].data.params[parameter] = different_value
    ridge[1].data.params[parameter] = different_value

    with pytest.raises(ValueError, match=parameter):
        _load(
            {"persistence": persistence, "ridge": ridge},
            experiments={"Persistence": "persistence", "Ridge": "ridge"},
        )


def test_load_latest_complete_returns_exact_tidy_schema_and_coverage() -> None:
    persistence = _execution_runs(
        "persistence", "persistence-execution", cohort_rows=100
    )
    ridge = _execution_runs("ridge", "ridge-execution", cohort_rows=96)

    result = _load(
        {"persistence": persistence, "ridge": ridge},
        experiments={"Persistence": "persistence", "Ridge": "ridge"},
    )

    assert list(result.columns) == COMPARISON_METRICS_COLUMNS
    assert len(result) == 2 * 2 * 4 * 3
    assert set(result["phase"]) == {"cross_validation", "sealed_test"}
    assert set(result["metric"]) == set(METRICS)
    assert set(result["scope"]) == {"aggregate", "horizon"}
    assert result.loc[result["scope"].eq("aggregate"), "horizon"].isna().all()
    assert set(result.loc[result["scope"].eq("horizon"), "horizon"]) == {1, 2}
    assert result.loc[result["phase"].eq("cross_validation"), "cv_std"].notna().all()
    assert result.loc[result["phase"].eq("sealed_test"), "cv_std"].isna().all()
    assert str(result["completed_at_utc"].dtype) == "datetime64[ms, UTC]"
    assert str(result["horizon"].dtype) == "Int64"
    assert not result.loc[
        result["model"].eq("Persistence"), "cohort_size_differs"
    ].any()
    assert result.loc[result["model"].eq("Ridge"), "cohort_size_differs"].all()


def test_load_feature_subset_metrics_marks_model_specific_winner_per_subset() -> None:
    runs = _execution_runs("ridge", "ridge-execution")
    provenance = {
        "train_input_sha256": "train-hash",
        "test_input_sha256": "test-hash",
        "forecast_horizon_hours": "2",
        "selection_metric": "rmse",
    }
    runs[0].data.params.update(feature_count="20", selection_metric="rmse")
    tied_log1p_candidate = _run(
        "full-log1p",
        run_type="candidate_parent",
        execution_uuid="ridge-execution",
        end_time=1_850,
        params={
            **provenance,
            "subset": "full",
            "alpha": "0.1",
            "log1p": "True",
            "feature_count": "20",
        },
        metrics=_cv_metrics(2),
    )
    target_only_candidate = _run(
        "target-only",
        run_type="candidate_parent",
        execution_uuid="ridge-execution",
        end_time=1_800,
        params={
            **provenance,
            "subset": "target_station_full",
            "alpha": "1.0",
            "log1p": "False",
            "feature_count": "10",
        },
        metrics=_cv_metrics(2),
    )

    result = load_latest_complete_feature_subset_cv_metrics(
        client=cast(
            MlflowClient,
            FakeMlflowClient(
                {"ridge": [*runs, tied_log1p_candidate, target_only_candidate]}
            ),
        ),
        model_experiments={"Ridge": "ridge"},
        forecast_horizon_hours=2,
    )

    assert list(result.columns) == FEATURE_SUBSET_CV_COLUMNS
    assert len(result) == 3
    assert set(result.loc[result["is_best_within_subset"], "run_id"]) == {
        "ridge-execution-candidate",
        "target-only",
    }
    assert str(result["completed_at_utc"].dtype) == "datetime64[ms, UTC]"
    assert str(result["is_best_within_subset"].dtype) == "boolean"
