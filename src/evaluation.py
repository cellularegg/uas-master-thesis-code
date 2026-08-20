"""Read-only MLflow queries for cross-model evaluation reporting."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config import FORECAST_HORIZON_HOURS

type EvaluationPhase = Literal["cross_validation", "sealed_test"]

METRIC_NAMES = ("mae", "rmse", "me", "r2")
MODEL_EXPERIMENTS: dict[str, str] = {
    "Persistence": "persistence",
    "Ridge": "ridge",
    "MLP": "mlp",
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Extra Trees": "extra_trees",
    "RNN": "rnn",
}
COMPARISON_METRICS_COLUMNS = [
    "model",
    "phase",
    "run_id",
    "execution_uuid",
    "completed_at_utc",
    "metric",
    "scope",
    "horizon",
    "value",
    "cv_std",
    "train_input_sha256",
    "test_input_sha256",
    "forecast_horizon_hours",
    "cohort_row_count",
    "cohort_size_differs",
]
FEATURE_SUBSET_CV_COLUMNS = [
    "model",
    "run_id",
    "execution_uuid",
    "completed_at_utc",
    "subset",
    "feature_count",
    "selection_metric",
    "candidate_parameters",
    "is_best_within_subset",
    *[
        f"cv_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in ("mean", "std")
    ],
    "train_input_sha256",
    "test_input_sha256",
    "forecast_horizon_hours",
]

_CANDIDATE_PARAMETERS: dict[str, tuple[str, ...]] = {
    "persistence": ("persistence_column",),
    "ridge": ("subset", "alpha", "log1p"),
    "mlp": ("subset", "alpha", "hidden_layer_sizes"),
    "xgboost": (
        "subset",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ),
    "random_forest": (
        "subset",
        "max_depth",
        "n_estimators",
        "min_samples_leaf",
        "max_features",
    ),
    "extra_trees": (
        "subset",
        "max_depth",
        "n_estimators",
        "min_samples_leaf",
        "max_features",
    ),
    "rnn": ("cell_type", "sequence_length", "hidden_size", "num_layers"),
}
FEATURE_SUBSET_MODEL_EXPERIMENTS: dict[str, str] = {
    model: experiment_name
    for model, experiment_name in MODEL_EXPERIMENTS.items()
    if "subset" in _CANDIDATE_PARAMETERS[experiment_name]
}
_PROVENANCE_PARAMETERS = (
    "train_input_sha256",
    "test_input_sha256",
    "forecast_horizon_hours",
)


@dataclass(frozen=True)
class _CompleteExecution:
    """Validated pair of a selected CV parent and its sealed-test run."""

    model: str
    experiment_name: str
    execution_uuid: str
    candidate_run: Run
    candidate_runs: tuple[Run, ...]
    sealed_run: Run
    train_input_sha256: str
    test_input_sha256: str
    forecast_horizon_hours: int
    cohort_row_count: int


def load_latest_complete_comparison_metrics(
    *,
    client: MlflowClient | None = None,
    model_experiments: Mapping[str, str] = MODEL_EXPERIMENTS,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """Load the newest internally complete execution for each model from MLflow.

    This function only reads MLflow metadata. For every configured experiment it
    inspects finished sealed-test runs newest-first, matches the uniquely selected
    candidate parent from the same execution, and validates complete finite CV and
    test metrics before including that execution. Missing models and incomplete
    executions are warned about and skipped.

    Args:
        client: Optional MLflow client. When omitted, uses the currently configured
            tracking URI.
        model_experiments: Display model names mapped to MLflow experiment names.
        forecast_horizon_hours: Expected direct-forecast horizon metric coverage.

    Returns:
        A tidy frame with one aggregate row and one row per forecast horizon for
        every metric, phase, and included model. Aggregate rows have a nullable
        horizon and CV standard deviations are null for sealed-test rows.

    Raises:
        ValueError: If configuration is invalid, an experiment name is unsupported,
            or included executions disagree on artifact hashes or forecast horizon.
    """
    complete_executions = _latest_complete_executions(
        client=client,
        model_experiments=model_experiments,
        forecast_horizon_hours=forecast_horizon_hours,
    )
    rows = [
        row
        for execution in complete_executions
        for row in _comparison_rows(execution, forecast_horizon_hours)
    ]
    result = _comparison_frame(rows)
    if not result.empty:
        largest_cohort = int(result["cohort_row_count"].max())
        result["cohort_size_differs"] = result["cohort_row_count"].ne(largest_cohort)
    return result


def load_latest_complete_feature_subset_cv_metrics(
    *,
    client: MlflowClient | None = None,
    model_experiments: Mapping[str, str] = FEATURE_SUBSET_MODEL_EXPERIMENTS,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """Load aggregate CV metrics for every feature-subset candidate.

    The candidates come from the same newest complete model executions used by
    :func:`load_latest_complete_comparison_metrics`. Only models whose candidate
    contract includes a named feature subset are supported. Every candidate is
    validated against its execution provenance and aggregate CV metric contract.
    Within each model/subset group, the winner is marked by applying that model's
    existing selection metric and deterministic simplicity tie-breakers.

    Args:
        client: Optional MLflow client. When omitted, uses the currently configured
            tracking URI.
        model_experiments: Display model names mapped to subset-aware MLflow
            experiment names.
        forecast_horizon_hours: Expected direct-forecast horizon.

    Returns:
        One row per candidate with aggregate CV mean/standard-deviation metrics
        and an ``is_best_within_subset`` marker.

    Raises:
        ValueError: If an experiment is not subset-aware, candidate metrics or
            parameters are incomplete, provenance differs, candidates duplicate
            one another, or model-specific selection does not identify one winner.
    """
    unsupported = sorted(
        experiment_name
        for experiment_name in model_experiments.values()
        if "subset" not in _CANDIDATE_PARAMETERS.get(experiment_name, ())
    )
    if unsupported:
        raise ValueError(f"Experiments do not search feature subsets: {unsupported}")
    complete_executions = _latest_complete_executions(
        client=client,
        model_experiments=model_experiments,
        forecast_horizon_hours=forecast_horizon_hours,
    )
    rows = [
        row
        for execution in complete_executions
        for row in _feature_subset_candidate_rows(execution)
    ]
    return _feature_subset_cv_frame(rows)


def _latest_complete_executions(
    *,
    client: MlflowClient | None,
    model_experiments: Mapping[str, str],
    forecast_horizon_hours: int,
) -> list[_CompleteExecution]:
    """Find and validate the newest complete execution for each requested model."""
    if forecast_horizon_hours <= 0:
        raise ValueError("forecast_horizon_hours must be positive")
    unsupported = sorted(
        set(model_experiments.values()).difference(_CANDIDATE_PARAMETERS)
    )
    if unsupported:
        raise ValueError(f"Unsupported MLflow experiments: {unsupported}")

    mlflow_client = client or MlflowClient()
    complete_executions: list[_CompleteExecution] = []
    for model, experiment_name in model_experiments.items():
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            warnings.warn(
                f"MLflow experiment {experiment_name!r} was not found; "
                f"model {model!r} was skipped.",
                stacklevel=3,
            )
            continue
        sealed_runs = list(
            mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    "tags.run_type = 'sealed_test' AND attributes.status = 'FINISHED'"
                ),
                max_results=10_000,
                order_by=["attributes.end_time DESC"],
            )
        )
        candidate_runs = list(
            mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    "tags.run_type = 'candidate_parent' AND "
                    "attributes.status = 'FINISHED'"
                ),
                max_results=10_000,
                order_by=["attributes.end_time DESC"],
            )
        )
        complete = _newest_complete_execution(
            model=model,
            experiment_name=experiment_name,
            sealed_runs=sealed_runs,
            candidate_runs=candidate_runs,
            forecast_horizon_hours=forecast_horizon_hours,
        )
        if complete is None:
            warnings.warn(
                f"MLflow experiment {experiment_name!r} has no complete execution; "
                f"model {model!r} was skipped.",
                stacklevel=3,
            )
            continue
        complete_executions.append(complete)

    _validate_cross_model_provenance(
        complete_executions, expected_forecast_horizon_hours=forecast_horizon_hours
    )
    return complete_executions


def _newest_complete_execution(
    *,
    model: str,
    experiment_name: str,
    sealed_runs: Sequence[Run],
    candidate_runs: Sequence[Run],
    forecast_horizon_hours: int,
) -> _CompleteExecution | None:
    """Return the newest complete run pair, warning about rejected executions."""
    ordered_sealed_runs = sorted(
        sealed_runs,
        key=lambda run: run.info.end_time if run.info.end_time is not None else -1,
        reverse=True,
    )
    for sealed_run in ordered_sealed_runs:
        execution_uuid = sealed_run.data.tags.get("execution_uuid", "")
        try:
            return _validate_execution(
                model=model,
                experiment_name=experiment_name,
                execution_uuid=execution_uuid,
                sealed_run=sealed_run,
                candidate_runs=candidate_runs,
                forecast_horizon_hours=forecast_horizon_hours,
            )
        except ValueError as error:
            execution_label = execution_uuid or sealed_run.info.run_id
            warnings.warn(
                f"MLflow experiment {experiment_name!r} execution "
                f"{execution_label!r} is incomplete and was skipped: {error}",
                stacklevel=3,
            )
    return None


def _validate_execution(
    *,
    model: str,
    experiment_name: str,
    execution_uuid: str,
    sealed_run: Run,
    candidate_runs: Sequence[Run],
    forecast_horizon_hours: int,
) -> _CompleteExecution:
    """Validate and return one sealed-test execution and selected CV parent."""
    if not execution_uuid:
        raise ValueError("sealed-test run has no execution_uuid tag")
    if sealed_run.info.end_time is None:
        raise ValueError("sealed-test run has no completion time")

    candidate_parameter_names = _CANDIDATE_PARAMETERS[experiment_name]
    missing_candidate_parameters = [
        name for name in candidate_parameter_names if name not in sealed_run.data.params
    ]
    if missing_candidate_parameters:
        raise ValueError(
            "sealed-test run is missing selected-candidate parameters: "
            f"{missing_candidate_parameters}"
        )
    execution_candidate_runs = tuple(
        run
        for run in candidate_runs
        if run.data.tags.get("execution_uuid") == execution_uuid
    )
    selected_candidates = [
        run
        for run in execution_candidate_runs
        if all(
            run.data.params.get(name) == sealed_run.data.params[name]
            for name in candidate_parameter_names
        )
    ]
    if len(selected_candidates) != 1:
        raise ValueError(
            f"sealed-test selection matched {len(selected_candidates)} "
            "candidate_parent runs; expected exactly 1"
        )
    candidate_run = selected_candidates[0]
    if candidate_run.info.end_time is None:
        raise ValueError("selected candidate_parent run has no completion time")

    for parameter in _PROVENANCE_PARAMETERS:
        sealed_value = _required_parameter(sealed_run, parameter)
        candidate_value = _required_parameter(candidate_run, parameter)
        if sealed_value != candidate_value:
            raise ValueError(
                f"selected candidate and sealed test disagree on {parameter}"
            )
    execution_horizon = _positive_integer_parameter(
        sealed_run, "forecast_horizon_hours"
    )
    cohort_row_count = _positive_integer_parameter(sealed_run, "scored_issue_times")
    _required_finite_metrics(
        candidate_run,
        _cv_metric_names(forecast_horizon_hours),
        run_label="selected candidate_parent",
    )
    _required_finite_metrics(
        sealed_run,
        _sealed_test_metric_names(forecast_horizon_hours),
        run_label="sealed_test",
    )
    return _CompleteExecution(
        model=model,
        experiment_name=experiment_name,
        execution_uuid=execution_uuid,
        candidate_run=candidate_run,
        candidate_runs=execution_candidate_runs,
        sealed_run=sealed_run,
        train_input_sha256=_required_parameter(sealed_run, "train_input_sha256"),
        test_input_sha256=_required_parameter(sealed_run, "test_input_sha256"),
        forecast_horizon_hours=execution_horizon,
        cohort_row_count=cohort_row_count,
    )


def _required_parameter(run: Run, name: str) -> str:
    """Return a required non-empty MLflow run parameter."""
    value = run.data.params.get(name)
    if value is None or not value.strip():
        raise ValueError(f"run is missing non-empty parameter {name}")
    return value


def _positive_integer_parameter(run: Run, name: str) -> int:
    """Return a required positive integer MLflow run parameter."""
    value = _required_parameter(run, name)
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"run parameter {name} is not an integer: {value!r}"
        ) from error
    if parsed <= 0:
        raise ValueError(f"run parameter {name} must be positive")
    return parsed


def _required_finite_metrics(
    run: Run, metric_names: Sequence[str], *, run_label: str
) -> None:
    """Require the named metrics to exist and contain finite numeric values."""
    missing = sorted(set(metric_names).difference(run.data.metrics))
    if missing:
        raise ValueError(f"{run_label} is missing metrics: {missing}")
    non_finite = [
        name for name in metric_names if not np.isfinite(float(run.data.metrics[name]))
    ]
    if non_finite:
        raise ValueError(f"{run_label} contains non-finite metrics: {non_finite}")


def _cv_metric_names(forecast_horizon_hours: int) -> list[str]:
    """Return required selected-parent CV mean and standard-deviation metrics."""
    names = [
        f"cv_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in ("mean", "std")
    ]
    names.extend(
        f"cv_{metric}_horizon_{horizon:02d}_{statistic}"
        for metric in METRIC_NAMES
        for horizon in range(1, forecast_horizon_hours + 1)
        for statistic in ("mean", "std")
    )
    return names


def _sealed_test_metric_names(forecast_horizon_hours: int) -> list[str]:
    """Return required sealed-test aggregate and horizon metric names."""
    names = [f"test_{metric}" for metric in METRIC_NAMES]
    names.extend(
        f"test_{metric}_horizon_{horizon:02d}"
        for metric in METRIC_NAMES
        for horizon in range(1, forecast_horizon_hours + 1)
    )
    return names


def _validate_cross_model_provenance(
    executions: Sequence[_CompleteExecution],
    *,
    expected_forecast_horizon_hours: int,
) -> None:
    """Require included executions to describe the same artifacts and horizon."""
    comparisons = {
        "train_input_sha256": {
            execution.train_input_sha256 for execution in executions
        },
        "test_input_sha256": {execution.test_input_sha256 for execution in executions},
        "forecast_horizon_hours": {
            str(execution.forecast_horizon_hours) for execution in executions
        },
    }
    for parameter, values in comparisons.items():
        if len(values) > 1:
            raise ValueError(
                f"Included executions disagree on {parameter}: {sorted(values)}"
            )
    if executions and (
        executions[0].forecast_horizon_hours != expected_forecast_horizon_hours
    ):
        raise ValueError(
            "Included executions have forecast_horizon_hours="
            f"{executions[0].forecast_horizon_hours}, expected "
            f"{expected_forecast_horizon_hours}"
        )


def _feature_subset_candidate_rows(
    execution: _CompleteExecution,
) -> list[dict[str, object]]:
    """Validate and convert every subset candidate in one complete execution."""
    if not execution.candidate_runs:
        raise ValueError(
            f"Model {execution.model!r} execution has no candidate_parent runs"
        )
    parameter_names = _CANDIDATE_PARAMETERS[execution.experiment_name]
    ranking_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    seen_candidate_keys: set[tuple[object, ...]] = set()
    selection_metrics: set[str] = set()

    for run in execution.candidate_runs:
        if run.info.end_time is None:
            raise ValueError(
                f"Model {execution.model!r} candidate {run.info.run_id!r} "
                "has no completion time"
            )
        for parameter, expected_value in (
            ("train_input_sha256", execution.train_input_sha256),
            ("test_input_sha256", execution.test_input_sha256),
            ("forecast_horizon_hours", str(execution.forecast_horizon_hours)),
        ):
            if _required_parameter(run, parameter) != expected_value:
                raise ValueError(
                    f"Model {execution.model!r} candidate {run.info.run_id!r} "
                    f"disagrees with its execution on {parameter}"
                )
        parsed_parameters = {
            name: _parse_candidate_parameter(name, _required_parameter(run, name))
            for name in parameter_names
        }
        subset = parsed_parameters["subset"]
        if not isinstance(subset, str) or not subset:
            raise ValueError("Candidate subset must be a non-empty string")
        selection_metric = _required_parameter(run, "selection_metric")
        if selection_metric not in {"mae", "rmse"}:
            raise ValueError(
                f"Candidate selection_metric is invalid: {selection_metric!r}"
            )
        selection_metrics.add(selection_metric)
        feature_count = _positive_integer_parameter(run, "feature_count")
        _required_finite_metrics(
            run,
            [
                f"cv_{metric}_{statistic}"
                for metric in METRIC_NAMES
                for statistic in ("mean", "std")
            ],
            run_label=f"{execution.model} candidate_parent",
        )

        ranking_row: dict[str, object] = {
            "run_id": run.info.run_id,
            "feature_count": feature_count,
            **parsed_parameters,
        }
        output_row: dict[str, object] = {
            "model": execution.model,
            "run_id": run.info.run_id,
            "execution_uuid": execution.execution_uuid,
            "completed_at_utc": pd.Timestamp(run.info.end_time, unit="ms", tz="UTC"),
            "subset": subset,
            "feature_count": feature_count,
            "selection_metric": selection_metric,
            "candidate_parameters": ", ".join(
                f"{name}={run.data.params[name]}"
                for name in parameter_names
                if name != "subset"
            ),
            "is_best_within_subset": False,
            "train_input_sha256": execution.train_input_sha256,
            "test_input_sha256": execution.test_input_sha256,
            "forecast_horizon_hours": execution.forecast_horizon_hours,
        }
        for metric in METRIC_NAMES:
            for statistic in ("mean", "std"):
                metric_name = f"cv_{metric}_{statistic}"
                value = float(run.data.metrics[metric_name])
                output_row[metric_name] = value
                ranking_row[f"{metric}_{statistic}"] = value

        candidate_key = _feature_subset_candidate_key(
            execution.experiment_name, ranking_row
        )
        if candidate_key in seen_candidate_keys:
            raise ValueError(
                f"Model {execution.model!r} execution duplicates candidate "
                f"parameters: {candidate_key}"
            )
        seen_candidate_keys.add(candidate_key)
        ranking_row["__candidate_key"] = candidate_key
        ranking_rows.append(ranking_row)
        output_rows.append(output_row)

    if len(selection_metrics) != 1:
        raise ValueError(
            f"Model {execution.model!r} candidates disagree on selection_metric: "
            f"{sorted(selection_metrics)}"
        )

    ranking_frame = pd.DataFrame(ranking_rows)
    selected_run_ids: set[str] = set()
    selection_metric = next(iter(selection_metrics))
    for subset, subset_rows in ranking_frame.groupby("subset", sort=False):
        winner_key = _select_feature_subset_candidate_key(
            execution.experiment_name,
            subset_rows,
            selection_metric,
        )
        winner_rows = subset_rows.loc[
            [key == winner_key for key in subset_rows["__candidate_key"]]
        ]
        if len(winner_rows) != 1:
            raise ValueError(
                f"Model {execution.model!r} subset {subset!r} selection matched "
                f"{len(winner_rows)} candidates; expected exactly 1"
            )
        selected_run_ids.add(str(winner_rows.iloc[0]["run_id"]))

    for output_row in output_rows:
        output_row["is_best_within_subset"] = (
            str(output_row["run_id"]) in selected_run_ids
        )
    return output_rows


def _parse_candidate_parameter(name: str, value: str) -> object:
    """Normalize one MLflow string parameter for model-specific selection."""
    if name == "log1p":
        if value not in {"True", "False"}:
            raise ValueError(f"Candidate log1p parameter is invalid: {value!r}")
        return value == "True"
    if name == "max_depth":
        return None if value == "None" else int(value)
    if name in {"n_estimators", "min_samples_leaf"}:
        return int(value)
    if name in {
        "alpha",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }:
        return float(value)
    if name == "max_features":
        return value if value == "sqrt" else float(value)
    return value


def _feature_subset_candidate_key(
    experiment_name: str, candidate: Mapping[str, object]
) -> tuple[object, ...]:
    """Return the canonical candidate key emitted by a model selector."""
    if experiment_name == "ridge":
        return (
            str(candidate["subset"]),
            float(str(candidate["alpha"])),
            bool(candidate["log1p"]),
        )
    if experiment_name == "mlp":
        from src.mlp import parse_hidden_layer_sizes

        return (
            str(candidate["subset"]),
            parse_hidden_layer_sizes(str(candidate["hidden_layer_sizes"])),
            float(str(candidate["alpha"])),
        )
    if experiment_name == "xgboost":
        return (
            str(candidate["subset"]),
            int(str(candidate["max_depth"])),
            int(str(candidate["n_estimators"])),
            float(str(candidate["learning_rate"])),
            float(str(candidate["subsample"])),
            float(str(candidate["colsample_bytree"])),
            float(str(candidate["reg_lambda"])),
        )
    if experiment_name == "random_forest":
        from src.random_forest import normalize_candidate_key

        return tuple(
            normalize_candidate_key(
                subset=candidate["subset"],
                max_depth=candidate["max_depth"],
                n_estimators=candidate["n_estimators"],
                min_samples_leaf=candidate["min_samples_leaf"],
                max_features=candidate["max_features"],
            )
        )
    if experiment_name == "extra_trees":
        from src.extra_trees import normalize_candidate_key

        return tuple(
            normalize_candidate_key(
                subset=candidate["subset"],
                max_depth=candidate["max_depth"],
                n_estimators=candidate["n_estimators"],
                min_samples_leaf=candidate["min_samples_leaf"],
                max_features=candidate["max_features"],
            )
        )
    raise ValueError(f"Experiment {experiment_name!r} does not search feature subsets")


def _select_feature_subset_candidate_key(
    experiment_name: str,
    candidates: pd.DataFrame,
    selection_metric: str,
) -> tuple[object, ...]:
    """Apply one model's established selection and tie-breaking policy."""
    if experiment_name == "ridge":
        from src.ridge import select_candidate as select_ridge_candidate

        return tuple(select_ridge_candidate(candidates, selection_metric))
    if experiment_name == "mlp":
        from src.mlp import select_candidate as select_mlp_candidate

        return tuple(select_mlp_candidate(candidates, selection_metric))
    if experiment_name == "xgboost":
        from src.xgboost_model import select_candidate as select_xgboost_candidate

        return tuple(select_xgboost_candidate(candidates, selection_metric))
    if experiment_name == "random_forest":
        from src.random_forest import select_candidate as select_random_forest_candidate

        return tuple(select_random_forest_candidate(candidates, selection_metric))
    if experiment_name == "extra_trees":
        from src.extra_trees import select_candidate as select_extra_trees_candidate

        return tuple(select_extra_trees_candidate(candidates, selection_metric))
    raise ValueError(f"Experiment {experiment_name!r} does not search feature subsets")


def _feature_subset_cv_frame(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Build the feature-subset candidate frame with stable nullable dtypes."""
    frame = pd.DataFrame(rows, columns=FEATURE_SUBSET_CV_COLUMNS)
    frame["completed_at_utc"] = frame["completed_at_utc"].astype("datetime64[ms, UTC]")
    frame["feature_count"] = frame["feature_count"].astype("Int64")
    frame["is_best_within_subset"] = frame["is_best_within_subset"].astype("boolean")
    frame["forecast_horizon_hours"] = frame["forecast_horizon_hours"].astype("Int64")
    for metric in METRIC_NAMES:
        for statistic in ("mean", "std"):
            column = f"cv_{metric}_{statistic}"
            frame[column] = frame[column].astype("Float64")
    return frame.sort_values(
        ["model", "subset", "candidate_parameters"], kind="stable"
    ).reset_index(drop=True)


def _comparison_rows(
    execution: _CompleteExecution, forecast_horizon_hours: int
) -> list[dict[str, object]]:
    """Convert a validated execution into canonical tidy metric rows."""
    common = {
        "model": execution.model,
        "execution_uuid": execution.execution_uuid,
        "train_input_sha256": execution.train_input_sha256,
        "test_input_sha256": execution.test_input_sha256,
        "forecast_horizon_hours": execution.forecast_horizon_hours,
        "cohort_row_count": execution.cohort_row_count,
        "cohort_size_differs": False,
    }
    rows: list[dict[str, object]] = []
    for phase, run in (
        ("cross_validation", execution.candidate_run),
        ("sealed_test", execution.sealed_run),
    ):
        phase_common = {
            **common,
            "phase": phase,
            "run_id": run.info.run_id,
            "completed_at_utc": pd.Timestamp(run.info.end_time, unit="ms", tz="UTC"),
        }
        for metric in METRIC_NAMES:
            if phase == "cross_validation":
                aggregate_name = f"cv_{metric}_mean"
                aggregate_std: float | None = float(
                    run.data.metrics[f"cv_{metric}_std"]
                )
            else:
                aggregate_name = f"test_{metric}"
                aggregate_std = None
            rows.append(
                {
                    **phase_common,
                    "metric": metric,
                    "scope": "aggregate",
                    "horizon": None,
                    "value": float(run.data.metrics[aggregate_name]),
                    "cv_std": aggregate_std,
                }
            )
            for horizon in range(1, forecast_horizon_hours + 1):
                if phase == "cross_validation":
                    value_name = f"cv_{metric}_horizon_{horizon:02d}_mean"
                    horizon_std: float | None = float(
                        run.data.metrics[f"cv_{metric}_horizon_{horizon:02d}_std"]
                    )
                else:
                    value_name = f"test_{metric}_horizon_{horizon:02d}"
                    horizon_std = None
                rows.append(
                    {
                        **phase_common,
                        "metric": metric,
                        "scope": "horizon",
                        "horizon": horizon,
                        "value": float(run.data.metrics[value_name]),
                        "cv_std": horizon_std,
                    }
                )
    return rows


def _comparison_frame(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Build the canonical comparison frame with stable nullable dtypes."""
    frame = pd.DataFrame(rows, columns=COMPARISON_METRICS_COLUMNS)
    frame["completed_at_utc"] = frame["completed_at_utc"].astype("datetime64[ms, UTC]")
    frame["horizon"] = frame["horizon"].astype("Int64")
    frame["cv_std"] = frame["cv_std"].astype("Float64")
    frame["forecast_horizon_hours"] = frame["forecast_horizon_hours"].astype("Int64")
    frame["cohort_row_count"] = frame["cohort_row_count"].astype("Int64")
    frame["cohort_size_differs"] = frame["cohort_size_differs"].astype("boolean")
    return frame
