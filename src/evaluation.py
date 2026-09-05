"""Read-only MLflow queries for cross-model evaluation reporting."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config import FORECAST_HORIZON_HOURS

type EvaluationPhase = Literal[
    "cross_validation",
    "sealed_test",
    "sealed_test_quartile",
    "sealed_test_alarm",
]

METRIC_NAMES = ("mae", "rmse", "me", "r2")
MODEL_EXPERIMENTS: dict[str, str] = {
    "Persistence": "persistence",
    "Ridge": "ridge",
    "MLP": "mlp",
    "XGBoost": "xgboost",
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
    "regime",
    "lower_bound_cm",
    "upper_bound_cm",
    "scored_values",
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
    regime_definition: "_RegimeDefinition | None" = None


@dataclass(frozen=True)
class _RegimeDefinition:
    """Provenance needed to interpret post-hoc water-level regime metrics."""

    quartile_cutoffs_cm: tuple[float, float, float]
    quartile_reference_count: int
    alarm_threshold_cm: float


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
    regime_definition = _extract_regime_definition(sealed_run)
    execution = _CompleteExecution(
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
        regime_definition=regime_definition,
    )
    if regime_definition is not None:
        regime_rows = _regime_comparison_rows(execution, forecast_horizon_hours)
        if not regime_rows:
            raise ValueError(
                "sealed-test run has regime provenance but no regime metrics"
            )
    return execution


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


def _parameter_alias(run: Run, names: Sequence[str]) -> str | None:
    """Return the first non-empty value among equivalent provenance names."""
    for name in names:
        value = run.data.params.get(name)
        if value is not None and value.strip():
            return value
    return None


def _finite_parameter(run: Run, names: Sequence[str], *, label: str) -> float:
    """Parse a finite floating-point run parameter with useful diagnostics."""
    value = _parameter_alias(run, names)
    if value is None:
        raise ValueError(f"sealed-test run is missing regime parameter {label}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"regime parameter {label} is not numeric: {value!r}"
        ) from error
    if not np.isfinite(parsed):
        raise ValueError(f"regime parameter {label} is not finite")
    return parsed


def _extract_regime_definition(run: Run) -> _RegimeDefinition | None:
    """Read the optional post-hoc regime provenance from a sealed run.

    Regime diagnostics were added after the first evaluation runs. Legacy runs
    therefore return ``None`` when none of the regime parameters are present.
    Once a run advertises any regime parameter, all definition fields are
    required and validated so a partially logged execution cannot be compared
    as if its diagnostic contract were complete.
    """
    quartile_aliases = (
        (
            "regime_q25_cm",
            "target_water_level_quartile_q25_cm",
            "water_level_quartile_q25_cm",
            "quartile_q25_cm",
        ),
        (
            "regime_q50_cm",
            "target_water_level_quartile_q50_cm",
            "water_level_quartile_q50_cm",
            "quartile_q50_cm",
        ),
        (
            "regime_q75_cm",
            "target_water_level_quartile_q75_cm",
            "water_level_quartile_q75_cm",
            "quartile_q75_cm",
        ),
    )
    reference_aliases = (
        "regime_quartile_reference_count",
        "target_water_level_quartile_reference_count",
        "water_level_quartile_reference_count",
        "quartile_reference_count",
    )
    alarm_aliases = (
        "regime_alarm_threshold_cm",
        "water_level_alarm_threshold_cm",
        "target_water_level_alarm_threshold_cm",
        "alarm_threshold_cm",
    )
    any_regime_parameter = any(
        _parameter_alias(run, aliases) is not None
        for aliases in (*quartile_aliases, reference_aliases, alarm_aliases)
    )
    if not any_regime_parameter:
        return None

    cutoff_values = tuple(
        _finite_parameter(run, aliases, label=f"q{quantile}")
        for quantile, aliases in zip((25, 50, 75), quartile_aliases, strict=True)
    )
    cutoffs: tuple[float, float, float] = (
        cutoff_values[0],
        cutoff_values[1],
        cutoff_values[2],
    )
    if any(lower > upper for lower, upper in pairwise(cutoffs)):
        raise ValueError("Regime quartile cutoffs must be ordered")
    reference_value = _parameter_alias(run, reference_aliases)
    if reference_value is None:
        raise ValueError(
            "sealed-test run is missing regime parameter quartile_reference_count"
        )
    try:
        reference_count = int(reference_value)
    except ValueError as error:
        raise ValueError(
            "regime parameter quartile_reference_count is not an integer"
        ) from error
    if reference_count <= 0:
        raise ValueError("regime parameter quartile_reference_count must be positive")
    alarm_threshold = _finite_parameter(run, alarm_aliases, label="alarm_threshold_cm")
    return _RegimeDefinition(
        quartile_cutoffs_cm=cutoffs,
        quartile_reference_count=reference_count,
        alarm_threshold_cm=alarm_threshold,
    )


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
    regime_definitions = [
        execution.regime_definition
        for execution in executions
        if execution.regime_definition is not None
    ]
    if regime_definitions:
        definition_values = {
            (
                definition.quartile_cutoffs_cm,
                definition.quartile_reference_count,
                definition.alarm_threshold_cm,
            )
            for definition in regime_definitions
        }
        if len(definition_values) > 1:
            raise ValueError(
                "Included executions disagree on water-level regime definition"
            )
        if len(regime_definitions) != len(executions):
            warnings.warn(
                "Some included executions have no water-level regime diagnostics; "
                "their regime rows will be omitted.",
                stacklevel=3,
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


_REGIME_METRICS = ("mae", "rmse", "me")
_QUARTILE_REGIMES = ("Q1", "Q2", "Q3", "Q4")


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
    if execution.regime_definition is not None:
        rows.extend(_regime_comparison_rows(execution, forecast_horizon_hours))
    return rows


def _regime_parameter_or_metric(
    run: Run,
    names: Sequence[str],
    *,
    required: bool = True,
    allow_nan: bool = False,
) -> float | None:
    """Read one diagnostic number from MLflow metrics, then parameters.

    Counts are logged as metrics by the training notebooks, while a few early
    exploratory runs stored them as parameters. Supporting both locations
    keeps the evaluation report read-only and makes the migration forgiving.
    """
    for name in names:
        if name in run.data.metrics:
            try:
                value = float(run.data.metrics[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Regime metric {name!r} is not numeric") from error
            if np.isnan(value) and allow_nan:
                return value
            if not np.isfinite(value):
                raise ValueError(f"Regime metric {name!r} is not finite")
            return value
    for name in names:
        value = run.data.params.get(name)
        if value is not None and value.strip():
            try:
                parsed = float(value)
                if np.isnan(parsed) and allow_nan:
                    return parsed
                if not np.isfinite(parsed):
                    raise ValueError(f"Regime value {name!r} is not finite")
                return parsed
            except ValueError as error:
                raise ValueError(f"Regime value {name!r} is not numeric") from error
    if required:
        raise ValueError(f"sealed-test run is missing regime metric: {names[0]}")
    return None


def _regime_metric_names(
    regime: str,
    metric: str,
    horizon: int | None,
) -> tuple[str, ...]:
    """Return accepted stable names for one logged regime metric."""
    slug = regime.lower()
    suffix = "" if horizon is None else f"_horizon_{horizon:02d}"
    # The first spelling is the canonical contract. The remaining spellings
    # keep loading compatible with the short names used by early notebooks.
    return (
        f"sealed_test_{slug}_{metric}{suffix}",
        f"sealed_test_{slug}{suffix}_{metric}",
        f"sealed_test_quartile_{slug}_{metric}{suffix}",
        f"sealed_test_quartile_{slug}{suffix}_{metric}",
        f"test_quartile_{slug}_{metric}{suffix}",
        f"test_quartile_{slug}{suffix}_{metric}",
        f"regime_{slug}_{metric}{suffix}",
        f"regime_{slug}{suffix}_{metric}",
        f"test_{slug}_{metric}{suffix}",
        f"test_{slug}{suffix}_{metric}",
        f"sealed_test_regime_{slug}_{metric}{suffix}",
        f"test_regime_{slug}_{metric}{suffix}",
    )


def _regime_count_names(regime: str, horizon: int | None) -> tuple[str, ...]:
    """Return accepted stable names for one logged regime value count."""
    slug = regime.lower()
    suffix = "" if horizon is None else f"_horizon_{horizon:02d}"
    return (
        f"sealed_test_{slug}_scored_value_count{suffix}",
        f"sealed_test_{slug}{suffix}_scored_value_count",
        f"sealed_test_{slug}_scored_values{suffix}",
        f"sealed_test_{slug}{suffix}_scored_values",
        f"sealed_test_{slug}_scored_forecast_values{suffix}",
        f"sealed_test_{slug}{suffix}_scored_forecast_values",
        f"sealed_test_{slug}_count{suffix}",
        f"sealed_test_{slug}{suffix}_count",
        f"sealed_test_quartile_{slug}_scored_value_count{suffix}",
        f"sealed_test_quartile_{slug}{suffix}_scored_value_count",
        f"sealed_test_quartile_{slug}_scored_values{suffix}",
        f"sealed_test_quartile_{slug}{suffix}_scored_values",
        f"test_quartile_{slug}_scored_value_count{suffix}",
        f"test_quartile_{slug}{suffix}_scored_value_count",
        f"test_quartile_{slug}_scored_values{suffix}",
        f"test_quartile_{slug}{suffix}_scored_values",
        f"regime_{slug}_scored_value_count{suffix}",
        f"regime_{slug}{suffix}_scored_value_count",
        f"regime_{slug}_scored_values{suffix}",
        f"regime_{slug}{suffix}_scored_values",
        f"test_{slug}_scored_value_count{suffix}",
        f"test_{slug}{suffix}_scored_value_count",
        f"test_{slug}_scored_values{suffix}",
        f"test_{slug}{suffix}_scored_values",
        f"test_{slug}_scored_forecast_values{suffix}",
        f"test_{slug}{suffix}_scored_forecast_values",
        f"test_{slug}_count{suffix}",
        f"sealed_test_regime_{slug}_scored_value_count{suffix}",
        f"test_regime_{slug}_scored_value_count{suffix}",
    )


def _regime_metric_is_present(run: Run, regime: str) -> bool:
    """Return whether a sealed run advertises metrics for one regime."""
    names = _regime_metric_names(regime, "mae", None) + _regime_count_names(
        regime, None
    )
    return any(name in run.data.metrics or name in run.data.params for name in names)


def _regime_comparison_rows(
    execution: _CompleteExecution,
    forecast_horizon_hours: int,
) -> list[dict[str, object]]:
    """Convert sealed-test regime metrics into canonical comparison rows."""
    definition = execution.regime_definition
    if definition is None:  # pragma: no cover - guarded by the caller
        return []
    run = execution.sealed_run
    base = {
        "model": execution.model,
        "execution_uuid": execution.execution_uuid,
        "train_input_sha256": execution.train_input_sha256,
        "test_input_sha256": execution.test_input_sha256,
        "forecast_horizon_hours": execution.forecast_horizon_hours,
        "cohort_row_count": execution.cohort_row_count,
        "cohort_size_differs": False,
        "run_id": run.info.run_id,
        "completed_at_utc": pd.Timestamp(run.info.end_time, unit="ms", tz="UTC"),
    }
    rows: list[dict[str, object]] = []
    quartile_bounds = (
        (None, definition.quartile_cutoffs_cm[0]),
        (definition.quartile_cutoffs_cm[0], definition.quartile_cutoffs_cm[1]),
        (definition.quartile_cutoffs_cm[1], definition.quartile_cutoffs_cm[2]),
        (definition.quartile_cutoffs_cm[2], None),
    )
    for phase, regimes, bounds in (
        (
            "sealed_test_quartile",
            _QUARTILE_REGIMES,
            quartile_bounds,
        ),
        (
            "sealed_test_alarm",
            ("Alarm",),
            ((definition.alarm_threshold_cm, None),),
        ),
    ):
        if not any(_regime_metric_is_present(run, regime) for regime in regimes):
            continue
        for regime, (lower_bound, upper_bound) in zip(regimes, bounds, strict=True):
            aggregate_count = _regime_parameter_or_metric(
                run, _regime_count_names(regime, None)
            )
            if (
                aggregate_count is None
                or aggregate_count < 0
                or not float(aggregate_count).is_integer()
            ):
                raise ValueError(
                    f"Regime {regime} aggregate count must be a non-negative integer"
                )
            for metric in _REGIME_METRICS:
                value = _regime_parameter_or_metric(
                    run,
                    _regime_metric_names(regime, metric, None),
                    required=aggregate_count > 0,
                    allow_nan=aggregate_count == 0,
                )
                rows.append(
                    {
                        **base,
                        "phase": phase,
                        "metric": metric,
                        "scope": "aggregate",
                        "horizon": None,
                        "value": value,
                        "cv_std": None,
                        "regime": regime,
                        "lower_bound_cm": lower_bound,
                        "upper_bound_cm": upper_bound,
                        "scored_values": aggregate_count,
                    }
                )
            for horizon in range(1, forecast_horizon_hours + 1):
                horizon_count = _regime_parameter_or_metric(
                    run, _regime_count_names(regime, horizon)
                )
                if (
                    horizon_count is None
                    or horizon_count < 0
                    or not float(horizon_count).is_integer()
                ):
                    raise ValueError(
                        f"Regime {regime} horizon {horizon} count must be a "
                        "non-negative integer"
                    )
                for metric in _REGIME_METRICS:
                    value = _regime_parameter_or_metric(
                        run,
                        _regime_metric_names(regime, metric, horizon),
                        required=horizon_count > 0,
                        allow_nan=horizon_count == 0,
                    )
                    rows.append(
                        {
                            **base,
                            "phase": phase,
                            "metric": metric,
                            "scope": "horizon",
                            "horizon": horizon,
                            "value": value,
                            "cv_std": None,
                            "regime": regime,
                            "lower_bound_cm": lower_bound,
                            "upper_bound_cm": upper_bound,
                            "scored_values": horizon_count,
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
    frame["lower_bound_cm"] = frame["lower_bound_cm"].astype("Float64")
    frame["upper_bound_cm"] = frame["upper_bound_cm"].astype("Float64")
    frame["scored_values"] = frame["scored_values"].astype("Int64")
    frame["value"] = frame["value"].astype("Float64")
    return frame
