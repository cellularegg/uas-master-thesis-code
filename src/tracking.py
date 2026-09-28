"""Shared MLflow bookkeeping for the stage-4 training notebooks.

The training notebooks keep their candidate loops, fitting, and selection
visible; this module holds the repetitive run tagging, metric flattening,
completeness verification, sealed-test tables, and figure logging they share.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.dataset import JoinedDataset
from src.metrics import metric_tables
from src.plots import (
    cv_error_boxplots_figure,
    predicted_vs_actual_figure,
    test_error_boxplots_figure,
)
from src.regime_persistence import (
    regime_mlflow_metrics,
    regime_mlflow_params,
    sealed_test_regime_tables,
)
from src.training import summarize_cv_metrics, validate_predictions

METRIC_NAMES = ("mae", "rmse", "me", "r2")
CV_SUMMARY_COLUMNS = (
    "mae_mean",
    "mae_std",
    "rmse_mean",
    "rmse_std",
    "me_mean",
    "me_std",
    "r2_mean",
    "r2_std",
)


class SealedTestTables(NamedTuple):
    """Sealed-test metric tables returned by :func:`sealed_test_tables`."""

    aggregate: pd.DataFrame
    per_horizon: pd.DataFrame
    regime_definition: dict[str, object]
    regime_aggregate: pd.DataFrame
    regime_horizon: pd.DataFrame


class CandidateCV(NamedTuple):
    """One candidate's cross-validation result from :func:`run_cv_candidate`."""

    parent_metrics: dict[str, float]
    fold_aggregate_metrics: pd.DataFrame
    fold_horizon_rows: list[pd.DataFrame]


def run_tags(
    phase: str, run_type: str, execution_uuid: str, **extra: object
) -> dict[str, str]:
    """Build the tags every training-notebook run carries.

    Args:
        phase: Pipeline phase, e.g. ``"cv"`` or ``"test"``.
        run_type: Run role, e.g. ``"candidate_parent"``, ``"fold"``, or
            ``"sealed_test"``.
        execution_uuid: Identifier shared by every run of one notebook execution.
        **extra: Candidate-identifying tags; values are converted with ``str``.

    Returns:
        The tag mapping, with ``execution_uuid`` last.
    """
    return {
        "phase": phase,
        "run_type": run_type,
        **{name: str(value) for name, value in extra.items()},
        "execution_uuid": execution_uuid,
    }


def metric_log_dict(
    prefix: str, aggregate: pd.DataFrame, per_horizon: pd.DataFrame
) -> dict[str, float]:
    """Flatten aggregate and per-horizon metric tables into MLflow metrics.

    Args:
        prefix: Metric-name prefix, e.g. ``"fold"`` or ``"test"``.
        aggregate: One-row aggregate table from :func:`src.metrics.metric_tables`.
        per_horizon: Per-horizon table from :func:`src.metrics.metric_tables`.

    Returns:
        ``{prefix}_{metric}`` and ``{prefix}_{metric}_horizon_{hh}`` values for
        MAE, RMSE, ME, and R².
    """
    return {
        **{
            f"{prefix}_{metric}": float(aggregate.iloc[0][metric])
            for metric in METRIC_NAMES
        },
        **{
            f"{prefix}_{metric}_horizon_{int(str(row.horizon_hours)):02d}": float(
                getattr(row, metric)
            )
            for row in per_horizon.itertuples()
            for metric in METRIC_NAMES
        },
    }


def fold_window_params(
    timestamps: pd.Series,
    train_indices: Sequence[int] | np.ndarray | None,
    validation_indices: Sequence[int] | np.ndarray,
    *,
    gap_rows: int,
    forecast_horizon_hours: int,
) -> dict[str, object]:
    """Describe one validation fold's row window for MLflow params.

    Args:
        timestamps: Issue timestamps of the positionally indexed training rows.
        train_indices: Fold training positions, or ``None`` for a model that
            never fits (the training fields are then omitted).
        validation_indices: Fold validation positions.
        gap_rows: Embargo rows between training and validation windows.
        forecast_horizon_hours: Configured direct-forecast horizon.

    Returns:
        Row counts, gap, horizon, first/last timestamps, and first/last indices.
    """
    params: dict[str, object] = {
        "validation_rows": len(validation_indices),
        "gap_rows": gap_rows,
        "forecast_horizon_hours": forecast_horizon_hours,
        "validation_start": timestamps.iloc[int(validation_indices[0])].isoformat(),
        "validation_end": timestamps.iloc[int(validation_indices[-1])].isoformat(),
        "validation_index_start": int(validation_indices[0]),
        "validation_index_end": int(validation_indices[-1]),
    }
    if train_indices is not None:
        params |= {
            "train_rows": len(train_indices),
            "train_start": timestamps.iloc[int(train_indices[0])].isoformat(),
            "train_end": timestamps.iloc[int(train_indices[-1])].isoformat(),
            "train_index_start": int(train_indices[0]),
            "train_index_end": int(train_indices[-1]),
        }
    return params


def cv_result_row(
    candidate: Mapping[str, object],
    parent_metrics: Mapping[str, float],
    *,
    include_horizons: bool = True,
) -> dict[str, object]:
    """Build one candidate's row of the in-memory CV result table.

    Args:
        candidate: Candidate-identifying columns, in display order.
        parent_metrics: Output of :func:`src.training.summarize_cv_metrics`.
        include_horizons: Whether to keep the per-horizon ``cv_*`` summaries.

    Returns:
        The candidate columns, the aggregate summary renamed from
        ``cv_{metric}_{stat}`` to ``{metric}_{stat}``, and optionally the
        per-horizon summaries under their original names.
    """
    aggregate_names = {f"cv_{column}" for column in CV_SUMMARY_COLUMNS}
    return {
        **candidate,
        **{column: parent_metrics[f"cv_{column}"] for column in CV_SUMMARY_COLUMNS},
        **(
            {
                name: value
                for name, value in parent_metrics.items()
                if name not in aggregate_names
            }
            if include_horizons
            else {}
        ),
    }


def run_cv_candidate(
    run_name: str,
    fit_predict: Callable[[np.ndarray, np.ndarray], tuple[pd.DataFrame, Any]],
    *,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    execution_uuid: str,
    tags: Mapping[str, object],
    parent_params: Mapping[str, object],
    fold_params: Mapping[str, object],
    target_columns: Sequence[str],
    station_id: str,
    window_timestamps: pd.Series | None,
    gap_rows: int,
    forecast_horizon_hours: int,
    log_train_window: bool = True,
    progress: Any = None,
    describe_fold: Callable[[int], str] | None = None,
) -> CandidateCV:
    """Cross-validate one candidate as an MLflow parent run with nested fold runs.

    For every fold, ``fit_predict`` fits on the training positions and returns
    the validation targets and predictions; the fold is then scored and logged.

    Args:
        run_name: Parent run name; fold runs append ``_fold_{n}``.
        fit_predict: Maps ``(train_indices, validation_indices)`` to the
            validation target frame and the raw predictions.
        folds: Chronological ``(train_indices, validation_indices)`` pairs.
        execution_uuid: The notebook execution's ``execution_uuid`` tag.
        tags: Candidate-identifying tags added to the parent and fold runs.
        parent_params: Params logged on the parent run.
        fold_params: Params logged on every fold run.
        target_columns: Ordered direct-forecast target columns.
        station_id: Target station identifier.
        window_timestamps: Issue timestamps of the fold positions, logged via
            :func:`fold_window_params`; ``None`` logs only row counts and gap.
        gap_rows: Embargo rows between training and validation windows.
        forecast_horizon_hours: Configured direct-forecast horizon.
        log_train_window: Whether the window params include training fields.
        progress: Optional ``tqdm`` bar advanced once per scored fold.
        describe_fold: Optional progress description for each fold number.

    Returns:
        The logged CV summary metrics, the per-fold aggregate metrics, and the
        per-fold horizon metric tables.

    Raises:
        ValueError: If a fold's predictions have the wrong shape or are
            non-finite.
    """
    fold_aggregate_rows = []
    fold_horizon_rows = []
    with mlflow.start_run(
        run_name=run_name,
        nested=False,
        tags=run_tags("cv", "candidate_parent", execution_uuid, **tags),
    ):
        mlflow.log_params(
            {"phase": "cv", "run_type": "candidate_parent", **parent_params}
        )
        for fold_number, (train_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            with mlflow.start_run(
                run_name=f"{run_name}_fold_{fold_number}",
                nested=True,
                tags=run_tags("cv", "fold", execution_uuid, **tags, fold=fold_number),
            ):
                if progress is not None and describe_fold is not None:
                    progress.set_description(describe_fold(fold_number))
                actual, raw_predictions = fit_predict(train_indices, validation_indices)
                predictions = validate_predictions(
                    raw_predictions,
                    expected_rows=len(validation_indices),
                    target_columns=target_columns,
                    artifact_name="fold",
                )
                aggregate, per_horizon = metric_tables(
                    actual,
                    predictions,
                    target_columns=target_columns,
                    station_id=station_id,
                )
                fold_aggregate_rows.append(aggregate.iloc[0])
                fold_horizon_rows.append(per_horizon)
                if progress is not None:
                    progress.update(1)
                window: dict[str, object] = (
                    fold_window_params(
                        window_timestamps,
                        train_indices if log_train_window else None,
                        validation_indices,
                        gap_rows=gap_rows,
                        forecast_horizon_hours=forecast_horizon_hours,
                    )
                    if window_timestamps is not None
                    else {
                        "train_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                        "gap_rows": gap_rows,
                    }
                )
                mlflow.log_params(
                    {
                        "phase": "cv",
                        "run_type": "fold",
                        **fold_params,
                        "fold": fold_number,
                        **window,
                    }
                )
                mlflow.log_metrics(metric_log_dict("fold", aggregate, per_horizon))
        fold_aggregate_metrics = pd.DataFrame(fold_aggregate_rows)
        parent_metrics = summarize_cv_metrics(
            fold_aggregate_metrics, pd.concat(fold_horizon_rows, ignore_index=True)
        )
        mlflow.log_metrics(parent_metrics)
    return CandidateCV(parent_metrics, fold_aggregate_metrics, fold_horizon_rows)


def verify_cv_runs(
    *,
    experiment_name: str,
    execution_uuid: str,
    expected_candidate_keys: set[tuple[Any, ...]],
    n_validation_folds: int,
    run_key: Callable[[pd.Series], tuple[Any, ...]],
) -> None:
    """Check that this execution logged every candidate and fold run.

    Args:
        experiment_name: MLflow experiment the CV runs were logged to.
        execution_uuid: The notebook execution's ``execution_uuid`` tag.
        expected_candidate_keys: Every candidate key the search should produce.
        n_validation_folds: Folds expected under each candidate.
        run_key: Rebuilds a candidate key from one ``mlflow.search_runs`` row.

    Raises:
        ValueError: If the experiment is missing, or the logged parent or fold
            runs do not match the expected candidates and folds exactly.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment {experiment_name!r} was not found")

    def current_runs(run_type: str) -> pd.DataFrame:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"tags.execution_uuid = '{execution_uuid}' "
                "and tags.phase = 'cv' "
                f"and tags.run_type = '{run_type}'"
            ),
        )
        assert isinstance(runs, pd.DataFrame)
        return runs

    parent_runs = current_runs("candidate_parent")
    fold_runs = current_runs("fold")
    parent_keys = {run_key(row) for _, row in parent_runs.iterrows()}
    if (
        len(parent_runs) != len(expected_candidate_keys)
        or parent_keys != expected_candidate_keys
    ):
        raise ValueError(
            "Current execution must produce every expected candidate: "
            f"expected {len(expected_candidate_keys)} "
            f"{sorted(expected_candidate_keys, key=repr)}, "
            f"got {len(parent_runs)} {sorted(parent_keys, key=repr)}"
        )
    expected_fold_count = len(expected_candidate_keys) * n_validation_folds
    if len(fold_runs) != expected_fold_count:
        raise ValueError(
            f"Current execution must produce {expected_fold_count} nested fold "
            f"runs, got {len(fold_runs)}"
        )
    fold_keys = {
        (*run_key(row), int(row["tags.fold"])) for _, row in fold_runs.iterrows()
    }
    expected_fold_keys = {
        (*candidate_key, fold_number)
        for candidate_key in expected_candidate_keys
        for fold_number in range(1, n_validation_folds + 1)
    }
    if fold_keys != expected_fold_keys:
        raise ValueError(
            "Current execution fold runs do not cover every candidate and fold"
        )


def cv_results_table(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_candidate_keys: set[tuple[Any, ...]],
    key_columns: Sequence[str],
    normalize_key: Callable[..., tuple[Any, ...]] = lambda *values: tuple(values),
) -> pd.DataFrame:
    """Build the in-memory CV result table and check it covers every candidate.

    Args:
        rows: One :func:`cv_result_row` per evaluated candidate.
        expected_candidate_keys: Every candidate key the search should produce.
        key_columns: Columns whose values, in order, form a candidate key.
        normalize_key: Maps one row's ``key_columns`` values to its candidate key.

    Returns:
        The result table in evaluation order.

    Raises:
        ValueError: If the table is incomplete or its keys differ from the
            expected candidates.
    """
    cv_results = pd.DataFrame(list(rows))
    if len(cv_results) != len(expected_candidate_keys):
        raise ValueError("The in-memory CV result table is incomplete")
    result_keys = {
        normalize_key(*values)
        for values in zip(*(cv_results[column] for column in key_columns), strict=True)
    }
    if result_keys != expected_candidate_keys:
        raise ValueError(
            "The in-memory CV result table does not match the candidate set"
        )
    return cv_results


def verified_cv_results(
    rows: Iterable[Mapping[str, object]],
    *,
    experiment_name: str,
    execution_uuid: str,
    expected_candidate_keys: set[tuple[Any, ...]],
    n_validation_folds: int,
    run_key: Callable[[pd.Series], tuple[Any, ...]],
    key_columns: Sequence[str],
    normalize_key: Callable[..., tuple[Any, ...]] = lambda *values: tuple(values),
    sort: bool = True,
) -> pd.DataFrame:
    """Verify the logged CV runs and return the checked in-memory result table.

    Combines :func:`verify_cv_runs` and :func:`cv_results_table`.

    Args:
        rows: One :func:`cv_result_row` per evaluated candidate.
        experiment_name: MLflow experiment the CV runs were logged to.
        execution_uuid: The notebook execution's ``execution_uuid`` tag.
        expected_candidate_keys: Every candidate key the search should produce.
        n_validation_folds: Folds expected under each candidate.
        run_key: Rebuilds a candidate key from one ``mlflow.search_runs`` row.
        key_columns: Result-table columns whose values form a candidate key.
        normalize_key: Maps one row's ``key_columns`` values to its candidate key.
        sort: Whether to sort the table stably by ``key_columns``.

    Returns:
        The CV result table, sorted by ``key_columns`` when ``sort`` is set.

    Raises:
        ValueError: If runs or table rows do not match the expected candidates.
    """
    verify_cv_runs(
        experiment_name=experiment_name,
        execution_uuid=execution_uuid,
        expected_candidate_keys=expected_candidate_keys,
        n_validation_folds=n_validation_folds,
        run_key=run_key,
    )
    cv_results = cv_results_table(
        rows,
        expected_candidate_keys=expected_candidate_keys,
        key_columns=key_columns,
        normalize_key=normalize_key,
    )
    if sort:
        cv_results = cv_results.sort_values(
            list(key_columns), kind="stable"
        ).reset_index(drop=True)
    return cv_results


def selected_cv_metric(
    cv_results: pd.DataFrame, selected: Mapping[str, object], metric: str
) -> float:
    """Return the selected candidate's mean CV metric.

    Args:
        cv_results: Table built by :func:`cv_results_table`.
        selected: The selected candidate's key columns and values.
        metric: Selection metric name, e.g. ``"rmse"``.

    Returns:
        The ``{metric}_mean`` value of the selected candidate's row.
    """
    mask = pd.Series(True, index=cv_results.index)
    for column, value in selected.items():
        mask &= cv_results[column] == value
    return float(cv_results.loc[mask, f"{metric}_mean"].iloc[0])


def sealed_test_tables(
    actual: pd.DataFrame,
    predictions: np.ndarray,
    *,
    dataset: JoinedDataset,
    target_columns: Sequence[str],
    station_id: str,
    model_label: str | None = None,
) -> SealedTestTables:
    """Score sealed-test predictions overall, per horizon, and per regime.

    Args:
        actual: Sealed-test actual target frame.
        predictions: Sealed-test predictions ordered like ``target_columns``.
        dataset: Joined dataset supplying the training quartile cutoffs.
        target_columns: Ordered direct-forecast target columns.
        station_id: Target station identifier.
        model_label: When given, non-finite aggregate or horizon metrics raise
            an error naming this model.

    Returns:
        Aggregate, per-horizon, and regime metric tables plus the regime
        definition.

    Raises:
        ValueError: If ``model_label`` is given and a metric is non-finite.
    """
    aggregate, per_horizon = metric_tables(
        actual, predictions, target_columns=target_columns, station_id=station_id
    )
    regime_definition, regime_aggregate, regime_horizon = sealed_test_regime_tables(
        actual,
        predictions,
        target_columns=list(target_columns),
        station_id=station_id,
        quartile_cutoffs_cm=dataset.target_water_level_quartile_cutoffs_cm,
        quartile_reference_count=dataset.target_water_level_quartile_reference_count,
    )
    if model_label is not None:
        if not np.isfinite(aggregate[list(METRIC_NAMES)].to_numpy()).all():
            raise ValueError(f"{model_label} reported non-finite aggregate metrics")
        if not np.isfinite(per_horizon[list(METRIC_NAMES)].to_numpy()).all():
            raise ValueError(f"{model_label} reported non-finite horizon metrics")
    return SealedTestTables(
        aggregate, per_horizon, regime_definition, regime_aggregate, regime_horizon
    )


def log_evaluation_figures(
    model_label: str,
    title_detail: str | None,
    *,
    cv_horizon_metrics: pd.DataFrame,
    test_rows: pd.DataFrame,
    test_predictions: np.ndarray,
    per_horizon_metrics: pd.DataFrame,
    target_columns: Sequence[str],
    show: bool = True,
) -> None:
    """Log the CV-error, test-error, and predicted-vs-actual figures.

    Must be called inside the active sealed-test MLflow run.

    Args:
        model_label: Model name leading each figure title.
        title_detail: Selected-candidate description appended after an em dash,
            or ``None`` for no suffix.
        cv_horizon_metrics: The selected candidate's per-fold horizon metrics.
        test_rows: Sealed-test rows with timestamps and targets.
        test_predictions: Sealed-test predictions.
        per_horizon_metrics: Sealed-test per-horizon metrics.
        target_columns: Ordered direct-forecast target columns.
        show: Whether to display each figure before closing it.
    """
    suffix = "" if title_detail is None else f" — {title_detail}"
    target_columns = list(target_columns)
    figures = {
        "cv_rmse_mae_boxplots.png": cv_error_boxplots_figure(
            cv_horizon_metrics, target_columns, title=f"{model_label} CV errors{suffix}"
        ),
        "test_error_boxplots.png": test_error_boxplots_figure(
            test_rows,
            test_predictions,
            per_horizon_metrics,
            target_columns,
            title=f"{model_label} final-test errors{suffix}",
        ),
        "test_predicted_vs_actual.png": predicted_vs_actual_figure(
            test_rows[target_columns],
            test_predictions,
            target_columns,
            title=f"{model_label} predicted vs actual{suffix}",
        ),
    }
    for artifact_file, figure in figures.items():
        mlflow.log_figure(figure, artifact_file)
        if show:
            plt.show()
        plt.close(figure)


def log_sealed_test_run(
    run_name: str,
    sealed: SealedTestTables,
    *,
    execution_uuid: str,
    tags: Mapping[str, object],
    params: Mapping[str, object],
    model_label: str,
    title_detail: str | None,
    cv_horizon_metrics: pd.DataFrame,
    test_rows: pd.DataFrame,
    test_predictions: np.ndarray,
    target_columns: Sequence[str],
    show_figures: bool = True,
    extra_figures: Mapping[str, Callable[[], Figure]] | None = None,
) -> None:
    """Log the single sealed-test run: params, metrics, and figures.

    Args:
        run_name: Sealed-test run name.
        sealed: Tables from :func:`sealed_test_tables`.
        execution_uuid: The notebook execution's ``execution_uuid`` tag.
        tags: Extra run tags.
        params: Run params; the phase, run type, and regime definition are added.
        model_label: Model name leading each figure title.
        title_detail: Selected-candidate description for figure titles.
        cv_horizon_metrics: The selected candidate's per-fold horizon metrics.
        test_rows: Sealed-test rows with timestamps and targets.
        test_predictions: Sealed-test predictions.
        target_columns: Ordered direct-forecast target columns.
        show_figures: Whether to display each figure before closing it.
        extra_figures: Further figures to log after the standard three, as
            artifact file name to figure factory.
    """
    with mlflow.start_run(
        run_name=run_name,
        nested=False,
        tags=run_tags("test", "sealed_test", execution_uuid, **tags),
    ):
        mlflow.log_params(
            {
                "phase": "test",
                "run_type": "sealed_test",
                **params,
                **regime_mlflow_params(sealed.regime_definition),
            }
        )
        mlflow.log_metrics(
            {
                **regime_mlflow_metrics(sealed.regime_aggregate, sealed.regime_horizon),
                **metric_log_dict("test", sealed.aggregate, sealed.per_horizon),
            }
        )
        log_evaluation_figures(
            model_label,
            title_detail,
            cv_horizon_metrics=cv_horizon_metrics,
            test_rows=test_rows,
            test_predictions=test_predictions,
            per_horizon_metrics=sealed.per_horizon,
            target_columns=target_columns,
            show=show_figures,
        )
        for artifact_file, make_figure in (extra_figures or {}).items():
            figure = make_figure()
            mlflow.log_figure(figure, artifact_file)
            if show_figures:
                plt.show()
            plt.close(figure)


@contextmanager
def backdated_run(
    client: MlflowClient,
    *,
    experiment_id: str,
    run_name: str,
    tags: Mapping[str, str],
    start_time_ms: int,
    end_time_ms: int,
) -> Iterator[Run]:
    """Open a run whose start and end times were measured earlier.

    Used where work ran before logging (e.g. in worker processes): the run is
    created with ``start_time_ms``, activated for logging, and terminated with
    ``end_time_ms`` once the block exits without an error.

    Args:
        client: MLflow client used to create and terminate the run.
        experiment_id: Experiment receiving the run.
        run_name: Run name.
        tags: Run tags.
        start_time_ms: Recorded start time in epoch milliseconds.
        end_time_ms: Recorded end time in epoch milliseconds.

    Yields:
        The created run, active for ``mlflow.log_*`` calls.
    """
    run = client.create_run(
        experiment_id=experiment_id,
        start_time=start_time_ms,
        run_name=run_name,
        tags=dict(tags),
    )
    with mlflow.start_run(run_id=run.info.run_id):
        yield run
    client.set_terminated(run.info.run_id, end_time=end_time_ms)
