"""Causal univariate ARIMA fitting on full observed history, rolling forecasts, and saved contracts."""

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import time_ns
from typing import Any

import numpy as np
import pandas as pd
from joblib import (  # type: ignore[import-untyped]
    Parallel,
    delayed,
    dump,
    load,
    parallel_config,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]

from src.dataset import JoinedFeatureContract
from src.metrics import metric_tables
from src.model_selection import select_candidate as rank_candidate
from src.training import summarize_cv_metrics, validate_predictions

_HOUR = pd.Timedelta(hours=1)
_TRAINING_DATA_POLICY = (
    "observed_levels_through_last_train_issue_plus_horizon; imputed_as_missing; "
    "full_history_from_first_observation"
)
_UPDATE_POLICY = "fixed_parameters; missing_observation_state_transition"
_FUTURE_PREDICTOR_POLICY = "none; univariate"
_ESTIMATOR = "statsmodels.tsa.statespace.sarimax.SARIMAX"


@dataclass(frozen=True)
class ArimaCandidate:
    """One fixed nonseasonal configuration.

    Attributes:
        order: Nonnegative p, d, q orders.
        intercept: Include a constant in the SARIMAX equation; only when d is 0.
    """

    order: tuple[int, int, int]
    intercept: bool

    def __post_init__(self) -> None:
        """Reject malformed configurations before fitting."""
        if len(self.order) != 3 or any(type(v) is not int or v < 0 for v in self.order):
            raise ValueError("order must contain three nonnegative integers")
        if type(self.intercept) is not bool:
            raise ValueError("intercept must be Boolean")
        if self.intercept and self.order[1] != 0:
            # A constant in a differenced model is a drift, not a level.
            raise ValueError("An intercept is only allowed when d is zero")

    def parameters(self) -> dict[str, Any]:
        """Return candidate identity for reporting and selection.

        Returns:
            Order string, intercept flag, and the separate p, d, q orders.
        """
        p, d, q = self.order
        return {
            "order": str(self.order),
            "intercept": self.intercept,
            "p": p,
            "d": d,
            "q": q,
        }


def candidate_grid() -> list[ArimaCandidate]:
    """Return every fixed order with an optional intercept when d is zero.

    Returns:
        Forty-eight candidates: p,q in 0..3 and d in {0, 1}, with and without
        an intercept for d=0 and without one for d=1.
    """
    return [
        ArimaCandidate((p, d, q), intercept)
        for p, d, q in product(range(4), range(2), range(4))
        for intercept in ((False, True) if d == 0 else (False,))
    ]


@dataclass(frozen=True)
class FittedArima:
    """A training-only state snapshot with a timestamped input/output contract.

    Attributes:
        results: Statsmodels state-space results selected by ARIMA.
        station_id: Station whose hourly levels are modeled.
        target_columns: Ordered future-hour outputs.
        state_time: Last hour incorporated into the stored filter state.
        order: Selected nonseasonal order.
        seasonal_order: Selected seasonal order.
        intercept: Whether the fitted equation includes a constant.
    """

    results: Any
    station_id: str
    target_columns: tuple[str, ...]
    state_time: pd.Timestamp
    order: tuple[int, ...]
    seasonal_order: tuple[int, ...]
    intercept: bool = False


def _utc_hour(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError("Timestamps must lie on the hourly grid")
    return timestamp


def _validate_hourly_series(history: pd.Series) -> None:
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("History must have a timezone-aware DatetimeIndex")
    if history.empty or history.index.hasnans or not history.index.is_unique:
        raise ValueError("History must be nonempty with unique valid timestamps")
    expected = pd.date_range(history.index[0], history.index[-1], freq="h")
    if not history.index.equals(expected) or not history.index.equals(
        history.index.floor("h")
    ):
        raise ValueError("History must be sorted on a complete hourly grid")
    if np.isinf(history.to_numpy(dtype=float)).any():
        raise ValueError("History contains infinite values")


def observed_hourly_history(frame: pd.DataFrame) -> pd.Series:
    """Recover actual observations from one unfiltered joined-data partition.

    Stage-2 fill eligibility depends on the eventual length of an outage.
    Mask those fills so forecasting never depends on future gap boundaries.
    Missing timestamps are inserted as missing hours, never dropped.

    Args:
        frame: UTC-indexed frame with water_level and Boolean imputed columns.

    Returns:
        Hourly observed levels with NaNs for absent or previously imputed values.

    Raises:
        ValueError: If timestamps, levels, or imputation flags are invalid.
    """
    if not {"water_level", "imputed"}.issubset(frame.columns):
        raise ValueError("History requires water_level and imputed columns")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("History must have timezone-aware timestamps")
    if frame.empty or index.hasnans or not index.is_unique:
        raise ValueError("History must have unique nonempty timestamps")
    if not index.equals(index.floor("h")):
        raise ValueError("History timestamps must lie on the hourly grid")
    if frame["imputed"].isna().any() or not pd.api.types.is_bool_dtype(
        frame["imputed"].dtype
    ):
        raise ValueError("History imputed flags must be non-null Booleans")
    values = pd.to_numeric(frame["water_level"], errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("History contains infinite values")
    values = values.mask(frame["imputed"])
    values.index = index.tz_convert("UTC")
    return values.sort_index().asfreq("h").rename("water_level")


def training_history(history: pd.Series, *, cutoff: pd.Timestamp) -> pd.Series:
    """Select every observed training hour available at a cutoff.

    Missing and imputed hours stay missing; the state-space likelihood skips
    them. Only leading missing hours before the first observation are dropped.

    Args:
        history: Observed hourly levels, with NaNs for missing observations.
        cutoff: Inclusive final hour permitted for fitting.

    Returns:
        Hourly levels from the first observation through the cutoff.

    Raises:
        ValueError: If the history or cutoff is invalid or nothing is observed.
    """
    _validate_hourly_series(history)
    cutoff = _utc_hour(cutoff)
    if cutoff < history.index[0] or cutoff > history.index[-1]:
        raise ValueError("Training cutoff is outside the supplied history")
    past = history.loc[:cutoff]
    first_valid = past.first_valid_index()
    if first_valid is None:
        raise ValueError("No observed training history at cutoff")
    return past.loc[first_valid:]


def latest_complete_segment(history: pd.Series) -> pd.Series:
    """Return the trailing run of consecutive observed hours.

    Used only for ACF/PACF diagnostics, which require gap-free input.

    Args:
        history: Hourly levels with NaNs for missing observations.

    Returns:
        The longest suffix of ``history`` after its last missing hour, ending
        at its last observation.

    Raises:
        ValueError: If the history has no observation.
    """
    last_valid = history.last_valid_index()
    if last_valid is None:
        raise ValueError("History has no observation")
    past = history.loc[:last_valid]
    missing = np.flatnonzero(past.isna().to_numpy())
    return past.iloc[int(missing[-1]) + 1 :] if len(missing) else past


def fit_arima(
    history: pd.Series,
    *,
    cutoff: pd.Timestamp,
    contract: JoinedFeatureContract,
    candidate: ArimaCandidate,
    maxiter: int = 200,
) -> tuple[FittedArima, dict[str, object]]:
    """Fit a recursive ARIMA model on the full observed history through a cutoff.

    Args:
        history: Observed hourly training history, NaN where missing or imputed.
        cutoff: Final permitted observation hour; the fitted state time.
        contract: Validated station and ordered forecast target contract.
        candidate: Fixed order and intercept.
        maxiter: Optimizer iteration limit.

    Returns:
        Training-only state snapshot and diagnostics for display/MLflow.

    Raises:
        ValueError: If training history is inadequate or no finite fit is found.
    """
    if type(maxiter) is not int or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    segment = training_history(history, cutoff=cutoff)
    results = SARIMAX(
        segment.to_numpy(dtype=float),
        order=candidate.order,
        seasonal_order=(0, 0, 0, 0),
        trend="c" if candidate.intercept else "n",
        simple_differencing=False,
    ).fit(maxiter=maxiter, disp=False)
    if not np.isfinite(results.params).all():
        raise ValueError("ARIMA fitted non-finite parameters")
    model = FittedArima(
        results=results,
        station_id=contract.station_id,
        target_columns=contract.target_columns,
        state_time=segment.index[-1],
        order=candidate.order,
        seasonal_order=(0, 0, 0, 0),
        intercept=candidate.intercept,
    )
    observed_hours = int(segment.notna().sum())
    diagnostics: dict[str, object] = {
        "order": str(model.order),
        "intercept": candidate.intercept,
        "maxiter": maxiter,
        "fit_start": segment.index[0].isoformat(),
        "fit_cutoff": segment.index[-1].isoformat(),
        "fit_hours": len(segment),
        "fit_observed_hours": observed_hours,
        "fit_missing_hours": len(segment) - observed_hours,
        "seasonal_order": str(model.seasonal_order),
        "aic": float(results.aic),
        "aicc": float(results.aicc),
        "bic": float(results.bic),
        "converged": bool(results.mle_retvals.get("converged", False)),
    }
    return model, diagnostics


def rolling_forecasts(
    model: FittedArima,
    history: pd.Series,
    issue_times: Iterable[pd.Timestamp],
) -> np.ndarray:
    """Forecast each origin using fixed parameters and observations through t.

    Missing hours advance the Kalman state without a measurement update.
    This function never mutates the training snapshot, re-estimates parameters,
    or consumes observations later than the origin being forecast.

    Args:
        model: Training-only state snapshot.
        history: Complete hourly grid covering state time through all origins.
        issue_times: Strictly increasing UTC forecast origins after state time.

    Returns:
        Finite forecast matrix ordered by origin and contractual future hour.

    Raises:
        ValueError: If origins precede the model, are unordered, or lack history.
    """
    _validate_hourly_series(history)
    issues = pd.DatetimeIndex(list(issue_times))
    if issues.empty or issues.tz is None or issues.hasnans:
        raise ValueError("Issue times must be nonempty timezone-aware timestamps")
    issues = issues.tz_convert("UTC")
    if not issues.is_unique or not issues.is_monotonic_increasing:
        raise ValueError("Issue times must be unique and increasing")
    if not issues.equals(issues.floor("h")):
        raise ValueError("Issue times must lie on the hourly grid")
    state_time = _utc_hour(model.state_time)
    if issues[0] <= state_time:
        raise ValueError("Issue times must follow the fitted state time")
    if history.index[0] > state_time + _HOUR or history.index[-1] < issues[-1]:
        raise ValueError("History must cover every hour from the state to the origins")
    results = model.results
    forecasts = np.empty((len(issues), len(model.target_columns)), dtype=float)
    for row, issue in enumerate(issues):
        observations = history.loc[state_time + _HOUR : issue].to_numpy(dtype=float)
        results = results.extend(observations)
        forecasts[row] = np.asarray(results.forecast(len(model.target_columns)))
        state_time = issue
    return validate_predictions(
        forecasts,
        expected_rows=len(issues),
        target_columns=model.target_columns,
        artifact_name="rolling ARIMA",
    )


def _model_contract(model: FittedArima) -> dict[str, object]:
    return {
        "schema_version": "5.0",
        "estimator": _ESTIMATOR,
        "station_id": model.station_id,
        "input_columns": [f"{model.station_id}__water_level"],
        "target_columns": list(model.target_columns),
        "forecast_horizon_hours": len(model.target_columns),
        "order": list(model.order),
        "seasonal_order": [0, 0, 0, 0],
        "trend": "c" if model.intercept else "n",
        "intercept": model.intercept,
        "state_time": model.state_time.isoformat(),
        "input_frequency": "h",
        "timezone": "UTC",
        "unit": "cm",
        "update_policy": _UPDATE_POLICY,
        "future_predictor_policy": _FUTURE_PREDICTOR_POLICY,
        "training_data_policy": _TRAINING_DATA_POLICY,
    }


def save_arima_model(
    model: FittedArima,
    model_path: Path,
    manifest_path: Path,
    *,
    execution_uuid: str,
) -> None:
    """Save the training snapshot and its model-only contract with artifact hash.

    Args:
        model: Fitted snapshot, unchanged by rolling evaluation.
        model_path: Destination joblib path.
        manifest_path: Destination JSON sidecar path.
        execution_uuid: Identifier linking to the execution's MLflow diagnostics.

    Raises:
        ValueError: If the execution identifier is empty.
    """
    if not execution_uuid:
        raise ValueError("execution_uuid must be nonempty")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    manifest = {
        **_model_contract(model),
        "execution_uuid": execution_uuid,
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_arima_model(
    model_path: Path, manifest_path: Path, *, contract: JoinedFeatureContract
) -> FittedArima:
    """Validate the current contract and artifact hash before deserializing.

    Args:
        model_path: Trusted local saved joblib artifact.
        manifest_path: Model-only JSON sidecar.
        contract: Current validated station and forecast target contract.

    Returns:
        Training state snapshot, ready for causal rolling forecasts.

    Raises:
        ValueError: If the manifest, current contract, or artifact disagree.
        TypeError: If the artifact is not an ARIMA state snapshot.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "5.0",
        "estimator": _ESTIMATOR,
        "station_id": contract.station_id,
        "input_columns": [f"{contract.station_id}__water_level"],
        "target_columns": list(contract.target_columns),
        "forecast_horizon_hours": len(contract.target_columns),
        "seasonal_order": [0, 0, 0, 0],
        "input_frequency": "h",
        "timezone": "UTC",
        "unit": "cm",
        "update_policy": _UPDATE_POLICY,
        "future_predictor_policy": _FUTURE_PREDICTOR_POLICY,
        "training_data_policy": _TRAINING_DATA_POLICY,
        "model_file": model_path.name,
    }
    if f"{contract.station_id}__water_level" not in contract.predictor_columns:
        raise ValueError("Current contract lacks the ARIMA input column")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"ARIMA manifest {key} does not match current contract; regenerate ARIMA artifacts"
            )
    try:
        candidate = ArimaCandidate(tuple(manifest["order"]), manifest["intercept"])
        if manifest["trend"] != ("c" if candidate.intercept else "n"):
            raise ValueError("ARIMA manifest trend does not match its intercept")
        if (
            not isinstance(manifest["execution_uuid"], str)
            or not manifest["execution_uuid"]
        ):
            raise ValueError("ARIMA manifest execution_uuid is empty")
        state_time = _utc_hour(pd.Timestamp(manifest["state_time"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("ARIMA manifest is incomplete or malformed") from exc
    if (
        manifest.get("model_sha256")
        != hashlib.sha256(model_path.read_bytes()).hexdigest()
    ):
        raise ValueError("ARIMA artifact hash does not match its manifest")
    model = load(model_path)
    if not isinstance(model, FittedArima):
        raise TypeError("ARIMA artifact is not a fitted state snapshot")
    for key, value in _model_contract(model).items():
        if manifest.get(key) != value:
            raise ValueError(f"ARIMA artifact {key} does not match its manifest")
    if model.state_time != state_time:
        raise ValueError("ARIMA artifact state time does not match its manifest")
    return model


@dataclass
class CandidateEvaluation:
    """Fold results retained for reporting and complete-fold selection.

    Attributes:
        candidate: Fixed configuration evaluated in every fold.
        fold_details: Fit boundaries, convergence, or failure details per fold.
        aggregate: Successful fold aggregate metrics.
        horizons: Successful fold per-horizon metrics.
        summary: Mean/std metrics, present only when every fold succeeds.
        started_at_ms: Wall-clock start of candidate evaluation, for MLflow.
        ended_at_ms: Wall-clock end of candidate evaluation, for MLflow.
        fold_times_ms: Wall-clock (start, end) of each fold, for MLflow.
    """

    candidate: ArimaCandidate
    fold_details: list[dict[str, Any]]
    aggregate: pd.DataFrame
    horizons: pd.DataFrame
    summary: dict[str, float]
    started_at_ms: int
    ended_at_ms: int
    fold_times_ms: list[tuple[int, int]]


def evaluate_candidate(
    candidate: ArimaCandidate,
    history: pd.Series,
    rows: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    contract: JoinedFeatureContract,
    maxiter: int = 200,
) -> CandidateEvaluation:
    """Fit and score one fixed configuration on all common validation folds.

    Each fold fits the observed levels through its final training issue time
    plus the forecast horizon: the latest level any training target uses.
    The embargo keeps that cutoff before the first validation issue time.

    Args:
        candidate: Configuration shared across folds.
        history: Unfiltered hourly training observations.
        rows: Common eligible training rows with timestamps and targets.
        folds: Shared training/validation positional indices.
        contract: Validated input/output contract.
        maxiter: Optimizer iteration budget.

    Returns:
        Successful metrics and all fold diagnostics, including failures.

    Raises:
        ValueError: If no folds are supplied.
    """
    if not folds:
        raise ValueError("ARIMA evaluation requires validation folds")
    horizon = len(contract.target_columns) * _HOUR
    started_at_ms = time_ns() // 1_000_000
    details: list[dict[str, Any]] = []
    fold_times_ms: list[tuple[int, int]] = []
    aggregates, horizons = [], []
    for number, (train, validation) in enumerate(folds, start=1):
        fold_started_at_ms = time_ns() // 1_000_000
        last_train_issue = rows.iloc[train[-1]]["timestamp"]
        fit_details: dict[str, Any] = {
            "fold": number,
            "order": str(candidate.order),
            "intercept": candidate.intercept,
            "maxiter": maxiter,
            "last_train_issue_time": last_train_issue.isoformat(),
            "fit_cutoff": (last_train_issue + horizon).isoformat(),
            "validation_start": rows.iloc[validation[0]]["timestamp"].isoformat(),
            "validation_end": rows.iloc[validation[-1]]["timestamp"].isoformat(),
            "scored_issue_times": len(validation),
        }
        try:
            model, diagnostics = fit_arima(
                history,
                cutoff=last_train_issue + horizon,
                contract=contract,
                candidate=candidate,
                maxiter=maxiter,
            )
            fit_details.update(diagnostics)
            predictions = rolling_forecasts(
                model, history, rows.iloc[validation]["timestamp"]
            )
            aggregate, horizon_metrics = metric_tables(
                rows.iloc[validation][list(contract.target_columns)],
                predictions,
                target_columns=list(contract.target_columns),
                station_id=contract.station_id,
            )
            aggregates.append(aggregate.assign(fold=number))
            horizons.append(horizon_metrics.assign(fold=number))
            fit_details["status"] = "ok"
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ) as exc:
            # A numerical failure disqualifies this configuration, not the search.
            fit_details.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        details.append(fit_details)
        fold_times_ms.append((fold_started_at_ms, time_ns() // 1_000_000))
    aggregate_table = (
        pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame()
    )
    horizon_table = (
        pd.concat(horizons, ignore_index=True) if horizons else pd.DataFrame()
    )
    summary = (
        summarize_cv_metrics(aggregate_table, horizon_table)
        if len(aggregates) == len(folds)
        else {}
    )
    return CandidateEvaluation(
        candidate,
        details,
        aggregate_table,
        horizon_table,
        summary,
        started_at_ms,
        time_ns() // 1_000_000,
        fold_times_ms,
    )


def evaluate_candidates(
    candidates: Sequence[ArimaCandidate],
    history: pd.Series,
    rows: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    contract: JoinedFeatureContract,
    n_workers: int = 4,
    maxiter: int = 200,
) -> Iterator[CandidateEvaluation]:
    """Evaluate candidates in separate processes, yielding in input order.

    Each worker evaluates all folds of one candidate sequentially. Numerical
    libraries use one thread per worker. Only metrics and diagnostics return
    to the caller, which owns MLflow logging and final model selection.
    With one worker, evaluation runs directly in the calling process.

    Args:
        candidates: Fixed configurations in stable reporting order.
        history: Unfiltered hourly training observations.
        rows: Common eligible training rows with timestamps and targets.
        folds: Shared training/validation positional indices.
        contract: Validated input/output contract.
        n_workers: Positive process count; one disables multiprocessing.
        maxiter: Optimizer iteration budget per fit.

    Yields:
        Candidate metrics and diagnostics, including failed folds.

    Raises:
        ValueError: If the worker count is not a positive integer.
    """
    if type(n_workers) is not int or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")
    # Avoid serializing the other models' predictor matrices to each worker.
    scoring_rows = rows[["timestamp", *contract.target_columns]]
    if n_workers == 1:
        for candidate in candidates:
            yield evaluate_candidate(
                candidate,
                history,
                scoring_rows,
                folds,
                contract=contract,
                maxiter=maxiter,
            )
        return
    # Loky supports notebook callers without forking a live notebook kernel.
    # Ordered delivery keeps candidate numbering independent of completion order.
    with (
        parallel_config(backend="loky", inner_max_num_threads=1),
        Parallel(
            n_jobs=n_workers,
            return_as="generator",
            batch_size=1,
            pre_dispatch=n_workers,
        ) as pool,
    ):
        yield from pool(
            delayed(evaluate_candidate)(
                candidate,
                history,
                scoring_rows,
                folds,
                contract=contract,
                maxiter=maxiter,
            )
            for candidate in candidates
        )


def candidate_table(evaluations: Sequence[CandidateEvaluation]) -> pd.DataFrame:
    """Build the selection table from successful complete-fold candidates.

    Args:
        evaluations: Candidate evaluations including failures.

    Returns:
        Successful candidates with shared selector metric names.
    """
    return pd.DataFrame(
        [
            {
                **result.candidate.parameters(),
                **{
                    key.removeprefix("cv_"): value
                    for key, value in result.summary.items()
                },
            }
            for result in evaluations
            if result.summary
            and result.fold_details
            and all(fold["status"] == "ok" for fold in result.fold_details)
        ]
    )


def select_candidate(
    cv_results: pd.DataFrame, metric: str = "rmse"
) -> tuple[tuple[int, int, int], bool]:
    """Select an order and intercept using metric and model complexity.

    Exact metric ties prefer lower p+q, lower d, no intercept, then ascending
    p and q.

    Args:
        cv_results: Complete-fold rows with mean metrics, p, d, q, intercept.
        metric: Aggregate selection metric, mae or rmse.

    Returns:
        Selected order and intercept flag.

    Raises:
        ValueError: If no candidate qualifies or ranking values are invalid.
    """
    if cv_results.empty:
        raise ValueError("No ARIMA candidate succeeded in every validation fold")
    frame = cv_results.assign(order_complexity=cv_results["p"] + cv_results["q"])
    if (
        metric in {"mae", "rmse"}
        and not np.isfinite(frame[f"{metric}_mean"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("ARIMA candidate ranking contains nonfinite values")
    winner = rank_candidate(
        frame,
        metric,
        tie_break_columns=["order_complexity", "d", "intercept", "p", "q"],
    )
    return (int(winner["p"]), int(winner["d"]), int(winner["q"])), bool(
        winner["intercept"]
    )


def log_candidate(evaluation: CandidateEvaluation) -> None:
    """Log fold children and complete-fold summaries under an active MLflow parent.

    Fold runs are back-dated to their worker-side evaluation times, because
    logging happens after evaluation has finished.

    Args:
        evaluation: Evaluated candidate with fit and scoring diagnostics.

    Raises:
        ValueError: If there is no active parent run.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    parent = mlflow.active_run()
    if parent is None:
        raise ValueError("Candidate logging requires an active parent run")
    mlflow.log_params(evaluation.candidate.parameters())
    mlflow.set_tag("status", "ok" if evaluation.summary else "failed")
    client = MlflowClient()
    for detail, (started_at_ms, ended_at_ms) in zip(
        evaluation.fold_details, evaluation.fold_times_ms, strict=True
    ):
        number = detail["fold"]
        child = client.create_run(
            experiment_id=parent.info.experiment_id,
            start_time=started_at_ms,
            run_name=f"arima_fold_{number:02d}",
            tags={
                "phase": "cv",
                "run_type": "fold",
                "fold": str(number),
                "execution_uuid": parent.data.tags.get("execution_uuid", ""),
                "mlflow.parentRunId": parent.info.run_id,
            },
        )
        with mlflow.start_run(run_id=child.info.run_id, nested=True):
            mlflow.log_params(detail)
            mlflow.set_tag("status", detail["status"])
            if detail["status"] == "ok":
                aggregate = evaluation.aggregate.loc[
                    evaluation.aggregate["fold"].eq(number)
                ].iloc[0]
                horizons = evaluation.horizons.loc[
                    evaluation.horizons["fold"].eq(number)
                ]
                mlflow.log_metrics(
                    {
                        **{
                            f"fold_{metric}": float(aggregate[metric])
                            for metric in ("mae", "rmse", "me", "r2")
                        },
                        **{
                            f"fold_{metric}_horizon_{int(str(row.horizon_hours)):02d}": float(
                                getattr(row, metric)
                            )
                            for row in horizons.itertuples()
                            for metric in ("mae", "rmse", "me", "r2")
                        },
                    }
                )
        client.set_terminated(child.info.run_id, end_time=ended_at_ms)
    mlflow.log_metrics(evaluation.summary)
