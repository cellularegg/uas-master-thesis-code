"""Recursive hourly forecasts from regressions with ARIMA errors."""

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from src.arima import observed_hourly_history
from src.dataset import JoinedDataset, JoinedFeatureContract
from src.feature_engineering import (
    target_feature_lookback_hours,
    target_level_features,
    utc_calendar_features,
)
from src.metrics import metric_tables
from src.model_selection import select_candidate as rank_candidate
from src.training import numeric_predictors, summarize_cv_metrics, validate_predictions

_HOUR = pd.Timedelta(hours=1)
_UPDATE_POLICY = (
    "fixed_parameters; observed_one_step_labels; hourly_missing_transitions"
)
_TRAINING_DATA_POLICY = (
    "common_eligible_cohort_issue_times; "
    "one_step_labels_through_last_train_issue_plus_1h; imputed_as_missing"
)
_ESTIMATION = "iterated_feasible_gls_ml"
_FORMULATION = "recursive_one_step_regression_with_arima_errors"
_FUTURE_PREDICTOR_POLICY = (
    "recompute_target_level_features_from_predictions; "
    "hold_issue_other_stations_and_weather; recompute_utc_calendar"
)


@dataclass(frozen=True)
class ArimaxCandidate:
    """One subset and nonseasonal order.

    Attributes:
        subset: Shared ablation subset name.
        feature_columns: Ordered predictors, including redundant columns.
        order: Nonnegative p, d, q orders.
    """

    subset: str
    feature_columns: tuple[str, ...]
    order: tuple[int, int, int]

    def __post_init__(self) -> None:
        """Reject ambiguous subsets and malformed orders before fitting."""
        if not self.subset or not self.feature_columns:
            raise ValueError("ARIMAX requires a named nonempty feature subset")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("ARIMAX feature columns must be unique")
        if len(self.order) != 3 or any(type(v) is not int or v < 0 for v in self.order):
            raise ValueError("order must contain three nonnegative integers")

    def parameters(self) -> dict[str, Any]:
        """Return candidate identity and input columns for reporting.

        Returns:
            Parameters shared by candidate-parent and sealed-test runs.
        """
        p, d, q = self.order
        return {
            "subset": self.subset,
            "p": p,
            "d": d,
            "q": q,
            "feature_count": len(self.feature_columns),
            "feature_columns": json.dumps(self.feature_columns),
        }


def candidate_grid(subsets: Mapping[str, Sequence[str]]) -> list[ArimaxCandidate]:
    """Build thirteen shared orders for each existing feature subset.

    Args:
        subsets: Contract-ordered columns for each ablation subset.

    Returns:
        Stable subset-major grid with p and q up to two when d is zero,
        plus p and q up to one when d is one.
    """
    orders = [
        (p, d, q)
        for d in range(2)
        for p, q in product(range(3 if d == 0 else 2), repeat=2)
    ]
    return [
        ArimaxCandidate(name, tuple(columns), order)
        for name, columns in subsets.items()
        for order in orders
    ]


@dataclass(frozen=True)
class FittedArimax:
    """Training-only state snapshot and its exact preprocessing contract.

    Attributes:
        contract: Full joined feature contract used to validate availability.
        candidate: Selected subset and order.
        result: One ARIMA error-model training result.
        coefficients: Regression coefficients on the scaled
            predictors, led by an intercept when d is zero.
        predictor_scaler: Training-only standardization retaining every column.
        target_scaler: Training-only one-step target standardization.
        state_time: Last fitted issue index s; the fitted state includes y(s+1).
    """

    contract: JoinedFeatureContract
    candidate: ArimaxCandidate
    result: Any
    coefficients: np.ndarray
    predictor_scaler: Any
    target_scaler: Any
    state_time: pd.Timestamp


def _hourly_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.empty:
        raise ValueError("ARIMAX requires nonempty timezone-aware timestamps")
    index = index.tz_convert("UTC")
    if (
        index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.floor("h"))
    ):
        raise ValueError("ARIMAX timestamps must be unique, increasing hourly times")
    return index


def _candidate_contract(
    candidate: ArimaxCandidate, contract: JoinedFeatureContract
) -> None:
    selected = set(candidate.feature_columns)
    if (
        tuple(c for c in contract.predictor_columns if c in selected)
        != candidate.feature_columns
    ):
        raise ValueError("ARIMAX subset must follow the current predictor contract")
    if not contract.target_columns:
        raise ValueError("ARIMAX requires forecast targets")


def _has_intercept(order: tuple[int, int, int]) -> bool:
    # With d=1 a constant is absorbed by the diffuse integrated level.
    return order[1] == 0


def _design(x: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) if _has_intercept(order) else x


def _error_model(endog: np.ndarray, order: tuple[int, int, int]) -> Any:
    return SARIMAX(
        endog,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        trend="n",
        concentrate_scale=True,
        simple_differencing=False,
    )


def _time_invariant_system(model: Any) -> tuple[np.ndarray, np.ndarray]:
    transition = np.asarray(model.ssm["transition"])
    design = np.asarray(model.ssm["design"])
    if transition.ndim != 2 or design.ndim != 2:
        raise ValueError("ARIMAX error models must be time invariant")
    return transition, design[0]


def _whitened(model: Any, params: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Standardized Kalman innovations of every column under fixed ARMA params.

    Gains and innovation variances depend only on the parameters and the
    missing pattern of the model's endogenous series, so one statsmodels pass
    supplies them and all columns then share one linear recursion.
    """
    filtered = model.filter(params).filter_results
    transition, design = _time_invariant_system(model)
    observed = np.isfinite(model.endog[:, 0])
    gain = filtered.kalman_gain[:, 0, :]
    values = np.where(observed[:, None], series, 0.0)
    state = np.zeros((transition.shape[0], values.shape[1]))
    innovations = np.empty_like(values)
    for time in range(len(values)):
        innovations[time] = values[time] - design @ state
        state = transition @ state + np.outer(gain[:, time], innovations[time])
    weight = np.where(observed, 1 / np.sqrt(filtered.forecasts_error_cov[0, 0]), 0.0)
    weight[: filtered.nobs_diffuse] = 0.0
    return innovations * weight[:, None]


def _gls_coefficients(
    model: Any, params: np.ndarray, target: np.ndarray, design: np.ndarray
) -> np.ndarray:
    whitened = _whitened(model, params, np.column_stack([target, design]))
    # Minimum-norm least squares keeps every constant or redundant column.
    return np.linalg.lstsq(whitened[:, 1:], whitened[:, 0], rcond=None)[0]


def _fit_one_step(
    target: np.ndarray,
    design: np.ndarray,
    order: tuple[int, int, int],
    *,
    maxiter: int,
    max_outer_iter: int,
    tol: float,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Alternate exact GLS coefficients and ARMA MLE until the likelihood settles.

    Each step optimizes one parameter block conditional on the other. This is
    an iterative fit with a finite stopping budget; it does not guarantee a
    global joint maximum or convergence when the budget is exhausted.
    """
    model = _error_model(target, order)
    params = np.zeros(model.k_params)
    previous = -np.inf
    for iteration in range(1, max_outer_iter + 1):
        coefficients = _gls_coefficients(model, params, target, design)
        residual_model = _error_model(target - design @ coefficients, order)
        result = (
            residual_model.fit(start_params=params, maxiter=maxiter, disp=False)
            if model.k_params
            else residual_model.filter(params)
        )
        params = np.asarray(result.params, dtype=float)
        loglike = float(result.llf)
        if not (
            np.isfinite(coefficients).all()
            and np.isfinite(params).all()
            and np.isfinite(loglike)
        ):
            return coefficients, result, {}
        outer_converged = not model.k_params or abs(loglike - previous) <= tol * max(
            1.0, abs(loglike)
        )
        previous = loglike
        if outer_converged:
            break
    retvals = getattr(result, "mle_retvals", None) or {}
    return (
        coefficients,
        result,
        {
            "converged": bool(retvals.get("converged", True)),
            "iterations": retvals.get("iterations", 0),
            "outer_iterations": iteration,
            "outer_converged": bool(outer_converged),
            "loglike": loglike,
        },
    )


def fit_arimax(
    rows: pd.DataFrame,
    *,
    contract: JoinedFeatureContract,
    candidate: ArimaxCandidate,
    maxiter: int = 200,
    max_outer_iter: int = 25,
    tol: float = 1e-7,
) -> tuple[FittedArimax, dict[str, Any]]:
    """Fit the selected recursive one-step formulation on eligible rows.

    The one-step target is a regression with ARIMA errors estimated by iterated
    feasible GLS: exact Kalman-whitened GLS coefficients given the ARMA
    parameters alternate with ARMA maximum likelihood given the coefficients.
    The finite iteration budget does not guarantee convergence or a global
    optimum. Missing issue hours are missing observations on the hourly grid.
    No columns are dropped.

    Args:
        rows: Common eligible training cohort, including timestamps and targets.
        contract: Validated joined input/output contract.
        candidate: Subset and ARIMA order.
        maxiter: Optimizer iteration budget per ARMA maximum-likelihood step.
        max_outer_iter: Maximum GLS/ARMA alternations.
        tol: Relative log-likelihood change that ends the alternation.

    Returns:
        Immutable training snapshot and numerical diagnostics.

    Raises:
        ValueError: If inputs are invalid or a fit has nonfinite parameters.
    """
    _candidate_contract(candidate, contract)
    if type(maxiter) is not int or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if type(max_outer_iter) is not int or max_outer_iter < 1:
        raise ValueError("max_outer_iter must be a positive integer")
    if not tol > 0:
        raise ValueError("tol must be positive")
    index = _hourly_index(pd.DatetimeIndex(rows["timestamp"]))
    if len(index) < 3:
        raise ValueError("ARIMAX requires at least three training rows")
    full_x = numeric_predictors(rows, contract.predictor_columns)
    targets = rows[list(contract.target_columns)].to_numpy(dtype=float)
    if not np.isfinite(full_x.to_numpy()).all() or not np.isfinite(targets).all():
        raise ValueError("Training rows must belong to the full eligible cohort")
    x_scaler = StandardScaler().fit(full_x[list(candidate.feature_columns)].to_numpy())
    y_scaler = StandardScaler().fit(targets[:, :1])
    x = x_scaler.transform(full_x[list(candidate.feature_columns)].to_numpy())
    y_scaled = y_scaler.transform(targets[:, :1])[:, 0]
    grid = pd.date_range(index[0], index[-1], freq="h")
    design = _design(
        pd.DataFrame(x, index=index).reindex(grid).fillna(0.0).to_numpy(),
        candidate.order,
    )
    target = pd.Series(y_scaled, index=index).reindex(grid).to_numpy()
    rank = int(np.linalg.matrix_rank(x))
    beta, result, fit_detail = _fit_one_step(
        target,
        design,
        candidate.order,
        maxiter=maxiter,
        max_outer_iter=max_outer_iter,
        tol=tol,
    )
    if not fit_detail:
        raise ValueError("ARIMAX one-step fit has nonfinite parameters")
    detail = {
        "target": contract.target_columns[0],
        **fit_detail,
        "predictor_rank": rank,
        "feature_count": x.shape[1],
        "constant_features": int(np.count_nonzero(x_scaler.var_ == 0)),
        "fit_rows": len(rows),
        "fit_grid_hours": len(grid),
        "fit_start": index[0].isoformat(),
        "state_time": index[-1].isoformat(),
        "last_label_time": (index[-1] + _HOUR).isoformat(),
    }
    return FittedArimax(
        contract, candidate, result, beta, x_scaler, y_scaler, index[-1]
    ), detail


def sealed_test_histories(
    dataset: JoinedDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the train and test histories for sealed-test rolling forecasts.

    Args:
        dataset: Joined dataset with separate train and test histories.

    Returns:
        The hourly predictor history and the hourly target history spanning
        train and test; target hours inserted by the hourly reindex are marked
        as not imputed so they remain explicit missing observations.

    Raises:
        ValueError: If the train history does not end before the test history.
    """
    if (
        dataset.predictor_train_history.index[-1]
        >= dataset.predictor_test_history.index[0]
    ):
        raise ValueError("ARIMAX requires strictly separated train/test histories")
    predictor_history = pd.concat(
        [dataset.predictor_train_history, dataset.predictor_test_history]
    ).asfreq("h")
    target_history = pd.concat(
        [dataset.target_train_history, dataset.target_test_history]
    ).asfreq("h")
    # Newly inserted target-history hours must remain explicit missing observations.
    target_history["imputed"] = target_history["imputed"].fillna(False).astype(bool)
    return predictor_history, target_history


def rolling_forecasts(
    model: FittedArimax,
    predictor_history: pd.DataFrame,
    target_history: pd.DataFrame,
    issue_times: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    """Forecast recursive ARIMAX for every requested origin.

    Observed levels through each origin update the residual state after the
    corresponding label becomes available. Missing or imputed levels, or
    incomplete predictor rows, advance time without a measurement update.

    Step ``k`` forecasts ``y(t+k)`` from the input hour ``t+k-1``. The first
    step uses the issue-time predictors. Later steps recompute the target
    station's water-level and imputation predictors with the Stage-3 formulas
    from the Stage-2 levels through ``t`` followed by the forecasts for
    ``t+1`` through ``t+k-1`` (counted as not imputed), recompute the UTC
    calendar, and hold every other predictor at its issue-time value.

    Args:
        model: Training-only snapshot, never mutated by this function.
        predictor_history: Unfiltered predictors on a UTC issue-time index.
        target_history: Unfiltered water_level/imputed history on a UTC index.
        issue_times: Increasing forecast origins from the scoring cohort.

    Returns:
        Finite matrix ordered by issue time and contractual horizon, in cm.

    Raises:
        ValueError: If history is incomplete, an origin lacks predictors, or
            a recomputed predictor lacks its lookback.
    """
    issues = _hourly_index(pd.DatetimeIndex(list(issue_times)))
    history = predictor_history.copy()
    history.index = _hourly_index(history.index)
    first_label_time = model.state_time + _HOUR
    if issues[0] < first_label_time:
        raise ValueError("Fitted labels must be observable before forecasting")
    if history.index[0] > model.state_time or history.index[-1] < issues[-1]:
        raise ValueError("Predictor history must cover the state through all origins")
    observed = observed_hourly_history(target_history)
    if observed.index[0] > first_label_time or observed.index[-1] < issues[-1]:
        raise ValueError(
            "Target history must cover all observed labels through origins"
        )
    grid = pd.date_range(model.state_time, issues[-1], freq="h")
    full_x = numeric_predictors(history.reindex(grid), model.contract.predictor_columns)
    complete = np.isfinite(full_x.to_numpy()).all(axis=1)
    issue_positions = grid.get_indexer(issues)
    if not complete[issue_positions].all():
        raise ValueError("Forecast origins require all contractual predictors")
    selected_x = full_x[list(model.candidate.feature_columns)].to_numpy()
    selected_x[~complete] = model.predictor_scaler.mean_
    design = _design(
        model.predictor_scaler.transform(selected_x), model.candidate.order
    )
    regression = design @ model.coefficients
    # Position zero is the fitted state after the final training label at s+1.
    # A residual at input hour u becomes observable at u+1.
    update_inputs = grid[1 : issue_positions[-1]]
    labels = observed.reindex(update_inputs + _HOUR).to_numpy(dtype=float)
    residuals = (labels - model.target_scaler.mean_[0]) / model.target_scaler.scale_[
        0
    ] - regression[1 : len(update_inputs) + 1]
    residuals[~complete[1 : len(update_inputs) + 1]] = np.nan
    result = model.result
    if len(residuals):
        result = result.append(residuals, refit=False)
    states = result.filter_results.filtered_state[:, model.result.nobs - 1 :]
    transition, loading = _time_invariant_system(model.result.model)
    width = len(model.contract.target_columns)

    columns = model.candidate.feature_columns
    prefix = f"{model.contract.station_id}__"
    level_positions = _base_positions(
        columns, target_level_features(pd.DataFrame(), pd.DataFrame()), prefix
    )
    calendar_positions = _base_positions(
        columns,
        utc_calendar_features(pd.Series(issues[:1])),
        None,
    )
    level_window, imputed_window = _issue_windows(target_history, issues, width)
    lookback = target_feature_lookback_hours()
    window_ends = np.arange(1, len(issues) + 1) * (lookback + 1) - 1
    future_x = selected_x[issue_positions].copy()
    origin_states = states[:, issue_positions - 1]
    power = np.eye(transition.shape[0])
    forecasts = np.empty((len(issues), width), dtype=float)
    for step in range(width):
        if step:
            # Stack the origin windows into one column: every window spans the
            # full lookback, so each window's final row reads only its own rows.
            window = level_window[step : step + lookback + 1].ravel(order="F")
            flags = imputed_window[step : step + lookback + 1].ravel(order="F")
            recomputed = target_level_features(
                pd.DataFrame({"level": window}), pd.DataFrame({"level": flags})
            )
            for base_name, positions in level_positions.items():
                values = recomputed[base_name]["level"].to_numpy(dtype=float)[
                    window_ends
                ]
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Recursive predictor {prefix}{base_name} lacks its lookback"
                    )
                future_x[:, positions] = values[:, None]
            calendar = utc_calendar_features(pd.Series(issues + step * _HOUR))
            for base_name, positions in calendar_positions.items():
                future_x[:, positions] = np.asarray(calendar[base_name])[:, None]
        future_design = _design(
            model.predictor_scaler.transform(future_x), model.candidate.order
        )
        power = transition @ power
        scaled = future_design @ model.coefficients + loading @ power @ origin_states
        forecasts[:, step] = (
            scaled * model.target_scaler.scale_[0] + model.target_scaler.mean_[0]
        )
        if step + 1 < width:
            level_window[lookback + step + 1] = forecasts[:, step]
    return validate_predictions(
        forecasts,
        expected_rows=len(issues),
        target_columns=model.contract.target_columns,
        artifact_name="recursive ARIMAX",
    )


def _base_positions(
    columns: Sequence[str], features: Mapping[str, Any], prefix: str | None
) -> dict[str, list[int]]:
    """Map recomputed base names to their positions among selected columns.

    A ``None`` prefix matches the base name for every station.
    """
    positions: dict[str, list[int]] = {}
    for position, column in enumerate(columns):
        station, separator, base_name = column.partition("__")
        if (
            separator
            and base_name in features
            and (prefix is None or f"{station}__" == prefix)
        ):
            positions.setdefault(base_name, []).append(position)
    return positions


def _issue_windows(
    target_history: pd.DataFrame, issues: pd.DatetimeIndex, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stage-2 target levels and flags from each origin's lookback onward.

    Rows run from ``t - lookback`` through ``t + horizon - 1`` with one column
    per origin. Rows after ``t`` are placeholders for recursive forecasts,
    flagged as not imputed. Hours outside the history are missing.
    """
    lookback = target_feature_lookback_hours()
    frame = target_history[["water_level", "imputed"]].copy()
    frame.index = _hourly_index(pd.DatetimeIndex(frame.index))
    frame = frame.reindex(
        pd.date_range(issues[0] - lookback * _HOUR, issues[-1], freq="h")
    )
    levels = pd.to_numeric(frame["water_level"], errors="raise").to_numpy(float)
    flags = frame["imputed"].fillna(False).to_numpy(dtype=bool)
    offsets = np.arange(-lookback, 1)
    rows = (issues - frame.index[0]) // _HOUR
    rows = np.asarray(rows)[None, :] + offsets[:, None]
    future = (horizon - 1, len(issues))
    level_window = np.vstack([levels[rows], np.full(future, np.nan)])
    imputed_window = np.vstack([flags[rows], np.zeros(future, dtype=bool)])
    return level_window, imputed_window


@dataclass
class CandidateEvaluation:
    """Complete-fold metrics and numerical diagnostics for one candidate.

    Attributes:
        candidate: Evaluated feature subset and order.
        fold_details: Boundaries, one-step fit diagnostics, and failures.
        aggregate: Aggregate metrics from successful folds.
        horizons: Per-horizon metrics from successful folds.
        summary: Mean/std metrics only when all folds succeed.
        started_at_ms: Wall-clock start of candidate evaluation, for MLflow.
        ended_at_ms: Wall-clock end of candidate evaluation, for MLflow.
    """

    candidate: ArimaxCandidate
    fold_details: list[dict[str, Any]]
    aggregate: pd.DataFrame
    horizons: pd.DataFrame
    summary: dict[str, float]
    started_at_ms: int
    ended_at_ms: int


def evaluate_candidate(
    candidate: ArimaxCandidate,
    predictor_history: pd.DataFrame,
    target_history: pd.DataFrame,
    rows: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    contract: JoinedFeatureContract,
    maxiter: int = 200,
    max_outer_iter: int = 25,
    tol: float = 1e-7,
) -> CandidateEvaluation:
    """Evaluate every fold, disqualifying any failed or nonfinite candidate.

    Args:
        candidate: Shared subset and order.
        predictor_history: Unfiltered training predictors.
        target_history: Unfiltered training observations and imputation flags.
        rows: Common eligible training cohort.
        folds: Shared positional training/validation indices.
        contract: Validated feature contract.
        maxiter: Optimizer budget per ARMA maximum-likelihood step.
        max_outer_iter: Maximum GLS/ARMA alternations.
        tol: Relative log-likelihood change that ends the alternation.

    Returns:
        Fold diagnostics and complete-fold metrics; finite nonconverged fits
        remain eligible.

    Raises:
        ValueError: If no validation folds are supplied.
    """
    if not folds:
        raise ValueError("ARIMAX requires validation folds")
    started_at_ms = time_ns() // 1_000_000
    details: list[dict[str, Any]] = []
    aggregates, horizons = [], []
    for number, (train, validation) in enumerate(folds, start=1):
        fold_started_at_ms = time_ns() // 1_000_000
        train_rows, validation_rows = rows.iloc[train], rows.iloc[validation]
        detail: dict[str, Any] = {
            "fold": number,
            "fit_rows": len(train),
            "fit_start": train_rows["timestamp"].iloc[0].isoformat(),
            "fit_end": train_rows["timestamp"].iloc[-1].isoformat(),
            "validation_start": validation_rows["timestamp"].iloc[0].isoformat(),
            "validation_end": validation_rows["timestamp"].iloc[-1].isoformat(),
            "scored_issue_times": len(validation),
        }
        try:
            model, fit_details = fit_arimax(
                train_rows,
                contract=contract,
                candidate=candidate,
                maxiter=maxiter,
                max_outer_iter=max_outer_iter,
                tol=tol,
            )
            detail["fit_diagnostics"] = fit_details
            predictions = rolling_forecasts(
                model, predictor_history, target_history, validation_rows["timestamp"]
            )
            aggregate, horizon = metric_tables(
                validation_rows[list(contract.target_columns)],
                predictions,
                target_columns=list(contract.target_columns),
                station_id=contract.station_id,
            )
            if (
                not np.isfinite(aggregate[["mae", "rmse", "me", "r2"]].to_numpy()).all()
                or not np.isfinite(
                    horizon[["mae", "rmse", "me", "r2"]].to_numpy()
                ).all()
            ):
                raise ValueError("ARIMAX fold metrics are nonfinite")
            aggregates.append(aggregate.assign(fold=number))
            horizons.append(horizon.assign(fold=number))
            detail["status"] = "ok"
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ) as exc:
            detail.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        detail["started_at_ms"] = fold_started_at_ms
        detail["ended_at_ms"] = time_ns() // 1_000_000
        details.append(detail)
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
    )


def evaluate_candidates(
    candidates: Sequence[ArimaxCandidate],
    predictor_history: pd.DataFrame,
    target_history: pd.DataFrame,
    rows: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    contract: JoinedFeatureContract,
    n_workers: int = 4,
    maxiter: int = 200,
    max_outer_iter: int = 25,
    tol: float = 1e-7,
) -> Iterator[CandidateEvaluation]:
    """Yield candidate results as they finish, with one BLAS thread per worker.

    Sequential search yields in grid order; parallel search yields in
    completion order so that one slow candidate never delays the others.

    Args:
        candidates: Fixed candidate grid.
        predictor_history: Unfiltered training predictors.
        target_history: Unfiltered training target history.
        rows: Eligible training cohort.
        folds: Shared validation folds.
        contract: Validated feature contract.
        n_workers: Candidate process count; one runs locally.
        maxiter: Optimizer iteration limit per ARMA maximum-likelihood step.
        max_outer_iter: Maximum GLS/ARMA alternations per horizon.
        tol: Relative log-likelihood change that ends the alternation.

    Yields:
        Complete candidate results for parent-process MLflow logging.

    Raises:
        ValueError: If the worker count is invalid.
    """
    if type(n_workers) is not int or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")
    if n_workers == 1:
        with threadpool_limits(limits=1):
            for candidate in candidates:
                yield evaluate_candidate(
                    candidate,
                    predictor_history,
                    target_history,
                    rows,
                    folds,
                    contract=contract,
                    maxiter=maxiter,
                    max_outer_iter=max_outer_iter,
                    tol=tol,
                )
        return
    with (
        parallel_config(backend="loky", inner_max_num_threads=1),
        Parallel(
            n_jobs=n_workers,
            return_as="generator_unordered",
            batch_size=1,
            pre_dispatch=n_workers,
        ) as pool,
    ):
        yield from pool(
            delayed(evaluate_candidate)(
                candidate,
                predictor_history,
                target_history,
                rows,
                folds,
                contract=contract,
                maxiter=maxiter,
                max_outer_iter=max_outer_iter,
                tol=tol,
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
) -> tuple[str, int, int, int]:
    """Select a subset/order using metric, feature count, and order complexity.

    Args:
        cv_results: Complete-fold rows with mean metrics, subset, p, d, q.
        metric: Aggregate selection metric, mae or rmse.

    Returns:
        Selected subset, p, d, q, also used by cross-model evaluation.

    Raises:
        ValueError: If no candidate qualifies or ranking values are invalid.
    """
    if cv_results.empty:
        raise ValueError("No ARIMAX candidate succeeded in every validation fold")
    frame = cv_results.assign(order_complexity=cv_results["p"] + cv_results["q"])
    if (
        metric in {"mae", "rmse"}
        and not np.isfinite(frame[f"{metric}_mean"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("ARIMAX candidate ranking contains nonfinite values")
    winner = rank_candidate(
        frame,
        metric,
        tie_break_columns=[
            "feature_count",
            "order_complexity",
            "d",
            "subset",
            "p",
            "q",
        ],
    )
    return (
        str(winner["subset"]),
        int(winner["p"]),
        int(winner["d"]),
        int(winner["q"]),
    )


def log_candidate(evaluation: CandidateEvaluation) -> None:
    """Log candidate metrics and nested folds under the active MLflow parent.

    Args:
        evaluation: Successful or failed candidate diagnostics.

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
    for detail in evaluation.fold_details:
        number = detail["fold"]
        child = client.create_run(
            experiment_id=parent.info.experiment_id,
            start_time=detail["started_at_ms"],
            run_name=f"arimax_fold_{number:02d}",
            tags={
                "phase": "cv",
                "run_type": "fold",
                "execution_uuid": parent.data.tags.get("execution_uuid", ""),
                "mlflow.parentRunId": parent.info.run_id,
            },
        )
        with mlflow.start_run(run_id=child.info.run_id, nested=True):
            mlflow.log_params(
                {
                    key: value
                    for key, value in detail.items()
                    if key not in {"fit_diagnostics", "started_at_ms", "ended_at_ms"}
                }
            )
            mlflow.set_tag("status", detail["status"])
            if "fit_diagnostics" in detail:
                mlflow.log_dict(detail["fit_diagnostics"], "fit_diagnostics.json")
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
        client.set_terminated(child.info.run_id, end_time=detail["ended_at_ms"])
    mlflow.log_metrics(evaluation.summary)


def _model_contract(model: FittedArimax) -> dict[str, Any]:
    return {
        "schema_version": "6.0",
        "estimator": "statsmodels.tsa.statespace.sarimax.SARIMAX",
        "formulation": _FORMULATION,
        "estimation": _ESTIMATION,
        "station_id": model.contract.station_id,
        "target_valid_column": model.contract.target_valid_column,
        "contract_predictor_columns": list(model.contract.predictor_columns),
        "input_columns": list(model.candidate.feature_columns),
        "target_columns": list(model.contract.target_columns),
        "forecast_horizon_hours": len(model.contract.target_columns),
        "subset": model.candidate.subset,
        "order": list(model.candidate.order),
        "seasonal_order": [0, 0, 0, 0],
        "trend": "n",
        "intercept": _has_intercept(model.candidate.order),
        "preprocessor": "sklearn.preprocessing.StandardScaler",
        "state_time": model.state_time.isoformat(),
        "input_frequency": "h",
        "timezone": "UTC",
        "unit": "cm",
        "update_policy": _UPDATE_POLICY,
        "future_predictor_policy": _FUTURE_PREDICTOR_POLICY,
        "training_data_policy": _TRAINING_DATA_POLICY,
        "fitted_order": list(model.result.model.order),
        "coefficient_count": len(model.coefficients),
        "predictor_mean": model.predictor_scaler.mean_.tolist(),
        "predictor_scale": model.predictor_scaler.scale_.tolist(),
        "target_mean": model.target_scaler.mean_.tolist(),
        "target_scale": model.target_scaler.scale_.tolist(),
    }


def save_arimax_model(
    model: FittedArimax,
    model_path: Path,
    metadata_path: Path,
    *,
    execution_uuid: str,
) -> None:
    """Save the training snapshot and an inference-only, hash-bound manifest.

    Args:
        model: Training snapshots untouched by sealed-test evaluation.
        model_path: Destination joblib artifact.
        metadata_path: Destination JSON contract.
        execution_uuid: Execution identifier shared with MLflow.

    Raises:
        ValueError: If the execution identifier is empty.
    """
    if not execution_uuid:
        raise ValueError("execution_uuid must be nonempty")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    manifest = {
        **_model_contract(model),
        "execution_uuid": execution_uuid,
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_arimax_model(
    model_path: Path, metadata_path: Path, *, contract: JoinedFeatureContract
) -> FittedArimax:
    """Validate the inference contract and artifact hash before deserialization.

    Args:
        model_path: Saved joblib snapshot.
        metadata_path: Saved manifest.
        contract: Current joined feature contract.

    Returns:
        Validated training-only model ready for independent causal forecasting.

    Raises:
        ValueError: If the manifest, artifact hash, or model contract mismatches.
    """
    manifest = json.loads(metadata_path.read_text())
    expected = {
        "schema_version": "6.0",
        "estimator": "statsmodels.tsa.statespace.sarimax.SARIMAX",
        "formulation": _FORMULATION,
        "estimation": _ESTIMATION,
        "station_id": contract.station_id,
        "target_valid_column": contract.target_valid_column,
        "contract_predictor_columns": list(contract.predictor_columns),
        "target_columns": list(contract.target_columns),
        "forecast_horizon_hours": len(contract.target_columns),
        "seasonal_order": [0, 0, 0, 0],
        "trend": "n",
        "preprocessor": "sklearn.preprocessing.StandardScaler",
        "input_frequency": "h",
        "timezone": "UTC",
        "unit": "cm",
        "update_policy": _UPDATE_POLICY,
        "future_predictor_policy": _FUTURE_PREDICTOR_POLICY,
        "training_data_policy": _TRAINING_DATA_POLICY,
        "model_file": model_path.name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"ARIMAX manifest {key} does not match current contract")
    try:
        candidate = ArimaxCandidate(
            manifest["subset"],
            tuple(manifest["input_columns"]),
            tuple(manifest["order"]),
        )
        _candidate_contract(candidate, contract)
        _hourly_index(pd.DatetimeIndex([pd.Timestamp(manifest["state_time"])]))
        intercept = _has_intercept(candidate.order)
        if manifest["intercept"] is not intercept:
            raise ValueError("ARIMAX manifest regression intercept mismatch")
        if (
            not isinstance(manifest["execution_uuid"], str)
            or not manifest["execution_uuid"]
        ):
            raise ValueError("ARIMAX manifest execution_uuid is empty")
        if manifest["fitted_order"] != list(candidate.order):
            raise ValueError("ARIMAX manifest fitted order mismatch")
        if manifest["coefficient_count"] != len(candidate.feature_columns) + intercept:
            raise ValueError("ARIMAX manifest regression coefficients mismatch")
        for prefix, count in (
            ("predictor", len(candidate.feature_columns)),
            ("target", 1),
        ):
            for suffix in ("mean", "scale"):
                values = np.asarray(manifest[f"{prefix}_{suffix}"], dtype=float)
                if (
                    values.shape != (count,)
                    or not np.isfinite(values).all()
                    or (suffix == "scale" and (values <= 0).any())
                ):
                    raise ValueError("ARIMAX manifest preprocessing is invalid")
    except (KeyError, TypeError) as exc:
        raise ValueError("ARIMAX manifest is incomplete or malformed") from exc
    if (
        manifest.get("model_sha256")
        != hashlib.sha256(model_path.read_bytes()).hexdigest()
    ):
        raise ValueError("ARIMAX artifact hash does not match its manifest")
    model = load(model_path)
    if not isinstance(model, FittedArimax):
        raise TypeError("ARIMAX artifact is not a fitted snapshot")
    for key, value in _model_contract(model).items():
        if manifest.get(key) != value:
            raise ValueError(f"ARIMAX artifact {key} does not match its manifest")
    return model


def run_params(
    dataset: JoinedDataset,
    *,
    station_id: str,
    forecast_horizon_hours: int,
    candidate_count: int,
    n_workers: int,
    maxiter: int,
    max_outer_iter: int,
    gls_tol: float,
    selection_metric: str,
    initial_train_fraction: float,
    embargo_rows: int,
) -> dict[str, object]:
    """Describe the ARIMAX search configuration and data for every MLflow run.

    Args:
        dataset: Joined dataset the search runs on.
        station_id: Target station identifier.
        forecast_horizon_hours: Configured direct-forecast horizon.
        candidate_count: Number of searched subset/order candidates.
        n_workers: Worker processes used for candidate evaluation.
        maxiter: Maximum optimizer iterations per ARMA fit.
        max_outer_iter: Maximum feasible-GLS outer iterations.
        gls_tol: Feasible-GLS convergence tolerance.
        selection_metric: CV metric used to select the candidate.
        initial_train_fraction: Fraction of rows in the first fold's training window.
        embargo_rows: Rows left between training and validation windows.

    Returns:
        Input hashes, search settings, row counts, and the model's formulation,
        estimation, and data policies.
    """
    return {
        **dataset.input_hashes,
        "station_id": station_id,
        "forecast_horizon_hours": forecast_horizon_hours,
        "candidate_count": candidate_count,
        "n_workers": n_workers,
        "maxiter": maxiter,
        "max_outer_iter": max_outer_iter,
        "gls_tol": gls_tol,
        "selection_metric": selection_metric,
        "n_validation_folds": len(dataset.folds),
        "initial_train_fraction": initial_train_fraction,
        "embargo_rows": embargo_rows,
        "eligible_train_rows": len(dataset.train_rows),
        "eligible_test_rows": len(dataset.test_rows),
        "raw_train_rows": dataset.raw_row_counts["train"],
        "raw_test_rows": dataset.raw_row_counts["test"],
        "formulation": "recursive one-step regression with ARIMA errors",
        "estimation": "iterated feasible GLS with conditional ARMA maximum likelihood",
        "intercept": "d == 0",
        "seasonal_order": "(0, 0, 0, 0)",
        "update_policy": "causal observed one-step label updates",
        "future_predictor_policy": "target-station level predictors recomputed from forecasts; UTC calendar recomputed; other predictors held at issue time",
        "training_data_policy": "common eligible cohort; one-step labels through last training issue time + 1 h; imputed as missing",
        "preprocessing": "training-only predictor and one-step target standardization; retain all columns",
        "nonconvergence_policy": "finite fits remain eligible",
    }


def log_cv_search(
    evaluations: Iterable[CandidateEvaluation],
    *,
    candidates: Sequence[ArimaxCandidate],
    client: Any,
    experiment_id: str,
    execution_uuid: str,
    common_params: dict[str, object],
) -> list[CandidateEvaluation]:
    """Log each evaluated candidate as a back-dated MLflow parent run.

    Parallel results arrive in completion order; run names keep each
    candidate's grid position.

    Args:
        evaluations: Candidate evaluations in the order they complete.
        candidates: The searched grid, defining run numbers and result order.
        client: MLflow client used to create and terminate the runs.
        experiment_id: Experiment receiving the runs.
        execution_uuid: The notebook execution's ``execution_uuid`` tag.
        common_params: Params logged on every candidate run.

    Returns:
        The logged evaluations, sorted into grid order.
    """
    import mlflow
    from tqdm.auto import tqdm  # type: ignore[import-untyped]

    from src.tracking import backdated_run, run_tags

    logged: list[CandidateEvaluation] = []
    for evaluation in tqdm(
        evaluations,
        total=len(candidates),
        desc="ARIMAX CV search",
        unit="candidate",
    ):
        number = candidates.index(evaluation.candidate) + 1
        with backdated_run(
            client,
            experiment_id=experiment_id,
            run_name=f"arimax_candidate_{number:03d}",
            tags=run_tags("cv", "candidate_parent", execution_uuid),
            start_time_ms=evaluation.started_at_ms,
            end_time_ms=evaluation.ended_at_ms,
        ):
            mlflow.log_params(common_params)
            log_candidate(evaluation)
        logged.append(evaluation)
        tqdm.write(
            f"Candidate {number}/{len(candidates)} ({len(logged)} done): {evaluation.candidate.subset}, {evaluation.candidate.order}, {'ok' if evaluation.summary else 'failed'}"
        )
    logged.sort(key=lambda result: candidates.index(result.candidate))
    return logged
