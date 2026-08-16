"""Ridge-specific MLflow candidate-selection and saved-model helpers."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as load_joblib  # type: ignore[import-untyped]

from src.config import CV_SELECTION_METRIC, FORECAST_HORIZON_HOURS
from src.training import (
    JoinedFeatureContract,
    mlflow_finite_float,
    mlflow_run_series,
    numeric_predictors,
    validate_predictions,
)

CV_METRIC_COLUMNS = [
    "metrics.cv_mae_mean",
    "metrics.cv_mae_std",
    "metrics.cv_rmse_mean",
    "metrics.cv_rmse_std",
    "metrics.cv_me_mean",
    "metrics.cv_me_std",
    "metrics.cv_r2_mean",
    "metrics.cv_r2_std",
]
CANDIDATE_ID_COLUMNS = [
    "tags.subset",
    "params.alpha",
    "params.feature_count",
]


def select_candidate(
    cv_results: pd.DataFrame, metric: str = "rmse"
) -> tuple[str, float]:
    """Select one subset/alpha pair with deterministic tie-breaking.

    Args:
        cv_results: One row per candidate subset/alpha with mean CV metrics.
        metric: Ranking metric, either ``"mae"`` or ``"rmse"``.

    Returns:
        The winning candidate's subset name and alpha.

    Raises:
        ValueError: If the metric is invalid, results are empty or missing
            required columns, or ranking values contain nulls.
    """
    if metric not in {"mae", "rmse"}:
        raise ValueError("CV selection metric must be either 'mae' or 'rmse'")
    metric_column = f"{metric}_mean"
    required_columns = {"subset", "alpha", "feature_count", metric_column}
    missing = sorted(required_columns.difference(cv_results.columns))
    if missing or cv_results.empty:
        raise ValueError(f"CV results are empty or missing columns: {missing}")
    if cv_results[[metric_column, "alpha", "feature_count"]].isna().any().any():
        raise ValueError("CV candidate results contain null ranking values")
    ranked = cv_results.sort_values(
        [metric_column, "feature_count", "alpha", "subset"],
        kind="stable",
    )
    winner = ranked.iloc[0]
    return str(winner["subset"]), float(winner["alpha"])


def _parse_subset_alpha(subset: object, alpha: object) -> tuple[str, float] | None:
    """Return a normalized (subset, alpha) key, or None if either is invalid.

    Args:
        subset: Raw MLflow subset tag value.
        alpha: Raw MLflow alpha param value.

    Returns:
        The stringified subset paired with a finite alpha, or ``None`` if the
        subset is missing/blank or the alpha is not a finite number.
    """
    parsed_alpha = mlflow_finite_float(alpha)
    if (
        subset is None
        or pd.isna(subset)  # type: ignore[call-overload]
        or not str(subset).strip()
        or parsed_alpha is None
    ):
        return None
    return str(subset), parsed_alpha


def validate_execution(
    sealed_test_run: pd.Series, all_runs: pd.DataFrame
) -> dict[str, object]:
    """Validate one sealed-test marker and assemble its candidate data.

    Args:
        sealed_test_run: One ``run_type == "sealed_test"`` MLflow run row.
        all_runs: All Ridge MLflow runs, including candidate-parent rows.

    Returns:
        A dict with ``candidate_table``, ``selected_candidate``,
        ``sealed_test_run``, ``sealed_test_metrics``, and ``horizon_records``.

    Raises:
        ValueError: If candidate-parent rows are missing or malformed, the
            sealed-test candidate identifiers are invalid or unmatched, or
            sealed-test metrics are missing or incomplete.
    """
    execution_uuid = str(sealed_test_run["tags.execution_uuid"])
    execution_candidates = all_runs.loc[
        mlflow_run_series(all_runs, "tags.execution_uuid").eq(execution_uuid)
        & mlflow_run_series(all_runs, "tags.run_type").eq("candidate_parent")
    ].copy()
    if execution_candidates.empty:
        raise ValueError("candidate-parent rows are missing")

    required_columns = CANDIDATE_ID_COLUMNS + CV_METRIC_COLUMNS
    missing_columns = [
        column
        for column in required_columns
        if column not in execution_candidates.columns
    ]
    if missing_columns:
        raise ValueError(
            f"candidate-parent rows are missing required fields: {missing_columns}"
        )

    candidate_records = []
    candidate_keys = []
    for _, candidate_run in execution_candidates.iterrows():
        feature_count = mlflow_finite_float(candidate_run["params.feature_count"])
        parsed_candidate = _parse_subset_alpha(
            candidate_run["tags.subset"], candidate_run["params.alpha"]
        )
        if (
            parsed_candidate is None
            or feature_count is None
            or not feature_count.is_integer()
        ):
            raise ValueError("candidate-parent identifiers are invalid")
        subset, alpha = parsed_candidate
        metric_values = {
            metric_column.removeprefix("metrics."): mlflow_finite_float(
                candidate_run[metric_column]
            )
            for metric_column in CV_METRIC_COLUMNS
        }
        missing_metrics = [
            metric_name
            for metric_name, metric_value in metric_values.items()
            if metric_value is None
        ]
        if missing_metrics:
            raise ValueError(
                f"candidate-parent CV metrics are missing: {missing_metrics}"
            )
        candidate_keys.append(parsed_candidate)
        candidate_records.append(
            {
                "subset": subset,
                "feature_count": int(feature_count),
                "alpha": alpha,
                **metric_values,
            }
        )

    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("candidate (subset, alpha) keys are not unique")
    candidate_table = (
        pd.DataFrame(candidate_records)
        .sort_values(f"cv_{CV_SELECTION_METRIC}_mean", kind="stable")
        .reset_index(drop=True)
    )

    parsed_sealed = _parse_subset_alpha(
        sealed_test_run.get("tags.subset"), sealed_test_run.get("params.alpha")
    )
    if parsed_sealed is None:
        raise ValueError("sealed-test candidate identifiers are invalid")
    sealed_subset, sealed_alpha = parsed_sealed
    if parsed_sealed not in set(candidate_keys):
        raise ValueError("sealed-test candidate does not match a candidate-parent row")

    sealed_test_metric_columns = [
        "metrics.test_mae",
        "metrics.test_rmse",
        "metrics.test_me",
        "metrics.test_r2",
    ]
    sealed_test_metrics = {
        column.removeprefix("metrics."): mlflow_finite_float(
            sealed_test_run.get(column)
        )
        for column in sealed_test_metric_columns
    }
    missing_test_metrics = [
        metric_name
        for metric_name, metric_value in sealed_test_metrics.items()
        if metric_value is None
    ]
    if missing_test_metrics:
        raise ValueError(f"sealed-test metrics are missing: {missing_test_metrics}")

    horizon_columns: dict[int, dict[str, str]] = {}
    for column in all_runs.columns:
        match = re.fullmatch(
            r"metrics\.(test_mae|test_rmse|test_me|test_r2)_horizon_(\d+)", str(column)
        )
        if match:
            metric_name, horizon = match.groups()
            horizon_columns.setdefault(int(horizon), {})[metric_name] = column
    horizon_records = []
    for horizon in sorted(horizon_columns):
        metric_columns = horizon_columns[horizon]
        if {"test_mae", "test_rmse", "test_me", "test_r2"}.difference(metric_columns):
            continue
        horizon_values = {
            metric_name: mlflow_finite_float(sealed_test_run.get(column))
            for metric_name, column in metric_columns.items()
        }
        if any(value is None for value in horizon_values.values()):
            raise ValueError("sealed-test per-horizon metrics are incomplete")
        horizon_records.append({"horizon_hours": horizon, **horizon_values})
    if not horizon_records:
        raise ValueError("sealed-test per-horizon metrics are missing")
    expected_horizons = set(range(1, FORECAST_HORIZON_HOURS + 1))
    if {record["horizon_hours"] for record in horizon_records} != expected_horizons:
        raise ValueError(
            "sealed-test per-horizon metrics do not match the forecast contract"
        )

    selected_candidate = candidate_table.loc[
        candidate_table["subset"].eq(sealed_subset)
        & candidate_table["alpha"].eq(sealed_alpha)
    ].iloc[0]
    return {
        "candidate_table": candidate_table,
        "selected_candidate": selected_candidate,
        "sealed_test_run": sealed_test_run,
        "sealed_test_metrics": sealed_test_metrics,
        "horizon_records": horizon_records,
    }


def select_execution(runs: pd.DataFrame) -> dict[str, object]:
    """Select the most recent valid sealed-test execution from Ridge MLflow runs.

    Iterates FINISHED sealed-test marker runs from most-recent to
    least-recent ``end_time``, validating each against its candidate-parent
    rows via :func:`validate_execution`, and returns the first execution that
    passes.

    Args:
        runs: All Ridge MLflow runs, including candidate-parent rows.

    Returns:
        A dict with ``execution_uuid`` plus the ``validate_execution`` fields
        (``candidate_table``, ``selected_candidate``, ``sealed_test_run``,
        ``sealed_test_metrics``, ``horizon_records``).

    Raises:
        ValueError: If there are no runs, no FINISHED sealed-test runs, or
            every finished sealed-test execution fails validation.
    """
    if runs.empty:
        raise ValueError(
            "No valid Ridge MLflow execution exists: the Ridge experiment has no runs. "
            "Run 04_train_ridge.ipynb through its sealed-test cell first."
        )
    finished_sealed_test_runs = runs.loc[
        mlflow_run_series(runs, "status").astype("string").str.upper().eq("FINISHED")
        & mlflow_run_series(runs, "tags.run_type").eq("sealed_test")
        & mlflow_run_series(runs, "tags.execution_uuid").notna()
    ].copy()
    if finished_sealed_test_runs.empty:
        raise ValueError(
            "No valid Ridge MLflow execution exists: no FINISHED sealed-test run was "
            "found. Run 04_train_ridge.ipynb through its sealed-test cell first."
        )
    finished_sealed_test_runs["_finished_at"] = pd.to_datetime(
        mlflow_run_series(finished_sealed_test_runs, "end_time"),
        errors="coerce",
        utc=True,
    )
    finished_sealed_test_runs = finished_sealed_test_runs.sort_values(
        ["_finished_at", "run_id"],
        ascending=[False, False],
        na_position="last",
    ).drop_duplicates("tags.execution_uuid", keep="first")

    execution_failures = []
    selected_execution: dict[str, object] | None = None
    selected_execution_uuid: str | None = None
    for _, sealed_test_run in finished_sealed_test_runs.iterrows():
        execution_uuid = str(sealed_test_run["tags.execution_uuid"])
        try:
            selected_execution = validate_execution(sealed_test_run, runs)
        except ValueError as error:
            execution_failures.append(f"{execution_uuid}: {error}")
            continue
        selected_execution_uuid = execution_uuid
        break
    if selected_execution is None:
        failure_details = "; ".join(execution_failures)
        raise ValueError(
            "No valid Ridge MLflow execution exists. Each finished sealed-test marker "
            f"failed the candidate sanity checks: {failure_details}. "
            "Run 04_train_ridge.ipynb through a successful sealed-test cell."
        )
    return {"execution_uuid": selected_execution_uuid, **selected_execution}


def load_and_score_saved_model(
    model_path: Path,
    manifest_path: Path,
    contract: JoinedFeatureContract,
    feature_subsets: dict[str, list[str]],
    test_rows: pd.DataFrame,
    selected_candidate: pd.Series,
    execution_uuid: str,
    *,
    forecast_horizon_hours: int,
    target_station_id: str,
) -> np.ndarray:
    """Reload the saved Ridge model, cross-check its manifest, and score it.

    Args:
        model_path: Saved Ridge joblib model path.
        manifest_path: Saved Ridge model manifest JSON path.
        contract: Current joined-feature contract for the target station.
        feature_subsets: Named feature subsets built from ``contract``.
        test_rows: Eligible sealed-test rows to score.
        selected_candidate: The MLflow-selected candidate row (subset/alpha).
        execution_uuid: The MLflow execution UUID the manifest must match.
        forecast_horizon_hours: Required direct-forecast horizon.
        target_station_id: Required target station identifier.

    Returns:
        Validated sealed-test predictions ordered like
        ``contract.target_columns``.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If the manifest disagrees with the current contract, the
            selected MLflow candidate, or the execution UUID.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Ridge model manifest: {manifest_path}")
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest_station_id = model_manifest.get("station_id")
    if manifest_station_id != target_station_id:
        raise ValueError(
            "Saved Ridge manifest station_id does not match the current target station"
        )
    if model_manifest.get("forecast_horizon_hours") != forecast_horizon_hours:
        raise ValueError(
            "Saved Ridge manifest forecast horizon does not match the current contract"
        )
    if tuple(model_manifest.get("full_feature_columns", ())) != tuple(
        contract.predictor_columns
    ):
        raise ValueError(
            "Saved Ridge manifest full feature contract does not match the current "
            "contract"
        )
    if tuple(model_manifest.get("target_columns", ())) != tuple(
        contract.target_columns
    ):
        raise ValueError(
            "Saved Ridge manifest target contract does not match the current contract"
        )

    manifest_execution_uuid = model_manifest.get("execution_uuid")
    if manifest_execution_uuid != execution_uuid:
        raise ValueError(
            "Saved Ridge manifest execution_uuid does not match the selected MLflow "
            "execution"
        )
    manifest_subset = model_manifest.get("selected_subset")
    if manifest_subset != selected_candidate["subset"]:
        raise ValueError(
            "Saved Ridge manifest selected subset does not match the selected MLflow "
            "candidate"
        )
    if model_manifest.get("feature_subset", manifest_subset) != manifest_subset:
        raise ValueError("Saved Ridge manifest feature subset aliases disagree")
    manifest_alpha = mlflow_finite_float(model_manifest.get("selected_alpha"))
    if manifest_alpha is None or manifest_alpha != float(selected_candidate["alpha"]):
        raise ValueError(
            "Saved Ridge manifest selected alpha does not match the selected MLflow "
            "candidate"
        )
    feature_columns = tuple(model_manifest.get("selected_feature_columns", ()))
    if not feature_columns or not set(feature_columns).issubset(
        set(contract.predictor_columns)
    ):
        raise ValueError(
            "Saved Ridge manifest selected features do not match the current contract"
        )
    manifest_feature_subsets = model_manifest.get("feature_subsets", {})
    if manifest_subset in feature_subsets:
        if feature_columns != tuple(feature_subsets[manifest_subset]):
            raise ValueError(
                "Saved Ridge manifest selected features do not match the selected "
                "subset contract"
            )
        if manifest_feature_subsets and tuple(
            manifest_feature_subsets.get(manifest_subset, ())
        ) != tuple(feature_subsets[manifest_subset]):
            raise ValueError(
                "Saved Ridge manifest feature subset does not match the current "
                "contract"
            )

    model = load_joblib(model_path)
    return validate_predictions(
        model.predict(numeric_predictors(test_rows, feature_columns)),
        expected_rows=len(test_rows),
        target_columns=list(contract.target_columns),
        artifact_name="saved Ridge sealed-test",
    )
