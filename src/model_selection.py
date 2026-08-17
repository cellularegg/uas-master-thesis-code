"""Estimator-agnostic candidate ranking shared by every CV-searched model."""

from collections.abc import Sequence

import pandas as pd


def select_candidate(
    cv_results: pd.DataFrame, metric: str = "rmse", *, tie_break_columns: Sequence[str]
) -> pd.Series:
    """Select one candidate row with deterministic tie-breaking.

    Args:
        cv_results: One row per candidate with mean CV metrics.
        metric: Ranking metric, either ``"mae"`` or ``"rmse"``.
        tie_break_columns: Ordered columns applied after the metric to break ties.

    Returns:
        The winning candidate row.

    Raises:
        ValueError: If the metric is invalid, results are empty or missing
            required columns, or ranking values contain nulls.
    """
    if metric not in {"mae", "rmse"}:
        raise ValueError("CV selection metric must be either 'mae' or 'rmse'")
    metric_column = f"{metric}_mean"
    tie_break_columns = list(tie_break_columns)
    required_columns = {metric_column, *tie_break_columns}
    missing = sorted(required_columns.difference(cv_results.columns))
    if missing or cv_results.empty:
        raise ValueError(f"CV results are empty or missing columns: {missing}")
    if cv_results[[metric_column, *tie_break_columns]].isna().any().any():
        raise ValueError("CV candidate results contain null ranking values")
    ranked = cv_results.sort_values([metric_column, *tie_break_columns], kind="stable")
    return ranked.iloc[0]


def tie_breaking_policy(
    selection_metric: str, tie_break_descriptions: Sequence[str]
) -> list[str]:
    """Describe the candidate tie-breaking order :func:`select_candidate` applies.

    Args:
        selection_metric: Aggregate CV metric used to rank candidates.
        tie_break_descriptions: Ordered tie-break rules applied after the metric.

    Returns:
        The ordered tie-breaking rules, recorded in the saved manifest.
    """
    return [
        f"lowest aggregate CV {selection_metric.upper()}",
        *tie_break_descriptions,
    ]
