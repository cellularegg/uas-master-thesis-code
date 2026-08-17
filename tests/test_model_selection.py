import numpy as np
import pandas as pd
import pytest

from src.model_selection import select_candidate, tie_breaking_policy

TIE_BREAK_COLUMNS = ["feature_count", "alpha", "subset"]


def test_select_candidate_ranks_by_metric_then_tie_break_columns_in_order() -> None:
    winner = select_candidate(
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
        ),
        tie_break_columns=TIE_BREAK_COLUMNS,
    )
    assert winner["subset"] == "Lean"

    tie_on_metric = pd.DataFrame(
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
    assert (
        select_candidate(tie_on_metric, tie_break_columns=TIE_BREAK_COLUMNS)["subset"]
        == "Lean"
    )

    tie_on_first_tie_break = pd.DataFrame(
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
    assert (
        select_candidate(tie_on_first_tie_break, tie_break_columns=TIE_BREAK_COLUMNS)[
            "alpha"
        ]
        == 0.1
    )

    tie_on_all_but_last = pd.DataFrame(
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
    assert (
        select_candidate(tie_on_all_but_last, tie_break_columns=TIE_BREAK_COLUMNS)[
            "subset"
        ]
        == "Alpha"
    )


def test_select_candidate_respects_the_requested_metric() -> None:
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
    assert (
        select_candidate(different_metrics, tie_break_columns=TIE_BREAK_COLUMNS)[
            "subset"
        ]
        == "Low RMSE"
    )
    assert (
        select_candidate(
            different_metrics, metric="mae", tie_break_columns=TIE_BREAK_COLUMNS
        )["subset"]
        == "Low MAE"
    )


def test_select_candidate_rejects_an_invalid_metric() -> None:
    with pytest.raises(ValueError, match="must be either 'mae' or 'rmse'"):
        select_candidate(
            pd.DataFrame([{"subset": "A", "mae_mean": 1.0, "rmse_mean": 1.0}]),
            metric="mse",
            tie_break_columns=["subset"],
        )


def test_select_candidate_rejects_empty_or_missing_columns() -> None:
    with pytest.raises(ValueError, match="empty or missing columns"):
        select_candidate(pd.DataFrame(), tie_break_columns=["subset"])
    with pytest.raises(ValueError, match="empty or missing columns"):
        select_candidate(
            pd.DataFrame([{"rmse_mean": 1.0}]), tie_break_columns=["subset"]
        )


def test_select_candidate_rejects_null_ranking_values() -> None:
    with pytest.raises(ValueError, match="null ranking values"):
        select_candidate(
            pd.DataFrame([{"subset": "A", "rmse_mean": np.nan}]),
            tie_break_columns=["subset"],
        )
    with pytest.raises(ValueError, match="null ranking values"):
        select_candidate(
            pd.DataFrame([{"subset": None, "rmse_mean": 1.0}]),
            tie_break_columns=["subset"],
        )


def test_tie_breaking_policy_composes_metric_and_descriptions() -> None:
    assert tie_breaking_policy("rmse", ["fewer features", "smaller alpha"]) == [
        "lowest aggregate CV RMSE",
        "fewer features",
        "smaller alpha",
    ]
