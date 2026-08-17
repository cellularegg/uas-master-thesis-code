import pandas as pd

from src.training import (
    numeric_predictors,
    prediction_preview,
    summarize_cv_metrics,
    validate_predictions,
)


def test_numeric_predictors_converts_selected_columns_to_float() -> None:
    frame = pd.DataFrame(
        {
            "water": ["1.5", "2.5"],
            "imputed": [False, True],
            "ignored": ["not numeric", "not numeric"],
        }
    )

    numeric = numeric_predictors(frame, ["water", "imputed"])

    assert numeric.to_dict(orient="list") == {
        "water": [1.5, 2.5],
        "imputed": [0.0, 1.0],
    }
    assert all(dtype == float for dtype in numeric.dtypes)


def test_validate_predictions_returns_finite_expected_shape() -> None:
    predictions = [[1, 2], [3, 4]]

    validated = validate_predictions(
        predictions,
        expected_rows=2,
        target_columns=("target-1", "target-2"),
        artifact_name="validation fold",
    )

    assert validated.dtype == float
    assert validated.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_summarize_cv_metrics_calculates_literal_fold_statistics() -> None:
    fold_aggregate_metrics = pd.DataFrame(
        {"mae": [1.0, 3.0], "rmse": [2.0, 4.0], "me": [1.0, 3.0], "r2": [2.0, 4.0]}
    )
    fold_horizon_metrics = pd.DataFrame(
        {
            "horizon_hours": [1, 2, 1, 2],
            "mae": [1.0, 2.0, 3.0, 4.0],
            "rmse": [2.0, 3.0, 4.0, 5.0],
            "me": [1.0, 2.0, 3.0, 4.0],
            "r2": [2.0, 3.0, 4.0, 5.0],
        }
    )

    summary = summarize_cv_metrics(
        fold_aggregate_metrics,
        fold_horizon_metrics,
    )

    assert summary == {
        "cv_mae_mean": 2.0,
        "cv_mae_std": 1.0,
        "cv_rmse_mean": 3.0,
        "cv_rmse_std": 1.0,
        "cv_me_mean": 2.0,
        "cv_me_std": 1.0,
        "cv_r2_mean": 3.0,
        "cv_r2_std": 1.0,
        "cv_mae_horizon_01_mean": 2.0,
        "cv_mae_horizon_01_std": 1.0,
        "cv_rmse_horizon_01_mean": 3.0,
        "cv_rmse_horizon_01_std": 1.0,
        "cv_me_horizon_01_mean": 2.0,
        "cv_me_horizon_01_std": 1.0,
        "cv_r2_horizon_01_mean": 3.0,
        "cv_r2_horizon_01_std": 1.0,
        "cv_mae_horizon_02_mean": 3.0,
        "cv_mae_horizon_02_std": 1.0,
        "cv_rmse_horizon_02_mean": 4.0,
        "cv_rmse_horizon_02_std": 1.0,
        "cv_me_horizon_02_mean": 3.0,
        "cv_me_horizon_02_std": 1.0,
        "cv_r2_horizon_02_mean": 4.0,
        "cv_r2_horizon_02_std": 1.0,
    }


def test_prediction_preview_combines_issue_time_actuals_and_predictions() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "target-1": [1.0, 2.0],
            "target-2": [3.0, 4.0],
            "unused": [5.0, 6.0],
        }
    )

    preview = prediction_preview(
        frame,
        [[1.5, 3.5], [2.5, 4.5]],
        target_columns=("target-1", "target-2"),
    )

    assert preview.columns.tolist() == [
        "timestamp",
        "target-1",
        "target-2",
        "prediction_target-1",
        "prediction_target-2",
    ]
    assert preview[
        ["prediction_target-1", "prediction_target-2"]
    ].to_numpy().tolist() == [
        [1.5, 3.5],
        [2.5, 4.5],
    ]
