from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import plots
from src.plots import (
    cv_error_boxplots_figure,
    forecast_window_figures,
    predicted_vs_actual_figure,
)

# `plots.test_error_boxplots_figure` is reached through the module so pytest does
# not collect the imported builder itself as a test case.
build_test_error_boxplots_figure = plots.test_error_boxplots_figure


def test_predicted_vs_actual_figure_shows_each_horizon_and_combined_values() -> None:
    actual = pd.DataFrame(
        {"station-a__target_01": [1.0, 2.0], "station-a__target_02": [3.0, 4.0]}
    )

    figure = predicted_vs_actual_figure(
        actual,
        [[1.5, 3.5], [2.5, 4.5]],
        ("station-a__target_01", "station-a__target_02"),
        title="Synthetic forecast",
    )

    try:
        assert "Synthetic forecast" in [text.get_text() for text in figure.texts]
        assert [axis.get_title() for axis in figure.axes[:3]] == [
            "target_01",
            "target_02",
            "All horizons",
        ]
        assert len(figure.axes) == 4
    finally:
        plt.close(figure)


def test_cv_error_boxplots_figure_draws_one_mae_and_rmse_box_per_horizon() -> None:
    fold_horizon_metrics = pd.DataFrame(
        {
            "horizon_hours": [1, 2, 1, 2],
            "mae": [1.0, 2.0, 3.0, 4.0],
            "rmse": [2.0, 3.0, 4.0, 5.0],
        }
    )

    figure = cv_error_boxplots_figure(
        fold_horizon_metrics,
        ("target-1", "target-2"),
        title="Synthetic CV errors",
    )

    try:
        axis = figure.axes[0]
        assert [label.get_text() for label in axis.get_xticklabels()] == [
            "H+01",
            "H+02",
        ]
        assert axis.get_legend_handles_labels()[1] == ["MAE", "RMSE"]
        # Two metric series over two horizons, drawn as four boxes.
        assert len(axis.patches) == 4
        assert "Synthetic CV errors" in [text.get_text() for text in figure.texts]
    finally:
        plt.close(figure)


def test_cv_error_boxplots_figure_rejects_horizon_without_fold_metrics() -> None:
    fold_horizon_metrics = pd.DataFrame(
        {"horizon_hours": [1, 1], "mae": [1.0, 3.0], "rmse": [2.0, 4.0]}
    )

    with pytest.raises(ValueError, match="empty or non-finite"):
        cv_error_boxplots_figure(
            fold_horizon_metrics,
            ("target-1", "target-2"),
            title="Missing horizon",
        )


def test_test_error_boxplots_figure_distinguishes_errors_and_summary_markers() -> None:
    actual = pd.DataFrame(
        {"target-1": [10.0, 20.0, 30.0], "target-2": [10.0, 20.0, 30.0]}
    )
    per_horizon_metrics = pd.DataFrame(
        {
            "horizon_hours": [1, 2],
            "mae": [2.0, 4.0],
            "rmse": [2.160246899469287, 4.320493798938574],
        }
    )

    figure = build_test_error_boxplots_figure(
        actual,
        [[9.0, 8.0], [18.0, 16.0], [27.0, 24.0]],
        per_horizon_metrics,
        ("target-1", "target-2"),
        title="Synthetic test errors",
    )

    try:
        axis = figure.axes[0]
        assert axis.get_legend_handles_labels()[1] == [
            "Absolute error",
            "MAE summary",
            "RMSE summary",
        ]
        assert [label.get_text() for label in axis.get_xticklabels()] == [
            "H+01",
            "H+02",
        ]
        marker_lines = [
            line
            for line in axis.lines
            if line.get_label() in {"MAE summary", "RMSE summary"}
        ]
        assert [line.get_marker() for line in marker_lines] == ["D", "X"]
        assert [np.asarray(line.get_ydata()).tolist() for line in marker_lines] == [
            [2.0, 4.0],
            [2.160246899469287, 4.320493798938574],
        ]
        # Absolute errors are 1/2/3 at H+01 and 2/4/6 at H+02, so the boxes span
        # the upper quartiles 2.5 and 5.0.
        assert [
            float(np.asarray(box.get_path().vertices)[:, 1].max())
            for box in axis.patches
        ] == [
            2.5,
            5.0,
        ]
    finally:
        plt.close(figure)


def test_test_error_boxplots_figure_rejects_out_of_order_horizon_metrics() -> None:
    actual = pd.DataFrame({"target-1": [1.0], "target-2": [2.0]})
    per_horizon_metrics = pd.DataFrame(
        {"horizon_hours": [2, 1], "mae": [1.0, 1.0], "rmse": [1.0, 1.0]}
    )

    with pytest.raises(ValueError, match="not in forecast-horizon order"):
        build_test_error_boxplots_figure(
            actual,
            [[1.5, 2.5]],
            per_horizon_metrics,
            ("target-1", "target-2"),
            title="Out of order",
        )


class _ForecastWindowFixture(NamedTuple):
    prediction_columns: list[str]
    target_columns: list[str]
    horizons: list[int]
    prediction_table: pd.DataFrame
    water_level_column: str
    imputed_column: str
    context_series: pd.DataFrame


def _forecast_window_fixture() -> _ForecastWindowFixture:
    target_columns = ["station-a__target_t_plus_01", "station-a__target_t_plus_02"]
    prediction_columns = [f"prediction_{column}" for column in target_columns]
    issue_times = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-03 00:00"], utc=True
    )
    prediction_table = pd.DataFrame(
        {
            "issue_time": issue_times,
            target_columns[0]: [10.0, 20.0, 30.0],
            target_columns[1]: [11.0, 21.0, 31.0],
            # Squared errors of 1, 4 and 9 give per-issue RMSEs of 1, 2 and 3.
            prediction_columns[0]: [11.0, 22.0, 33.0],
            prediction_columns[1]: [12.0, 23.0, 34.0],
        }
    )
    water_level_column = "station-a__water_level"
    imputed_column = "station-a__imputed"
    context_times = pd.date_range(
        "2023-12-29 00:00", "2024-01-05 00:00", freq="h", tz="UTC"
    )
    context_series = pd.DataFrame(
        {
            water_level_column: np.arange(len(context_times), dtype=float),
            imputed_column: False,
        },
        index=context_times,
    )
    return _ForecastWindowFixture(
        prediction_columns=prediction_columns,
        target_columns=target_columns,
        horizons=[1, 2],
        prediction_table=prediction_table,
        water_level_column=water_level_column,
        imputed_column=imputed_column,
        context_series=context_series,
    )


def _forecast_window_figures(
    fixture: _ForecastWindowFixture, context_series: pd.DataFrame | None = None
) -> dict:
    return forecast_window_figures(
        fixture.prediction_table,
        fixture.context_series if context_series is None else context_series,
        water_level_column=fixture.water_level_column,
        imputed_column=fixture.imputed_column,
        prediction_columns=fixture.prediction_columns,
        target_columns=fixture.target_columns,
        horizons=fixture.horizons,
        label_prefix="Ridge",
    )


def test_forecast_window_figures_selects_best_and_worst_issue_by_rmse() -> None:
    fixture = _forecast_window_fixture()

    figures = _forecast_window_figures(fixture)

    assert set(figures) == {"best", "worst"}
    best_title = figures["best"].layout.title.text
    worst_title = figures["worst"].layout.title.text
    assert "Best-RMSE Ridge forecast window" in best_title
    assert "Worst-RMSE Ridge forecast window" in worst_title
    # Lowest RMSE is the first issue time, highest is the third.
    assert "2024-01-01T00:00:00+00:00" in best_title
    assert "24-hour RMSE: 1.0000" in best_title
    assert "2024-01-03T00:00:00+00:00" in worst_title
    assert "24-hour RMSE: 3.0000" in worst_title
    # Ground truth plus prediction, and the neighbouring issue time six hours
    # after the best issue is reachable from its slider.
    assert [trace.name for trace in figures["best"].data] == [
        "Ground truth",
        "Prediction",
    ]
    assert len(figures["best"].data[0].x) == 97
    assert len(figures["best"].frames) == 2
    assert figures["best"].layout.sliders[0].active == 0
    # The worst issue time has no eligible neighbour within twelve hours.
    assert len(figures["worst"].frames) == 1


def test_forecast_window_figures_excludes_incomplete_and_imputed_context() -> None:
    fixture = _forecast_window_fixture()
    # Only the lowest-RMSE issue time's context window reaches this hour.
    disqualifying_time = pd.Timestamp("2023-12-30 02:00", tz="UTC")
    gapped_context = fixture.context_series.drop(disqualifying_time)
    imputed_context = fixture.context_series.copy()
    imputed_context.loc[disqualifying_time, fixture.imputed_column] = True

    for context_series in (gapped_context, imputed_context):
        figures = _forecast_window_figures(fixture, context_series)

        # The best eligible issue is now the second-lowest-RMSE issue time.
        assert "2024-01-01T06:00:00+00:00" in figures["best"].layout.title.text
        assert "24-hour RMSE: 2.0000" in figures["best"].layout.title.text
        assert "2024-01-03T00:00:00+00:00" in figures["worst"].layout.title.text


def test_forecast_window_figures_requires_two_eligible_issue_times() -> None:
    fixture = _forecast_window_fixture()
    short_context = fixture.context_series.loc["2023-12-30":"2024-01-02"]

    with pytest.raises(ValueError, match="at least two issue timestamps"):
        _forecast_window_figures(fixture, short_context)
