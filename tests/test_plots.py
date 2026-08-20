from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest

from src import plots
from src.plots import (
    aggregate_comparison_figure,
    alarm_threshold_comparison_figure,
    alarm_threshold_horizon_figure,
    cv_error_boxplots_figure,
    feature_subset_best_comparison_figure,
    feature_subset_candidate_distribution_figure,
    forecast_window_figures,
    horizon_comparison_figure,
    model_feature_subset_candidate_distribution_figure,
    predicted_vs_actual_figure,
    quartile_comparison_figure,
    quartile_horizon_small_multiples_figure,
)

# `plots.test_error_boxplots_figure` is reached through the module so pytest does
# not collect the imported builder itself as a test case.
build_test_error_boxplots_figure = plots.test_error_boxplots_figure


def _comparison_metrics() -> pd.DataFrame:
    rows = []
    for phase in ("cross_validation", "sealed_test"):
        for model_number, model in enumerate(("Persistence", "Ridge"), start=1):
            for metric_number, metric in enumerate(
                ("mae", "rmse", "me", "r2"), start=1
            ):
                rows.append(
                    {
                        "model": model,
                        "phase": phase,
                        "metric": metric,
                        "scope": "aggregate",
                        "horizon": pd.NA,
                        "value": float(model_number + metric_number),
                        "cv_std": 0.25 if phase == "cross_validation" else pd.NA,
                    }
                )
                for horizon in (1, 2):
                    rows.append(
                        {
                            "model": model,
                            "phase": phase,
                            "metric": metric,
                            "scope": "horizon",
                            "horizon": horizon,
                            "value": float(model_number + metric_number + horizon),
                            "cv_std": (0.25 if phase == "cross_validation" else pd.NA),
                        }
                    )
    return pd.DataFrame(rows)


def _feature_subset_metrics() -> pd.DataFrame:
    rows = []
    for model_number, model in enumerate(("Ridge", "XGBoost"), start=1):
        for subset_number, subset in enumerate(
            ("full", "target_station_full"), start=1
        ):
            for candidate_number in (1, 2):
                row: dict[str, object] = {
                    "model": model,
                    "subset": subset,
                    "candidate_parameters": f"candidate={candidate_number}",
                    "is_best_within_subset": candidate_number == 1,
                }
                for metric_number, metric in enumerate(
                    ("mae", "rmse", "me", "r2"), start=1
                ):
                    row[f"cv_{metric}_mean"] = float(
                        model_number + subset_number + candidate_number + metric_number
                    )
                    row[f"cv_{metric}_std"] = metric_number / 10
                rows.append(row)
    return pd.DataFrame(rows)


def _model_candidate_metrics() -> pd.DataFrame:
    rows = []
    for subset_number, subset in enumerate(
        ("target_station_full", "full", "raw_all_stations"), start=1
    ):
        for candidate_number in (1, 2):
            row: dict[str, object] = {
                "subset": subset,
                "feature_count": subset_number * 10,
                "alpha": candidate_number / 10,
            }
            for metric_number, metric in enumerate(
                ("mae", "rmse", "me", "r2"), start=1
            ):
                row[f"{metric}_mean"] = float(
                    subset_number + candidate_number + metric_number
                )
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "builder", [aggregate_comparison_figure, horizon_comparison_figure]
)
def test_comparison_figures_show_all_metrics_and_cv_uncertainty(
    builder: object,
) -> None:
    figure = builder(_comparison_metrics(), phase="cross_validation")  # type: ignore[operator]

    assert isinstance(figure, go.Figure)
    assert [annotation.text for annotation in figure.layout.annotations] == [
        "MAE",
        "RMSE",
        "ME",
        "R²",
    ]
    assert all(trace.error_y.visible for trace in figure.data)
    colors_by_model: dict[str, set[str]] = {}
    for trace in figure.data:
        color = trace.line.color or trace.marker.color
        colors_by_model.setdefault(trace.name, set()).add(color)
    assert all(len(colors) == 1 for colors in colors_by_model.values())
    assert len({next(iter(colors)) for colors in colors_by_model.values()}) == len(
        colors_by_model
    )


@pytest.mark.parametrize(
    "builder", [aggregate_comparison_figure, horizon_comparison_figure]
)
def test_sealed_test_comparison_figures_have_no_error_bars(builder: object) -> None:
    figure = builder(_comparison_metrics(), phase="sealed_test")  # type: ignore[operator]

    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) == 4
    assert all(trace.error_y.visible is None for trace in figure.data)


def test_horizon_comparison_figure_uses_same_grid_interval_in_every_panel() -> None:
    figure = horizon_comparison_figure(_comparison_metrics(), phase="cross_validation")

    assert all(
        getattr(figure.layout, axis_name).dtick == 1
        for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4")
    )


def _regime_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ("Persistence", "RNN"):
        for phase, regimes in (
            ("sealed_test_quartile", ("Q1", "Q2", "Q3", "Q4")),
            ("sealed_test_alarm", ("Alarm",)),
        ):
            for regime in regimes:
                for scope in ("aggregate", "horizon"):
                    horizons = (None,) if scope == "aggregate" else (1, 2)
                    for horizon in horizons:
                        for metric in ("mae", "rmse", "me"):
                            empty = model == "RNN" and regime == "Q4"
                            rows.append(
                                {
                                    "model": model,
                                    "phase": phase,
                                    "scope": scope,
                                    "horizon": horizon,
                                    "metric": metric,
                                    "value": np.nan if empty else 1.0,
                                    "cv_std": pd.NA,
                                    "regime": regime,
                                    "lower_bound_cm": pd.NA,
                                    "upper_bound_cm": pd.NA,
                                    "scored_values": 0 if empty else 2,
                                }
                            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "builder",
    [quartile_comparison_figure, alarm_threshold_comparison_figure],
)
def test_regime_comparison_figures_keep_sparse_cells(builder: object) -> None:
    figure = builder(_regime_metrics())  # type: ignore[operator]

    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) == 3
    assert len(figure.data) == 2 * 3
    if builder is quartile_comparison_figure:
        assert any(np.isnan(value) for trace in figure.data for value in trace.y)


@pytest.mark.parametrize(
    "builder",
    [quartile_horizon_small_multiples_figure, alarm_threshold_horizon_figure],
)
def test_regime_horizon_figures_support_different_cohorts(builder: object) -> None:
    figure = builder(_regime_metrics())  # type: ignore[operator]

    assert isinstance(figure, go.Figure)
    assert (
        len(figure.data) == 8
        if builder is quartile_horizon_small_multiples_figure
        else 2
    )


def test_feature_subset_best_figure_uses_one_winner_per_model_and_subset() -> None:
    comparison_metrics = _comparison_metrics()
    figure = feature_subset_best_comparison_figure(
        _feature_subset_metrics(), comparison_metrics
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 3 * 4
    assert all(trace.error_y.visible for trace in figure.data)
    assert all(len(trace.x) == 2 for trace in figure.data)
    baseline_traces = [
        trace for trace in figure.data if trace.name == "Persistence baseline"
    ]
    assert len(baseline_traces) == 4
    assert all(trace.line.dash == "dash" for trace in baseline_traces)
    assert [trace.y[0] for trace in baseline_traces] == [2.0, 3.0, 4.0, 5.0]
    assert [annotation.text for annotation in figure.layout.annotations] == [
        "MAE",
        "RMSE",
        "ME",
        "R²",
    ]


def test_feature_subset_candidate_distribution_shows_every_candidate() -> None:
    metrics = _feature_subset_metrics()

    figure = feature_subset_candidate_distribution_figure(
        metrics, _comparison_metrics()
    )

    assert isinstance(figure, go.Figure)
    box_traces = [trace for trace in figure.data if trace.type == "box"]
    baseline_traces = [
        trace for trace in figure.data if trace.name == "Persistence baseline"
    ]
    assert len(box_traces) == 2 * 4
    assert all(len(trace.y) == 4 for trace in box_traces)
    assert len(baseline_traces) == 4
    assert all(trace.type == "scatter" for trace in baseline_traces)
    assert all(trace.line.dash == "dash" for trace in baseline_traces)
    assert figure.layout.boxmode == "group"


def test_model_feature_subset_candidate_distribution_shows_every_candidate() -> None:
    metrics = _model_candidate_metrics()

    figure = model_feature_subset_candidate_distribution_figure(
        metrics,
        model_name="Ridge",
        hover_columns=("feature_count", "alpha"),
    )

    assert isinstance(figure, go.Figure)
    assert [annotation.text for annotation in figure.layout.annotations] == [
        "MAE candidate means",
        "RMSE candidate means",
        "ME candidate means",
        "R² candidate means",
    ]
    assert len(figure.data) == 3 * 4
    expected_trace_names = [
        "Full",
        "Raw, all stations",
        "Target-station full",
    ]
    for panel_number in range(4):
        panel_traces = figure.data[panel_number * 3 : (panel_number + 1) * 3]
        assert [trace.name for trace in panel_traces] == expected_trace_names
        assert sum(len(trace.y) for trace in panel_traces) == len(metrics)
        assert all(trace.type == "box" for trace in panel_traces)
        assert all(trace.boxpoints == "all" for trace in panel_traces)
    assert list(figure.layout.xaxis.categoryarray) == [
        "Full",
        "Raw,<br>all stations",
        "Target-station<br>full",
    ]
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].y0 == 0.0
    assert figure.layout.shapes[0].y1 == 0.0
    assert "not fold uncertainty" in figure.layout.title.text

    first_trace = figure.data[0]
    assert "feature_count=%{customdata[0]}" in first_trace.hovertemplate
    assert "alpha=%{customdata[1]}" in first_trace.hovertemplate
    assert first_trace.customdata.tolist() == [[20, 0.1], [20, 0.2]]


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda frame: frame.drop(columns="r2_mean"), "missing columns"),
        (
            lambda frame: frame.assign(rmse_mean=np.inf),
            "non-finite metric values",
        ),
        (
            lambda frame: frame.assign(subset="not_a_feature_subset"),
            "subset names",
        ),
    ],
)
def test_model_feature_subset_candidate_distribution_rejects_invalid_rows(
    mutation: object,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        model_feature_subset_candidate_distribution_figure(
            mutation(_model_candidate_metrics()),  # type: ignore[operator]
            model_name="Ridge",
        )


def test_model_feature_subset_candidate_distribution_rejects_invalid_hover_column() -> (
    None
):
    with pytest.raises(ValueError, match="hover columns"):
        model_feature_subset_candidate_distribution_figure(
            _model_candidate_metrics(),
            model_name="Ridge",
            hover_columns=("missing_parameter",),
        )


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
