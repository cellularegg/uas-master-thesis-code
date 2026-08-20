"""Evaluation figures for joined-data forecast models."""

import locale
import math
from collections.abc import Sequence
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from src.config import MATPLOTLIB_STYLE
from src.training import validate_predictions

plt.style.use(MATPLOTLIB_STYLE)
# Group thousands in axis tick labels: matplotlib only does this through the
# numeric locale, so pin one rather than inherit the ambient (grouping-less) C.
locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
plt.rcParams["axes.formatter.use_locale"] = True

# Hours of ground-truth context shown on either side of a plotted issue time.
_CONTEXT_WINDOW_HOURS = 48
# Hours of neighbouring issue times reachable from a forecast-window slider.
_SLIDER_WINDOW_HOURS = 12

type ComparisonPhase = Literal[
    "cross_validation",
    "sealed_test",
    "sealed_test_quartile",
    "sealed_test_alarm",
]
type BaseComparisonPhase = Literal["cross_validation", "sealed_test"]
type RegimePhase = Literal["sealed_test_quartile", "sealed_test_alarm"]

_COMPARISON_METRIC_TITLES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "me": "ME",
    "r2": "R²",
}
_REGIME_METRIC_TITLES = {"mae": "MAE", "rmse": "RMSE", "me": "ME"}
_QUARTILE_REGIME_LABELS = ("Q1", "Q2", "Q3", "Q4")
type RegimeGroup = Literal["quartile", "alarm"]
type RegimeMetric = Literal["mae", "rmse", "me"]
_COMPARISON_MODEL_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)
_PERSISTENCE_BASELINE_COLOR = "#7F7F7F"
_FEATURE_SUBSET_ORDER = (
    "full",
    "all_station_hydrology_quality_time",
    "raw_all_stations",
    "target_station_full",
    "target_station_hydrology_quality_time",
    "current_water_levels_all_stations",
)
_FEATURE_SUBSET_LABELS = {
    "full": "Full",
    "all_station_hydrology_quality_time": "All-station hydrology,<br>quality & time",
    "raw_all_stations": "Raw,<br>all stations",
    "target_station_full": "Target-station<br>full",
    "target_station_hydrology_quality_time": (
        "Target-station hydrology,<br>quality & time"
    ),
    "current_water_levels_all_stations": "Current water levels,<br>all stations",
}


def predicted_vs_actual_figure(
    actual: pd.DataFrame,
    predictions: np.ndarray | Sequence[Sequence[float]],
    target_columns: Sequence[str],
    *,
    title: str = "Predicted vs actual",
) -> plt.Figure:
    """Build horizon-specific and combined actual-versus-predicted plots.

    Args:
        actual: Frame containing actual target values.
        predictions: Direct predictions ordered like ``target_columns``.
        target_columns: Ordered actual target columns.
        title: Figure title.

    Returns:
        A scatterplot figure with one panel per horizon and one combined panel.

    Raises:
        ValueError: If predictions cannot be converted or fail validation.
    """
    target_columns = list(target_columns)
    actual_values = actual[target_columns].to_numpy(dtype=float)
    prediction_values = validate_predictions(
        predictions,
        expected_rows=len(actual),
        target_columns=target_columns,
        artifact_name="forecast",
    )
    all_values = np.concatenate((actual_values.ravel(), prediction_values.ravel()))
    axis_min = float(np.min(all_values))
    axis_max = float(np.max(all_values))
    padding = max((axis_max - axis_min) * 0.05, 1e-9)
    axis_min -= padding
    axis_max += padding

    panel_count = len(target_columns) + 1
    grid_size = math.ceil(math.sqrt(panel_count))
    figure, axes_grid = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size * 4, grid_size * 4),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes_grid).ravel()
    for horizon, target_column in enumerate(target_columns):
        axes[horizon].scatter(
            actual_values[:, horizon],
            prediction_values[:, horizon],
            s=12,
            alpha=0.35,
        )
        axes[horizon].set_title(target_column.rsplit("__", maxsplit=1)[-1])

    combined_axis = axes[len(target_columns)]
    combined_axis.scatter(
        actual_values.ravel(),
        prediction_values.ravel(),
        s=12,
        alpha=0.35,
    )
    combined_axis.set_title("All horizons")

    for axis in axes[:panel_count]:
        axis.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.grid(alpha=0.25)
    for axis in axes[panel_count:]:
        axis.set_visible(False)
    figure.supxlabel("Actual values")
    figure.supylabel("Predicted values")
    figure.suptitle(title)
    figure.tight_layout(rect=(0.03, 0.03, 1, 0.97))
    return figure


def cv_error_boxplots_figure(
    fold_horizon_metrics: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    title: str,
) -> plt.Figure:
    """Plot the fold-to-fold spread of CV MAE and RMSE per forecast horizon.

    Args:
        fold_horizon_metrics: Per-fold MAE and RMSE rows keyed by horizon.
        target_columns: Ordered targets defining the forecast horizons.
        title: Figure title.

    Returns:
        A grouped boxplot figure with one MAE and one RMSE box per horizon.

    Raises:
        ValueError: If a horizon has no fold metrics or they are non-finite.
    """
    horizons = list(range(1, len(target_columns) + 1))
    labels = [f"H+{horizon:02d}" for horizon in horizons]
    values = {
        metric.upper(): [
            fold_horizon_metrics.loc[
                fold_horizon_metrics["horizon_hours"].eq(horizon), metric
            ].to_numpy(dtype=float)
            for horizon in horizons
        ]
        for metric in ("mae", "rmse")
    }
    return _error_boxplots_figure(values, labels, title=title)


def test_error_boxplots_figure(
    actual: pd.DataFrame,
    predictions: np.ndarray | Sequence[Sequence[float]],
    per_horizon_metrics: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    title: str,
) -> plt.Figure:
    """Plot sealed-test absolute errors per horizon with MAE and RMSE markers.

    Args:
        actual: Frame containing actual target values.
        predictions: Direct predictions ordered like ``target_columns``.
        per_horizon_metrics: Ordered per-horizon MAE and RMSE rows.
        target_columns: Ordered actual target columns.
        title: Figure title.

    Returns:
        A boxplot figure of absolute errors with MAE and RMSE marker series.

    Raises:
        ValueError: If predictions or metrics do not match the forecast contract,
            or calculated errors are non-finite.
    """
    target_columns = list(target_columns)
    actual_values = actual[target_columns].to_numpy(dtype=float)
    prediction_values = validate_predictions(
        predictions,
        expected_rows=len(actual),
        target_columns=target_columns,
        artifact_name="test",
    )
    expected_horizons = np.arange(1, len(target_columns) + 1)
    if not np.array_equal(
        per_horizon_metrics["horizon_hours"].to_numpy(), expected_horizons
    ):
        raise ValueError("Horizon metrics are not in forecast-horizon order")

    absolute_errors = np.abs(actual_values - prediction_values)
    if not np.isfinite(absolute_errors).all():
        raise ValueError("Test errors contain non-finite values")
    labels = [f"H+{horizon:02d}" for horizon in expected_horizons]
    values = {
        "Absolute error": [
            absolute_errors[:, horizon - 1] for horizon in expected_horizons
        ]
    }
    markers = {
        metric.upper(): per_horizon_metrics[metric].to_numpy(dtype=float)
        for metric in ("mae", "rmse")
    }
    return _error_boxplots_figure(
        values,
        labels,
        title=title,
        summary_markers=markers,
    )


def aggregate_comparison_figure(
    comparison_metrics: pd.DataFrame,
    *,
    phase: ComparisonPhase,
) -> go.Figure:
    """Build a four-metric aggregate comparison figure for one evaluation phase.

    Args:
        comparison_metrics: Canonical tidy comparison metrics from
            :func:`src.evaluation.load_latest_complete_comparison_metrics`.
        phase: A base comparison phase or one of the sealed-test regime phases.

    Returns:
        A four-panel model comparison. Cross-validation values include their
        logged standard deviations; sealed-test values are point estimates.

    Raises:
        ValueError: If required rows are absent, duplicated, or non-finite.
    """
    if phase in ("sealed_test_quartile", "sealed_test_alarm"):
        return regime_comparison_figure(
            comparison_metrics, phase=phase, scope="aggregate"
        )
    rows = _comparison_plot_rows(
        comparison_metrics,
        phase=cast(BaseComparisonPhase, phase),
        scope="aggregate",
    )
    models = rows["model"].drop_duplicates().astype(str).tolist()
    model_colors = _comparison_model_color_map(models)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(_COMPARISON_METRIC_TITLES.values()),
    )
    for plot_number, (metric, _) in enumerate(
        _COMPARISON_METRIC_TITLES.items(), start=1
    ):
        row = (plot_number - 1) // 2 + 1
        column = (plot_number - 1) % 2 + 1
        metric_rows = rows.loc[rows["metric"].eq(metric)]
        if metric_rows["model"].duplicated().any():
            raise ValueError(f"Aggregate {metric.upper()} rows duplicate a model")
        metric_models = metric_rows["model"].astype(str).tolist()
        values = metric_rows["value"].to_numpy(dtype=float)
        standard_deviations = (
            metric_rows["cv_std"].to_numpy(dtype=float)
            if phase == "cross_validation"
            else np.full(len(metric_rows), np.nan)
        )
        for model, value, standard_deviation in zip(
            metric_models, values, standard_deviations, strict=True
        ):
            error_y = None
            if phase == "cross_validation":
                error_y = {
                    "type": "data",
                    "array": [standard_deviation],
                    "visible": True,
                }
            figure.add_trace(
                go.Scatter(
                    x=[model],
                    y=[value],
                    mode="markers",
                    name=model,
                    legendgroup=model,
                    showlegend=plot_number == 1,
                    error_y=error_y,
                    marker={"color": model_colors[model], "size": 9},
                    hovertemplate=(
                        "Model=%{x}<br>Value=%{y:.3f}<extra>"
                        f"{_COMPARISON_METRIC_TITLES[metric]}</extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(tickangle=-20, row=row, col=column)
        figure.update_yaxes(title_text="Metric value", row=row, col=column)
        if metric == "me":
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=row,
                col=column,
            )
    phase_title = (
        "Cross-validation mean ± standard deviation"
        if phase == "cross_validation"
        else "Sealed-test point estimates"
    )
    figure.update_layout(
        title=f"Aggregate model comparison — {phase_title}",
        template="plotly_white",
        height=750,
    )
    return figure


def horizon_comparison_figure(
    comparison_metrics: pd.DataFrame,
    *,
    phase: ComparisonPhase,
) -> go.Figure:
    """Build a four-metric per-horizon comparison figure for one phase.

    Args:
        comparison_metrics: Canonical tidy comparison metrics from
            :func:`src.evaluation.load_latest_complete_comparison_metrics`.
        phase: A base comparison phase or one of the sealed-test regime phases.

    Returns:
        A four-panel per-horizon model comparison. Cross-validation values
        include their logged standard deviations; sealed-test values are point
        estimates.

    Raises:
        ValueError: If required rows are absent, duplicated, incomplete, or
            non-finite.
    """
    if phase in ("sealed_test_quartile", "sealed_test_alarm"):
        return regime_comparison_figure(
            comparison_metrics, phase=phase, scope="horizon"
        )
    rows = _comparison_plot_rows(
        comparison_metrics,
        phase=cast(BaseComparisonPhase, phase),
        scope="horizon",
    )
    models = rows["model"].drop_duplicates().astype(str).tolist()
    model_colors = _comparison_model_color_map(models)
    horizons = sorted(rows["horizon"].astype(int).unique().tolist())
    if horizons != list(range(1, max(horizons) + 1)):
        raise ValueError("Horizon comparison rows do not form a contiguous horizon")

    figure = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=list(_COMPARISON_METRIC_TITLES.values()),
    )
    for plot_number, (metric, _) in enumerate(
        _COMPARISON_METRIC_TITLES.items(), start=1
    ):
        row = (plot_number - 1) // 2 + 1
        column = (plot_number - 1) % 2 + 1
        metric_rows = rows.loc[rows["metric"].eq(metric)]
        if metric_rows.duplicated(["model", "horizon"]).any():
            raise ValueError(f"Horizon {metric.upper()} rows duplicate a model/horizon")
        for model in models:
            model_rows = metric_rows.loc[metric_rows["model"].eq(model)].sort_values(
                "horizon", kind="stable"
            )
            if model_rows["horizon"].astype(int).tolist() != horizons:
                raise ValueError(
                    f"Horizon {metric.upper()} rows do not cover every horizon "
                    f"for {model}"
                )
            values = model_rows["value"].to_numpy(dtype=float)
            error_y = None
            if phase == "cross_validation":
                error_y = {
                    "type": "data",
                    "array": model_rows["cv_std"].to_numpy(dtype=float),
                    "visible": True,
                }
            figure.add_trace(
                go.Scatter(
                    x=horizons,
                    y=values,
                    mode="lines+markers",
                    name=model,
                    legendgroup=model,
                    showlegend=plot_number == 1,
                    error_y=error_y,
                    line={"color": model_colors[model]},
                    marker={"color": model_colors[model]},
                    hovertemplate=(
                        "Model=%{fullData.name}<br>Horizon=%{x} h"
                        "<br>Value=%{y:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_yaxes(title_text="Metric value", row=row, col=column)
        if metric == "me":
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=row,
                col=column,
            )
    figure.update_xaxes(dtick=1)
    for column in (1, 2):
        figure.update_xaxes(
            title_text="Forecast horizon (hours)",
            row=2,
            col=column,
        )
    phase_title = (
        "Cross-validation mean ± standard deviation"
        if phase == "cross_validation"
        else "Sealed-test point estimates"
    )
    figure.update_layout(
        title=f"Per-horizon model comparison — {phase_title}",
        template="plotly_white",
        height=750,
        hovermode="x unified",
    )
    return figure


def regime_comparison_figure(
    comparison_metrics: pd.DataFrame,
    *,
    phase: RegimePhase = "sealed_test_quartile",
    scope: Literal["aggregate", "horizon"] = "aggregate",
) -> go.Figure:
    """Compare sealed-test errors within water-level regimes.

    The figure is deliberately descriptive: regime rows are never used by
    training or candidate selection. Empty cells remain visible as gaps, which
    is important for alarm-level subsets and sparse RNN cohorts.

    Args:
        comparison_metrics: Canonical tidy comparison metrics from
            :func:`src.evaluation.load_latest_complete_comparison_metrics`.
        phase: ``sealed_test_quartile`` or ``sealed_test_alarm``.
        scope: Aggregate regime rows or per-horizon regime rows.

    Returns:
        A three-panel Plotly comparison for MAE, RMSE, and ME.

    Raises:
        ValueError: If the phase, scope, or required regime rows are invalid.
    """
    rows = _regime_plot_rows(comparison_metrics, phase=phase, scope=scope)
    models = rows["model"].drop_duplicates().astype(str).tolist()
    colors = _comparison_model_color_map(models)
    regimes = _ordered_regimes(rows["regime"], phase=phase)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=list(_REGIME_METRIC_TITLES.values()),
        shared_xaxes=scope == "aggregate" and phase == "sealed_test_alarm",
    )
    for column, metric in enumerate(_REGIME_METRIC_TITLES, start=1):
        metric_rows = rows.loc[rows["metric"].eq(metric)]
        for model in models:
            model_rows = metric_rows.loc[metric_rows["model"].eq(model)].copy()
            model_rows["__regime_order"] = model_rows["regime"].map(regimes.index)
            model_rows = model_rows.sort_values("__regime_order", kind="stable")
            # Plotly accepts NaN values and leaves a gap, allowing groups with
            # count zero to be retained in the same compact comparison figure.
            values = model_rows["value"].astype(float).to_numpy()
            if scope == "aggregate" and phase == "sealed_test_alarm":
                x_values = [model] * len(model_rows)
                mode = "markers"
            else:
                x_values = [str(regime) for regime in model_rows["regime"]]
                mode = "lines+markers"
            customdata = np.column_stack(
                [
                    model_rows["regime"].astype(str),
                    model_rows["scored_values"].astype("Int64").astype(str),
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=values,
                    mode=mode,
                    name=model,
                    legendgroup=model,
                    showlegend=column == 1,
                    line={"color": colors[model]},
                    marker={"color": colors[model]},
                    customdata=customdata,
                    connectgaps=False,
                    hovertemplate=(
                        "Model=%{fullData.name}<br>Regime=%{customdata[0]}"
                        "<br>Scored values=%{customdata[1]}<br>Value=%{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.update_yaxes(title_text="Metric value", row=1, col=column)
        if metric == "me":
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=1,
                col=column,
            )
        figure.update_xaxes(
            categoryorder="array",
            categoryarray=(models if phase == "sealed_test_alarm" else regimes),
            tickangle=-20,
            row=1,
            col=column,
        )
    definition_label = (
        "Q1–Q4 water-level regimes"
        if phase == "sealed_test_quartile"
        else "Alarm-level (actual ≥ configured threshold)"
    )
    scope_label = "aggregate" if scope == "aggregate" else "per horizon"
    figure.update_layout(
        title=f"Sealed-test regime comparison — {definition_label}, {scope_label}",
        template="plotly_white",
        height=450,
        hovermode="closest",
    )
    return figure


def quartile_comparison_figure(
    comparison_metrics: pd.DataFrame,
) -> go.Figure:
    """Build the compact Q1–Q4 aggregate comparison figure."""
    return regime_comparison_figure(
        comparison_metrics, phase="sealed_test_quartile", scope="aggregate"
    )


def alarm_threshold_comparison_figure(
    comparison_metrics: pd.DataFrame,
) -> go.Figure:
    """Build the separate aggregate comparison for the configured alarm level."""
    return regime_comparison_figure(
        comparison_metrics, phase="sealed_test_alarm", scope="aggregate"
    )


def quartile_horizon_small_multiples_figure(
    comparison_metrics: pd.DataFrame,
    *,
    metric: Literal["mae", "rmse", "me"] = "mae",
) -> go.Figure:
    """Build Q1–Q4 per-horizon small multiples for one diagnostic metric.

    Missing or empty cells are plotted as gaps; models are not required to have
    the same regime cohort counts or finite values at every horizon.
    """
    if metric not in _REGIME_METRIC_TITLES:
        raise ValueError(f"Unknown regime metric: {metric!r}")
    rows = _regime_plot_rows(
        comparison_metrics, phase="sealed_test_quartile", scope="horizon"
    )
    rows = rows.loc[rows["metric"].eq(metric)].copy()
    models = (
        comparison_metrics.loc[
            comparison_metrics["phase"].eq("sealed_test_quartile"), "model"
        ]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    if not models:
        raise ValueError("No quartile models were found")
    colors = _comparison_model_color_map(models)
    regimes = list(_QUARTILE_REGIME_LABELS)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=regimes,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for index, regime in enumerate(regimes):
        row_number = index // 2 + 1
        column_number = index % 2 + 1
        regime_rows = rows.loc[rows["regime"].eq(regime)]
        for model in models:
            model_rows = regime_rows.loc[regime_rows["model"].eq(model)].sort_values(
                "horizon", kind="stable"
            )
            figure.add_trace(
                go.Scatter(
                    x=model_rows["horizon"].astype("Int64").astype(float),
                    y=model_rows["value"].astype(float),
                    mode="lines+markers",
                    name=model,
                    legendgroup=model,
                    showlegend=index == 0,
                    line={"color": colors[model]},
                    marker={"color": colors[model]},
                    connectgaps=False,
                    hovertemplate=(
                        "Model=%{fullData.name}<br>Horizon=%{x} h"
                        "<br>Value=%{y:.3f}<extra></extra>"
                    ),
                ),
                row=row_number,
                col=column_number,
            )
        figure.update_xaxes(
            dtick=1, title_text="Horizon (hours)", row=row_number, col=column_number
        )
        figure.update_yaxes(
            title_text=_REGIME_METRIC_TITLES[metric], row=row_number, col=column_number
        )
    if metric == "me":
        for index in range(4):
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=index // 2 + 1,
                col=index % 2 + 1,
            )
    figure.update_layout(
        title=f"Sealed-test {metric.upper()} by horizon — Q1–Q4 small multiples",
        template="plotly_white",
        height=700,
        hovermode="x unified",
    )
    return figure


def alarm_threshold_horizon_figure(
    comparison_metrics: pd.DataFrame,
    *,
    metric: Literal["mae", "rmse", "me"] = "mae",
) -> go.Figure:
    """Build the separate alarm-level per-horizon diagnostic view."""
    if metric not in _REGIME_METRIC_TITLES:
        raise ValueError(f"Unknown regime metric: {metric!r}")
    rows = _regime_plot_rows(
        comparison_metrics, phase="sealed_test_alarm", scope="horizon"
    )
    rows = rows.loc[rows["metric"].eq(metric)]
    models = rows["model"].drop_duplicates().astype(str).tolist()
    colors = _comparison_model_color_map(models)
    figure = go.Figure()
    for model in models:
        model_rows = rows.loc[rows["model"].eq(model)].sort_values(
            "horizon", kind="stable"
        )
        figure.add_trace(
            go.Scatter(
                x=model_rows["horizon"].astype("Int64").astype(float),
                y=model_rows["value"].astype(float),
                mode="lines+markers",
                name=model,
                line={"color": colors[model]},
                connectgaps=False,
            )
        )
    if metric == "me":
        figure.add_hline(
            y=0.0,
            line={"color": "black", "dash": "dash", "width": 1},
        )
    figure.update_layout(
        title=f"Sealed-test alarm-level {metric.upper()} by horizon",
        xaxis_title="Forecast horizon (hours)",
        yaxis_title=_REGIME_METRIC_TITLES[metric],
        template="plotly_white",
        hovermode="x unified",
    )
    figure.update_xaxes(dtick=1)
    return figure


def regime_horizon_comparison_figure(
    comparison_metrics: pd.DataFrame,
    *,
    phase: RegimePhase = "sealed_test_quartile",
    metric: Literal["mae", "rmse", "me"] = "mae",
) -> go.Figure:
    """Dispatch to the quartile or alarm per-horizon diagnostic view."""
    if phase == "sealed_test_quartile":
        return quartile_horizon_small_multiples_figure(
            comparison_metrics, metric=metric
        )
    if phase == "sealed_test_alarm":
        return alarm_threshold_horizon_figure(comparison_metrics, metric=metric)
    raise ValueError(f"Unknown regime phase: {phase!r}")


def alarm_comparison_figure(comparison_metrics: pd.DataFrame) -> go.Figure:
    """Alias for :func:`alarm_threshold_comparison_figure`."""
    return alarm_threshold_comparison_figure(comparison_metrics)


def quartile_horizon_comparison_figure(
    comparison_metrics: pd.DataFrame,
    *,
    metric: Literal["mae", "rmse", "me"] = "mae",
) -> go.Figure:
    """Alias for :func:`quartile_horizon_small_multiples_figure`."""
    return quartile_horizon_small_multiples_figure(comparison_metrics, metric=metric)


def regime_aggregate_figure(
    aggregate_metrics: pd.DataFrame,
    *,
    regime_group: RegimeGroup,
    model_label: str,
) -> go.Figure:
    """Plot aggregate sealed-test errors for one model and regime group.

    The input is the wide aggregate table returned by
    :func:`src.metrics.water_level_regime_tables`, rather than the tidy
    cross-model table used by the evaluation notebook.  The other regime group
    may remain in the frame; it is ignored after the requested labels are
    selected.

    Args:
        aggregate_metrics: Aggregate regime metrics with ``regime``,
            ``scored_values``, ``mae``, ``rmse``, and ``me`` columns.
        regime_group: ``"quartile"`` for Q1--Q4 or ``"alarm"`` for Alarm.
        model_label: Human-readable model name used in the figure title.

    Returns:
        A three-panel Plotly figure for MAE, RMSE, and ME.

    Raises:
        ValueError: If the group, model label, or metric table is invalid.
    """
    _validate_model_label(model_label)
    rows = _validated_direct_regime_rows(
        aggregate_metrics, regime_group=regime_group, scope="aggregate"
    )
    regimes = _regime_group_labels(regime_group)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=list(_REGIME_METRIC_TITLES.values()),
        shared_xaxes=regime_group == "alarm",
    )
    for column, metric in enumerate(_REGIME_METRIC_TITLES, start=1):
        metric_rows = rows.set_index("regime").loc[list(regimes)].reset_index()
        values = metric_rows[metric].to_numpy(dtype=float)
        customdata = np.column_stack(
            [
                metric_rows["regime"].astype(str).to_numpy(),
                metric_rows["scored_values"].to_numpy(dtype=int),
            ]
        )
        if regime_group == "alarm":
            x_values = [model_label]
            mode = "markers"
        else:
            x_values = metric_rows["regime"].astype(str).tolist()
            mode = "lines+markers"
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode=mode,
                name=model_label,
                line={"color": _COMPARISON_MODEL_COLORS[0]},
                marker={"color": _COMPARISON_MODEL_COLORS[0], "size": 9},
                customdata=customdata,
                connectgaps=False,
                hovertemplate=(
                    f"Model={model_label}<br>"
                    "Regime=%{customdata[0]}<br>"
                    "Scored values=%{customdata[1]}<br>"
                    "Value=%{y:.3f}<extra>"
                    f"{_REGIME_METRIC_TITLES[metric]}</extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.update_yaxes(title_text="Metric value", row=1, col=column)
        if metric == "me":
            _add_zero_reference_line(figure, row=1, col=column)
        figure.update_xaxes(
            categoryorder="array",
            categoryarray=([model_label] if regime_group == "alarm" else regimes),
            tickangle=-20,
            row=1,
            col=column,
        )
    definition_label = (
        "Q1–Q4 water-level regimes"
        if regime_group == "quartile"
        else "Alarm-level (actual ≥ configured threshold)"
    )
    figure.update_layout(
        title=f"{model_label} sealed-test regime diagnostics — {definition_label}, aggregate",
        template="plotly_white",
        height=450,
        hovermode="closest",
    )
    return figure


def regime_horizon_figure(
    horizon_metrics: pd.DataFrame,
    *,
    regime_group: RegimeGroup,
    metric: RegimeMetric = "mae",
    model_label: str,
) -> go.Figure:
    """Plot one sealed-test error metric across forecast horizons.

    Quartile diagnostics use a 2x2 small-multiple layout, while the alarm
    diagnostic uses one separate horizon plot.  Empty cohorts retain their
    rows and therefore appear as gaps in the traces.

    Args:
        horizon_metrics: Per-horizon regime metrics with ``regime``,
            ``horizon_hours``, ``scored_values``, ``mae``, ``rmse``, and ``me``
            columns.
        regime_group: ``"quartile"`` for Q1--Q4 or ``"alarm"`` for Alarm.
        metric: Metric to plot: ``"mae"``, ``"rmse"``, or ``"me"``.
        model_label: Human-readable model name used in the figure title.

    Returns:
        A Plotly horizon figure for the selected regime group and metric.

    Raises:
        ValueError: If the group, metric, model label, or metric table is
            invalid.
    """
    _validate_model_label(model_label)
    if metric not in _REGIME_METRIC_TITLES:
        raise ValueError(f"Unknown regime metric: {metric!r}")
    rows = _validated_direct_regime_rows(
        horizon_metrics, regime_group=regime_group, scope="horizon"
    )
    regimes = _regime_group_labels(regime_group)
    metric_rows = rows.loc[rows["regime"].isin(regimes)].copy()
    if regime_group == "quartile":
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=list(regimes),
            shared_xaxes=True,
            shared_yaxes=True,
        )
        for index, regime in enumerate(regimes):
            row_number = index // 2 + 1
            column_number = index % 2 + 1
            regime_rows = metric_rows.loc[metric_rows["regime"].eq(regime)]
            _add_regime_horizon_trace(
                figure,
                regime_rows,
                metric=metric,
                model_label=model_label,
                row=row_number,
                column=column_number,
                showlegend=index == 0,
            )
            figure.update_xaxes(
                dtick=1,
                title_text="Horizon (hours)",
                row=row_number,
                col=column_number,
            )
            figure.update_yaxes(
                title_text=_REGIME_METRIC_TITLES[metric],
                row=row_number,
                col=column_number,
            )
            if metric == "me":
                _add_zero_reference_line(figure, row=row_number, col=column_number)
        figure.update_layout(
            title=(
                f"{model_label} sealed-test {metric.upper()} by horizon — "
                "Q1–Q4 small multiples"
            ),
            template="plotly_white",
            height=700,
            hovermode="x unified",
        )
        return figure

    figure = go.Figure()
    alarm_rows = metric_rows.loc[metric_rows["regime"].eq("Alarm")]
    _add_regime_horizon_trace(
        figure,
        alarm_rows,
        metric=metric,
        model_label=model_label,
        showlegend=True,
    )
    if metric == "me":
        _add_zero_reference_line(figure)
    figure.update_layout(
        title=f"{model_label} sealed-test alarm-level {metric.upper()} by horizon",
        xaxis_title="Forecast horizon (hours)",
        yaxis_title=_REGIME_METRIC_TITLES[metric],
        template="plotly_white",
        hovermode="x unified",
    )
    figure.update_xaxes(dtick=1)
    return figure


def feature_subset_best_comparison_figure(
    feature_subset_metrics: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
) -> go.Figure:
    """Compare each model's best aggregate-CV candidate per feature subset.

    Args:
        feature_subset_metrics: Candidate rows from
            :func:`src.evaluation.load_latest_complete_feature_subset_cv_metrics`.
        comparison_metrics: Canonical tidy comparison metrics from
            :func:`src.evaluation.load_latest_complete_comparison_metrics`.

    Returns:
        A four-panel comparison of CV means with fold-to-fold standard deviations.

    Raises:
        ValueError: If required candidate rows are absent, duplicated, or non-finite.
    """
    rows = _feature_subset_plot_rows(feature_subset_metrics)
    persistence_rows = _persistence_baseline_rows(comparison_metrics)
    rows = rows.loc[rows["is_best_within_subset"].eq(True)].copy()
    if rows.empty:
        raise ValueError("No best-within-subset candidate rows were found")
    if rows.duplicated(["model", "subset"]).any():
        raise ValueError("Best feature-subset rows duplicate a model/subset")

    models = rows["model"].drop_duplicates().astype(str).tolist()
    model_colors = _comparison_model_color_map(models)
    subsets = _ordered_feature_subsets(rows["subset"])
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(_COMPARISON_METRIC_TITLES.values()),
    )
    for plot_number, metric in enumerate(_COMPARISON_METRIC_TITLES, start=1):
        row = (plot_number - 1) // 2 + 1
        column = (plot_number - 1) % 2 + 1
        baseline = persistence_rows.loc[persistence_rows["metric"].eq(metric)].iloc[0]
        subset_labels = [_feature_subset_label(name) for name in subsets]
        figure.add_trace(
            go.Scatter(
                x=subset_labels,
                y=[float(baseline["value"])] * len(subsets),
                mode="lines",
                name="Persistence baseline",
                legendgroup="Persistence baseline",
                showlegend=plot_number == 1,
                error_y={
                    "type": "data",
                    "array": [float(baseline["cv_std"])] * len(subsets),
                    "visible": True,
                },
                line={
                    "color": _PERSISTENCE_BASELINE_COLOR,
                    "dash": "dash",
                    "width": 2,
                },
                hovertemplate=(
                    "Persistence baseline<br>Value=%{y:.3f}<br>"
                    f"CV standard deviation={float(baseline['cv_std']):.3f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        for model in models:
            model_rows = rows.loc[rows["model"].eq(model)].copy()
            model_rows["__subset_order"] = model_rows["subset"].map(subsets.index)
            model_rows = model_rows.sort_values("__subset_order", kind="stable")
            raw_subsets = model_rows["subset"].astype(str).tolist()
            figure.add_trace(
                go.Scatter(
                    x=[_feature_subset_label(name) for name in raw_subsets],
                    y=model_rows[f"cv_{metric}_mean"].to_numpy(dtype=float),
                    mode="lines+markers",
                    name=model,
                    legendgroup=model,
                    showlegend=plot_number == 1,
                    error_y={
                        "type": "data",
                        "array": model_rows[f"cv_{metric}_std"].to_numpy(dtype=float),
                        "visible": True,
                    },
                    line={"color": model_colors[model]},
                    marker={"color": model_colors[model], "size": 8},
                    customdata=np.column_stack(
                        [
                            raw_subsets,
                            model_rows["candidate_parameters"].astype(str),
                        ]
                    ),
                    hovertemplate=(
                        "Model=%{fullData.name}<br>Subset=%{customdata[0]}"
                        "<br>Value=%{y:.3f}<br>%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(tickangle=-20, row=row, col=column)
        figure.update_yaxes(title_text="CV metric value", row=row, col=column)
        if metric == "me":
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=row,
                col=column,
            )
    figure.update_layout(
        title="Best candidate within each feature subset — aggregate CV mean ± standard deviation",
        template="plotly_white",
        height=800,
        hovermode="closest",
    )
    return figure


def model_feature_subset_candidate_distribution_figure(
    candidate_metrics: pd.DataFrame,
    *,
    model_name: str,
    hover_columns: Sequence[str] = (),
) -> go.Figure:
    """Show one model's candidate-metric distributions by feature subset.

    Each plotted point is the validation-fold mean for one hyperparameter
    candidate. The boxes summarize the candidates within a feature subset; they
    do not represent fold-to-fold uncertainty.

    Args:
        candidate_metrics: In-memory candidate results with ``subset`` and one
            ``{metric}_mean`` column for each of MAE, RMSE, ME, and R².
        model_name: Human-readable model name used in the figure title.
        hover_columns: Additional candidate columns to include in point hover
            details, such as feature count and hyperparameters.

    Returns:
        A four-panel Plotly box-plot figure containing every candidate.

    Raises:
        ValueError: If the model name, candidate rows, subsets, metrics, or hover
            columns are invalid.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    metric_columns = [f"{metric}_mean" for metric in _COMPARISON_METRIC_TITLES]
    required_columns = ["subset", *metric_columns]
    missing_columns = sorted(set(required_columns).difference(candidate_metrics))
    if missing_columns:
        raise ValueError(f"Candidate metrics are missing columns: {missing_columns}")
    if candidate_metrics.empty:
        raise ValueError("No candidate metric rows were found")

    invalid_hover_columns = [
        column
        for column in hover_columns
        if not isinstance(column, str) or column not in candidate_metrics.columns
    ]
    if invalid_hover_columns:
        raise ValueError(
            f"Requested hover columns are invalid: {invalid_hover_columns}"
        )
    if not np.isfinite(candidate_metrics[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Candidate metrics contain non-finite metric values")

    subset_values = candidate_metrics["subset"]
    invalid_subset_values = [
        value
        for value in subset_values.drop_duplicates().tolist()
        if not isinstance(value, str) or not value or value not in _FEATURE_SUBSET_ORDER
    ]
    if subset_values.isna().any() or invalid_subset_values:
        raise ValueError(
            "Candidate subset names must be non-null established feature-subset "
            f"names; invalid values: {invalid_subset_values}"
        )

    subsets = _ordered_feature_subsets(subset_values)
    subset_colors = {
        subset: _COMPARISON_MODEL_COLORS[index % len(_COMPARISON_MODEL_COLORS)]
        for index, subset in enumerate(subsets)
    }
    subset_labels = [_feature_subset_label(subset) for subset in subsets]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{title} candidate means" for title in _COMPARISON_METRIC_TITLES.values()
        ],
    )
    for plot_number, metric in enumerate(_COMPARISON_METRIC_TITLES, start=1):
        row = (plot_number - 1) // 2 + 1
        column = (plot_number - 1) % 2 + 1
        for subset in subsets:
            subset_rows = candidate_metrics.loc[subset_values.eq(subset)]
            customdata = (
                subset_rows[list(hover_columns)].to_numpy(dtype=object)
                if hover_columns
                else None
            )
            hover_details = "".join(
                f"<br>{hover_column}=%{{customdata[{index}]}}"
                for index, hover_column in enumerate(hover_columns)
            )
            label = _feature_subset_label(subset)
            figure.add_trace(
                go.Box(
                    x=[label] * len(subset_rows),
                    y=subset_rows[f"{metric}_mean"].to_numpy(dtype=float),
                    name=label.replace("<br>", " "),
                    legendgroup=subset,
                    showlegend=plot_number == 1,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    marker={
                        "color": subset_colors[subset],
                        "size": 6,
                        "opacity": 0.7,
                    },
                    line={"color": subset_colors[subset]},
                    customdata=customdata,
                    hovertemplate=(
                        f"Model={model_name}<br>Subset={subset}"
                        "<br>Candidate CV mean=%{y:.4f}"
                        f"{hover_details}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(
            categoryorder="array",
            categoryarray=subset_labels,
            tickangle=-20,
            row=row,
            col=column,
        )
        figure.update_yaxes(title_text="Candidate CV mean", row=row, col=column)
        if metric == "me":
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=row,
                col=column,
            )
    figure.update_layout(
        title=(
            f"{model_name} performance across feature subsets — "
            "hyperparameter-candidate distributions (not fold uncertainty)"
        ),
        template="plotly_white",
        height=800,
        boxmode="group",
        hovermode="closest",
    )
    return figure


def feature_subset_candidate_distribution_figure(
    feature_subset_metrics: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
) -> go.Figure:
    """Show aggregate-CV metric distributions across every subset candidate.

    Args:
        feature_subset_metrics: Candidate rows from
            :func:`src.evaluation.load_latest_complete_feature_subset_cv_metrics`.
        comparison_metrics: Canonical tidy comparison metrics from
            :func:`src.evaluation.load_latest_complete_comparison_metrics`.

    Returns:
        A four-panel grouped box-plot comparison. Each point is one candidate's
        mean metric across validation folds.

    Raises:
        ValueError: If required candidate rows are absent or non-finite.
    """
    rows = _feature_subset_plot_rows(feature_subset_metrics)
    persistence_rows = _persistence_baseline_rows(comparison_metrics)
    models = rows["model"].drop_duplicates().astype(str).tolist()
    model_colors = _comparison_model_color_map(models)
    subsets = _ordered_feature_subsets(rows["subset"])
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(_COMPARISON_METRIC_TITLES.values()),
    )
    for plot_number, metric in enumerate(_COMPARISON_METRIC_TITLES, start=1):
        row = (plot_number - 1) // 2 + 1
        column = (plot_number - 1) % 2 + 1
        baseline = persistence_rows.loc[persistence_rows["metric"].eq(metric)].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=[_feature_subset_label(name) for name in subsets],
                y=[float(baseline["value"])] * len(subsets),
                mode="lines",
                name="Persistence baseline",
                legendgroup="Persistence baseline",
                showlegend=plot_number == 1,
                line={
                    "color": _PERSISTENCE_BASELINE_COLOR,
                    "dash": "dash",
                    "width": 2,
                },
                hovertemplate=(
                    "Persistence baseline<br>CV mean=%{y:.3f}<br>"
                    f"CV standard deviation={float(baseline['cv_std']):.3f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        for model in models:
            model_rows = rows.loc[rows["model"].eq(model)].copy()
            model_rows["__subset_order"] = model_rows["subset"].map(subsets.index)
            model_rows = model_rows.sort_values("__subset_order", kind="stable")
            raw_subsets = model_rows["subset"].astype(str).tolist()
            figure.add_trace(
                go.Box(
                    x=[_feature_subset_label(name) for name in raw_subsets],
                    y=model_rows[f"cv_{metric}_mean"].to_numpy(dtype=float),
                    name=model,
                    legendgroup=model,
                    offsetgroup=model,
                    showlegend=plot_number == 1,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    boxmean=True,
                    marker={"color": model_colors[model], "size": 4, "opacity": 0.6},
                    line={"color": model_colors[model]},
                    customdata=np.column_stack(
                        [
                            raw_subsets,
                            model_rows["candidate_parameters"].astype(str),
                        ]
                    ),
                    hovertemplate=(
                        "Model=%{fullData.name}<br>Subset=%{customdata[0]}"
                        "<br>Candidate CV mean=%{y:.3f}"
                        "<br>%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(tickangle=-20, row=row, col=column)
        figure.update_yaxes(title_text="Candidate CV mean", row=row, col=column)
        if metric == "me":
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=row,
                col=column,
            )
    figure.update_layout(
        title="All candidates by feature subset — distribution of aggregate CV means",
        template="plotly_white",
        height=800,
        boxmode="group",
        hovermode="closest",
    )
    return figure


def _feature_subset_plot_rows(feature_subset_metrics: pd.DataFrame) -> pd.DataFrame:
    """Validate and return canonical feature-subset candidate rows."""
    required_columns = {
        "model",
        "subset",
        "candidate_parameters",
        "is_best_within_subset",
        *[
            f"cv_{metric}_{statistic}"
            for metric in _COMPARISON_METRIC_TITLES
            for statistic in ("mean", "std")
        ],
    }
    missing_columns = sorted(
        required_columns.difference(feature_subset_metrics.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Feature-subset metrics are missing columns: {missing_columns}"
        )
    if feature_subset_metrics.empty:
        raise ValueError("No feature-subset candidate rows were found")
    metric_columns = [
        f"cv_{metric}_{statistic}"
        for metric in _COMPARISON_METRIC_TITLES
        for statistic in ("mean", "std")
    ]
    if not np.isfinite(
        feature_subset_metrics[metric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Feature-subset metrics contain non-finite values")
    if feature_subset_metrics[["model", "subset"]].isna().any(axis=None):
        raise ValueError("Feature-subset model and subset names must be non-null")
    return feature_subset_metrics.copy()


def _persistence_baseline_rows(comparison_metrics: pd.DataFrame) -> pd.DataFrame:
    """Validate and return aggregate cross-validation persistence rows."""
    rows = _comparison_plot_rows(
        comparison_metrics,
        phase="cross_validation",
        scope="aggregate",
    )
    rows = rows.loc[rows["model"].eq("Persistence")].copy()
    if rows.empty:
        raise ValueError("No aggregate cross-validation persistence rows were found")
    if rows["metric"].duplicated().any():
        raise ValueError("Persistence baseline rows duplicate a metric")
    missing_metrics = sorted(set(_COMPARISON_METRIC_TITLES).difference(rows["metric"]))
    if missing_metrics:
        raise ValueError(
            f"Persistence baseline rows are missing metrics: {missing_metrics}"
        )
    return rows


def _ordered_feature_subsets(subsets: pd.Series) -> list[str]:
    """Return known feature subsets first, preserving unknown subset order."""
    observed = subsets.drop_duplicates().astype(str).tolist()
    return [name for name in _FEATURE_SUBSET_ORDER if name in observed] + [
        name for name in observed if name not in _FEATURE_SUBSET_ORDER
    ]


def _feature_subset_label(subset: str) -> str:
    """Return a compact plot label for one feature-subset contract name."""
    return _FEATURE_SUBSET_LABELS.get(subset, subset.replace("_", " ").title())


def _comparison_model_color_map(models: Sequence[str]) -> dict[str, str]:
    """Assign each model one stable color within a comparison figure."""
    return {
        model: _COMPARISON_MODEL_COLORS[index % len(_COMPARISON_MODEL_COLORS)]
        for index, model in enumerate(models)
    }


def _validate_model_label(model_label: str) -> None:
    """Validate a model label used in a direct regime diagnostic."""
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be a non-empty string")


def _add_zero_reference_line(
    figure: go.Figure,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add the shared zero reference line to a metric figure."""
    line = {"color": "black", "dash": "dash", "width": 1}
    if row is None and col is None:
        figure.add_hline(y=0.0, line=line)
        return
    if row is None or col is None:
        raise ValueError("row and col must be provided together")
    figure.add_hline(y=0.0, line=line, row=row, col=col)


def _regime_group_labels(regime_group: RegimeGroup) -> tuple[str, ...]:
    """Return the required regime labels for one direct diagnostic group."""
    if regime_group == "quartile":
        return _QUARTILE_REGIME_LABELS
    if regime_group == "alarm":
        return ("Alarm",)
    raise ValueError(f"Unknown regime group: {regime_group!r}")


def _direct_regime_numeric_values(
    rows: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Convert one direct regime-table column to a float array."""
    try:
        numeric = pd.to_numeric(rows[column], errors="raise")
        return numeric.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Regime column {column!r} must contain numbers") from error


def _validated_direct_regime_rows(
    regime_metrics: pd.DataFrame,
    *,
    regime_group: RegimeGroup,
    scope: Literal["aggregate", "horizon"],
) -> pd.DataFrame:
    """Validate and canonicalize one wide per-model regime table."""
    regimes = _regime_group_labels(regime_group)
    required_columns = {"regime", "scored_values", *_REGIME_METRIC_TITLES}
    if scope == "horizon":
        required_columns.add("horizon_hours")
    missing_columns = sorted(required_columns.difference(regime_metrics.columns))
    if missing_columns:
        raise ValueError(f"Regime metrics are missing columns: {missing_columns}")

    rows = regime_metrics.loc[regime_metrics["regime"].isin(regimes)].copy()
    observed_regimes = set(rows["regime"].dropna().astype(str))
    missing_regimes = [regime for regime in regimes if regime not in observed_regimes]
    if missing_regimes:
        raise ValueError(f"Regime metrics are missing regimes: {missing_regimes}")

    duplicate_columns = ["regime"]
    if scope == "horizon":
        duplicate_columns.append("horizon_hours")
    if rows.duplicated(duplicate_columns).any():
        raise ValueError(
            "Regime metrics contain duplicate rows for " + "/".join(duplicate_columns)
        )

    counts = _direct_regime_numeric_values(rows, "scored_values")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Regime scored-value counts must be non-negative integers")
    rows["scored_values"] = counts.astype(np.int64)

    for metric in _REGIME_METRIC_TITLES:
        values = _direct_regime_numeric_values(rows, metric)
        if np.isinf(values).any() or (np.isnan(values) & (counts != 0)).any():
            raise ValueError(
                f"Regime {metric.upper()} values must be finite when scored_values > 0"
            )
        rows[metric] = values

    if scope == "aggregate":
        if "horizon_hours" in rows:
            horizons = _direct_regime_numeric_values(rows, "horizon_hours")
            if np.isfinite(horizons).any():
                raise ValueError("Aggregate regime rows must have null horizons")
        return rows

    horizons = _direct_regime_numeric_values(rows, "horizon_hours")
    if (
        not np.isfinite(horizons).all()
        or (horizons < 1).any()
        or (horizons % 1 != 0).any()
    ):
        raise ValueError("Regime horizon values must be positive integers")
    rows["horizon_hours"] = horizons.astype(np.int64)
    expected_horizons = list(range(1, int(horizons.max()) + 1))
    observed_horizons = sorted(set(rows["horizon_hours"].tolist()))
    if observed_horizons != expected_horizons:
        raise ValueError("Regime horizon values must form a contiguous sequence")
    for regime in regimes:
        regime_horizons = rows.loc[rows["regime"].eq(regime), "horizon_hours"].tolist()
        if regime_horizons != expected_horizons:
            raise ValueError(
                f"Regime {regime} rows are not in complete forecast-horizon order"
            )
    return rows


def _add_regime_horizon_trace(
    figure: go.Figure,
    rows: pd.DataFrame,
    *,
    metric: RegimeMetric,
    model_label: str,
    showlegend: bool,
    row: int | None = None,
    column: int | None = None,
) -> None:
    """Add one validated regime/horizon trace to a Plotly figure."""
    rows = rows.sort_values("horizon_hours", kind="stable")
    customdata = np.column_stack(
        [
            rows["regime"].astype(str).to_numpy(),
            rows["scored_values"].to_numpy(dtype=int),
        ]
    )
    trace = go.Scatter(
        x=rows["horizon_hours"].to_numpy(dtype=int),
        y=rows[metric].to_numpy(dtype=float),
        mode="lines+markers",
        name=model_label,
        showlegend=showlegend,
        line={"color": _COMPARISON_MODEL_COLORS[0]},
        marker={"color": _COMPARISON_MODEL_COLORS[0]},
        customdata=customdata,
        connectgaps=False,
        hovertemplate=(
            "Model=%{fullData.name}<br>Horizon=%{x} h<br>"
            "Regime=%{customdata[0]}<br>"
            "Scored values=%{customdata[1]}<br>"
            "Value=%{y:.3f}<extra></extra>"
        ),
    )
    if row is None or column is None:
        figure.add_trace(trace)
    else:
        figure.add_trace(trace, row=row, col=column)


def _comparison_plot_rows(
    comparison_metrics: pd.DataFrame,
    *,
    phase: BaseComparisonPhase,
    scope: Literal["aggregate", "horizon"],
) -> pd.DataFrame:
    """Validate and return canonical comparison rows for a figure."""
    if phase not in ("cross_validation", "sealed_test"):
        raise ValueError(f"Unknown comparison phase: {phase!r}")
    required_columns = {
        "model",
        "phase",
        "metric",
        "scope",
        "horizon",
        "value",
        "cv_std",
    }
    missing_columns = sorted(required_columns.difference(comparison_metrics.columns))
    if missing_columns:
        raise ValueError(f"Comparison metrics are missing columns: {missing_columns}")
    rows = comparison_metrics.loc[
        comparison_metrics["phase"].eq(phase)
        & comparison_metrics["scope"].eq(scope)
        & comparison_metrics["metric"].isin(_COMPARISON_METRIC_TITLES)
    ].copy()
    if rows.empty:
        raise ValueError(f"No {phase} {scope} comparison rows were found")
    missing_metrics = sorted(set(_COMPARISON_METRIC_TITLES).difference(rows["metric"]))
    if missing_metrics:
        raise ValueError(f"Comparison rows are missing metrics: {missing_metrics}")
    if not np.isfinite(rows["value"].to_numpy(dtype=float)).all():
        raise ValueError("Comparison metric values contain non-finite values")
    if (
        phase == "cross_validation"
        and not np.isfinite(rows["cv_std"].to_numpy(dtype=float)).all()
    ):
        raise ValueError(
            "Cross-validation standard deviations contain non-finite values"
        )
    if scope == "aggregate" and rows["horizon"].notna().any():
        raise ValueError("Aggregate comparison rows must have null horizons")
    if scope == "horizon" and rows["horizon"].isna().any():
        raise ValueError("Horizon comparison rows must have integer horizons")
    return rows


def _regime_plot_rows(
    comparison_metrics: pd.DataFrame,
    *,
    phase: RegimePhase,
    scope: Literal["aggregate", "horizon"],
) -> pd.DataFrame:
    """Validate and return sparse-safe sealed-test regime rows."""
    if phase not in ("sealed_test_quartile", "sealed_test_alarm"):
        raise ValueError(f"Unknown regime phase: {phase!r}")
    comparison_metrics = comparison_metrics.copy()
    for canonical, fallbacks in (
        ("lower_bound_cm", ("regime_lower_bound_cm",)),
        ("upper_bound_cm", ("regime_upper_bound_cm",)),
        (
            "scored_values",
            ("scored_value_count", "scored_forecast_values", "regime_count"),
        ),
    ):
        if canonical not in comparison_metrics:
            for fallback in fallbacks:
                if fallback in comparison_metrics:
                    comparison_metrics[canonical] = comparison_metrics[fallback]
                    break
    required_columns = {
        "model",
        "phase",
        "metric",
        "scope",
        "horizon",
        "value",
        "regime",
        "lower_bound_cm",
        "upper_bound_cm",
        "scored_values",
    }
    missing_columns = sorted(required_columns.difference(comparison_metrics.columns))
    if missing_columns:
        raise ValueError(f"Comparison metrics are missing columns: {missing_columns}")
    rows = comparison_metrics.loc[
        comparison_metrics["phase"].eq(phase)
        & comparison_metrics["scope"].eq(scope)
        & comparison_metrics["metric"].isin(_REGIME_METRIC_TITLES)
    ].copy()
    if rows.empty:
        raise ValueError(f"No {phase} {scope} comparison rows were found")
    if rows[["model", "regime", "metric"]].isna().any(axis=None):
        raise ValueError("Regime model, label, and metric values must be non-null")
    if rows.duplicated(["model", "regime", "metric", "horizon"]).any():
        raise ValueError("Regime comparison rows duplicate a model/regime/metric")
    numeric_values = (
        rows["value"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    )
    if np.isinf(numeric_values).any():
        raise ValueError("Regime metric values contain infinite values")
    counts = (
        rows["scored_values"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    )
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Regime scored-value counts must be non-negative integers")
    bounds = (
        rows[["lower_bound_cm", "upper_bound_cm"]]
        .astype("Float64")
        .to_numpy(dtype=float, na_value=np.nan)
    )
    if np.isinf(bounds).any():
        raise ValueError("Regime bounds contain infinite values")
    if scope == "aggregate" and rows["horizon"].notna().any():
        raise ValueError("Aggregate regime rows must have null horizons")
    if scope == "horizon" and rows["horizon"].isna().any():
        raise ValueError("Horizon regime rows must have integer horizons")
    rows["value"] = rows["value"].astype("Float64")
    rows["scored_values"] = rows["scored_values"].astype("Int64")
    rows["lower_bound_cm"] = rows["lower_bound_cm"].astype("Float64")
    rows["upper_bound_cm"] = rows["upper_bound_cm"].astype("Float64")
    return rows


def _ordered_regimes(regimes: pd.Series, *, phase: RegimePhase) -> list[str]:
    """Return stable labels, retaining observed non-standard regime names."""
    observed = regimes.drop_duplicates().astype(str).tolist()
    preferred = (
        list(_QUARTILE_REGIME_LABELS) if phase == "sealed_test_quartile" else ["Alarm"]
    )
    return [regime for regime in preferred if regime in observed] + [
        regime for regime in observed if regime not in preferred
    ]


def _error_boxplots_figure(
    boxplot_values: dict[str, list[np.ndarray]],
    category_labels: Sequence[str],
    *,
    title: str,
    x_axis_label: str = "Forecast horizon",
    summary_markers: dict[str, np.ndarray] | None = None,
) -> plt.Figure:
    """Plot category-wise distributions with optional summary marker series.

    Args:
        boxplot_values: Named distributions for every category.
        category_labels: Ordered labels for the x-axis categories.
        title: Figure title.
        x_axis_label: Label shown below the category axis.
        summary_markers: Optional named finite marker value per category.

    Returns:
        A boxplot figure with optional summary marker series.

    Raises:
        ValueError: If series are missing, empty, non-finite, or do not cover
            every category.
    """
    category_labels = list(category_labels)
    positions = np.arange(1, len(category_labels) + 1, dtype=float)
    if not boxplot_values:
        raise ValueError("At least one box-plot series is required")
    for label, values in boxplot_values.items():
        if len(values) != len(positions):
            raise ValueError(f"{label} does not cover every category")
        arrays = [
            np.asarray(values_for_category, dtype=float)
            for values_for_category in values
        ]
        if any(array.size == 0 or not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"{label} contains empty or non-finite values")
    if summary_markers is not None:
        for label, marker_series in summary_markers.items():
            marker_values = np.asarray(marker_series, dtype=float)
            if (
                marker_values.shape != positions.shape
                or not np.isfinite(marker_values).all()
            ):
                raise ValueError(
                    f"{label} markers do not cover every category with finite values"
                )

    figure, axis = plt.subplots(figsize=(20, 6))
    series_count = len(boxplot_values)
    group_width = 0.8
    box_width = group_width / series_count * 0.8
    offsets = (np.arange(series_count) - (series_count - 1) / 2) * (
        group_width / series_count
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for series_number, (label, values) in enumerate(boxplot_values.items()):
        color = colors[series_number % len(colors)]
        boxplot = axis.boxplot(
            values,
            positions=positions + offsets[series_number],
            widths=box_width,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.35},
            medianprops={"color": color, "linewidth": 1.5},
        )
        boxplot["boxes"][0].set_label(label)

    marker_styles = ("D", "X", "o", "P", "s")
    for marker_number, (label, marker_series) in enumerate(
        (summary_markers or {}).items()
    ):
        axis.plot(
            positions,
            np.asarray(marker_series, dtype=float),
            color=colors[(series_count + marker_number) % len(colors)],
            marker=marker_styles[marker_number % len(marker_styles)],
            linestyle="none",
            label=f"{label} summary",
        )
    axis.set_ylabel("Error")
    axis.set_xticks(positions)
    axis.set_xticklabels(category_labels)
    axis.set_xlabel(x_axis_label)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def forecast_window_figures(
    prediction_table: pd.DataFrame,
    context_series: pd.DataFrame,
    *,
    water_level_column: str,
    imputed_column: str,
    prediction_columns: Sequence[str],
    target_columns: Sequence[str],
    horizons: Sequence[int],
    label_prefix: str,
) -> dict[str, go.Figure]:
    """Build the best- and worst-RMSE forecast-window charts for one model.

    Scores every issue time by the RMSE across all forecast horizons, keeps the
    issue times whose target-station context is complete and non-imputed from
    ``_CONTEXT_WINDOW_HOURS`` before through ``_CONTEXT_WINDOW_HOURS`` after, then
    charts the lowest- and highest-RMSE eligible issue.

    Args:
        prediction_table: Per-issue frame with ``issue_time``, actual targets,
            and correspondingly ordered prediction columns.
        context_series: Timestamp-indexed target-station water-level and
            imputation frame.
        water_level_column: Column holding the target station's water level.
        imputed_column: Column marking imputed target-station rows.
        prediction_columns: Ordered direct-forecast prediction columns.
        target_columns: Ordered actual target columns.
        horizons: Ordered forecast horizons, in hours.
        label_prefix: Model label used in both chart titles.

    Returns:
        The ``"best"`` and ``"worst"`` forecast-window figures.

    Raises:
        ValueError: If per-issue RMSE is non-finite, or fewer than two issue
            times have a complete, non-imputed context window.
    """
    prediction_columns = list(prediction_columns)
    windows = prediction_table.copy()
    windows["issue_time"] = pd.to_datetime(windows["issue_time"], utc=True)
    windows["issue_rmse"] = np.sqrt(
        np.mean(
            (
                windows[prediction_columns].to_numpy(dtype=float)
                - windows[list(target_columns)].to_numpy(dtype=float)
            )
            ** 2,
            axis=1,
        )
    )
    if not np.isfinite(windows["issue_rmse"]).all():
        raise ValueError("Per-issue forecast-window RMSE contains non-finite values")

    context_by_row = {
        row_index: context
        for row_index, window_row in windows.iterrows()
        if (
            context := _complete_context(
                window_row["issue_time"],
                context_series,
                water_level_column=water_level_column,
                imputed_column=imputed_column,
            )
        )
        is not None
    }
    if len(context_by_row) < 2:
        raise ValueError(
            "Forecast-window plots require at least two issue timestamps with "
            f"complete, non-imputed target-station context from "
            f"-{_CONTEXT_WINDOW_HOURS}h through +{_CONTEXT_WINDOW_HOURS}h; "
            f"found {len(context_by_row)}."
        )

    eligible_windows = windows.loc[list(context_by_row)]
    best_window_row = eligible_windows.sort_values(
        ["issue_rmse", "issue_time"], kind="stable"
    ).iloc[0]
    worst_window_row = eligible_windows.sort_values(
        ["issue_rmse", "issue_time"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return {
        name: _forecast_window_figure(
            window_row,
            context_by_row[window_row.name],
            f"{quality}-RMSE {label_prefix}",
            prediction_table=windows,
            context_by_row=context_by_row,
            water_level_column=water_level_column,
            prediction_columns=prediction_columns,
            horizons=horizons,
        )
        for name, quality, window_row in (
            ("best", "Best", best_window_row),
            ("worst", "Worst", worst_window_row),
        )
    }


def _complete_context(
    issue_time: pd.Timestamp,
    context_series: pd.DataFrame,
    *,
    water_level_column: str,
    imputed_column: str,
) -> pd.DataFrame | None:
    """Return a complete, non-imputed context window if available.

    Args:
        issue_time: Forecast issue timestamp to center the context window on.
        context_series: Timestamp-indexed water-level and imputation frame.
        water_level_column: Column holding the target station's water level.
        imputed_column: Column marking imputed target-station rows.

    Returns:
        The reindexed context window, or ``None`` if any hour is missing or
        imputed.
    """
    issue_time = pd.Timestamp(issue_time)
    if issue_time.tz is None:
        issue_time = issue_time.tz_localize("UTC")
    else:
        issue_time = issue_time.tz_convert("UTC")
    expected_times = pd.date_range(
        issue_time - pd.to_timedelta(_CONTEXT_WINDOW_HOURS, unit="h"),
        issue_time + pd.to_timedelta(_CONTEXT_WINDOW_HOURS, unit="h"),
        freq="h",
    )
    context = context_series.reindex(expected_times)
    if (
        len(context) != 2 * _CONTEXT_WINDOW_HOURS + 1
        or not context[water_level_column].notna().all()
        or not context[imputed_column].eq(False).all()
    ):
        return None
    return context


def _forecast_window_figure(
    window_row: pd.Series,
    context: pd.DataFrame,
    label: str,
    *,
    prediction_table: pd.DataFrame,
    context_by_row: dict,
    water_level_column: str,
    prediction_columns: Sequence[str],
    horizons: Sequence[int],
) -> go.Figure:
    """Build one issue-time forecast chart with a neighbouring-issue slider.

    The selected issue's ground-truth context remains fixed while the slider
    updates the prediction trace and issue-time annotations.

    Args:
        window_row: Prediction-table row the chart is centered on.
        context: Complete ground-truth context for the issue.
        label: Complete chart label to prefix the title with.
        prediction_table: Full per-issue prediction table.
        context_by_row: Prediction-table indices with a complete context.
        water_level_column: Column holding the target station's water level.
        prediction_columns: Ordered direct-forecast prediction columns.
        horizons: Ordered forecast horizons, in hours.

    Returns:
        A Plotly figure with an animated issue-time slider.
    """
    slider_rows = _forecast_window_slider_rows(
        window_row, prediction_table, context_by_row
    )
    slider_row_indices = slider_rows.index.tolist()
    active_slider_index = slider_row_indices.index(window_row.name)
    frames = []
    slider_steps = []
    for row_index, slider_row in slider_rows.iterrows():
        frame_name = f"forecast-window-{row_index}"
        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    _forecast_window_prediction_trace(
                        slider_row, prediction_columns, horizons
                    )
                ],
                traces=[1],
                layout=_forecast_window_layout(slider_row, label),
            )
        )
        slider_steps.append(
            {
                "label": pd.Timestamp(slider_row["issue_time"]).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )
    return go.Figure(
        data=[
            go.Scatter(
                x=context.index,
                y=context[water_level_column].to_numpy(dtype=float),
                mode="lines+markers",
                name="Ground truth",
                hovertemplate=(
                    "Valid time=%{x}<br>Ground truth=%{y:.3f}<extra></extra>"
                ),
            ),
            _forecast_window_prediction_trace(window_row, prediction_columns, horizons),
        ],
        frames=frames,
        layout={
            **_forecast_window_layout(window_row, label),
            "sliders": [
                {
                    "active": active_slider_index,
                    "y": -0.28,
                    "pad": {"t": 12},
                    "currentvalue": {"prefix": "Issue time: "},
                    "steps": slider_steps,
                }
            ],
        },
    )


def _forecast_window_slider_rows(
    window_row: pd.Series,
    prediction_table: pd.DataFrame,
    context_by_row: dict,
) -> pd.DataFrame:
    """Return eligible issue times reachable from one chart's slider.

    Args:
        window_row: Prediction-table row the chart is centered on.
        prediction_table: Full per-issue prediction table.
        context_by_row: Prediction-table indices with a complete context.

    Returns:
        Prediction-table rows within the slider window, sorted by issue time.
    """
    issue_time = pd.Timestamp(window_row["issue_time"])
    slider_span = pd.to_timedelta(_SLIDER_WINDOW_HOURS, unit="h")
    return (
        prediction_table.loc[list(context_by_row)]
        .loc[
            lambda rows: rows["issue_time"].between(
                issue_time - slider_span, issue_time + slider_span
            )
        ]
        .sort_values("issue_time", kind="stable")
    )


def _forecast_window_prediction_trace(
    window_row: pd.Series,
    prediction_columns: Sequence[str],
    horizons: Sequence[int],
) -> go.Scatter:
    """Build the issue-specific direct-forecast prediction trace.

    Args:
        window_row: Prediction-table row to plot.
        prediction_columns: Ordered direct-forecast prediction columns.
        horizons: Ordered forecast horizons, in hours.

    Returns:
        A Plotly scatter trace of predictions across the forecast horizon.
    """
    issue_time = pd.Timestamp(window_row["issue_time"])
    prediction_times = issue_time + pd.to_timedelta(list(horizons), unit="h")
    return go.Scatter(
        x=prediction_times,
        y=window_row[list(prediction_columns)].to_numpy(dtype=float),
        mode="lines+markers",
        name="Prediction",
        hovertemplate="Valid time=%{x}<br>Prediction=%{y:.3f}<extra></extra>",
    )


def _forecast_window_layout(window_row: pd.Series, label: str) -> dict:
    """Build the issue-specific title and marker annotation.

    Args:
        window_row: Prediction-table row the chart is centered on.
        label: Complete chart label to prefix the title with.

    Returns:
        A Plotly layout dict with the issue-time title, marker line, and
        annotation.
    """
    issue_time = pd.Timestamp(window_row["issue_time"])
    return {
        "title": (
            f"{label} forecast window<br>"
            f"Issue time: {issue_time.isoformat()} | "
            f"24-hour RMSE: {window_row['issue_rmse']:.4f}"
        ),
        "xaxis_title": "Valid time",
        "yaxis": {"title": "Water level", "separatethousands": True},
        "hovermode": "x unified",
        "margin": {"b": 160},
        "shapes": [
            {
                "type": "line",
                "x0": issue_time,
                "x1": issue_time,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"dash": "dash", "color": "black"},
            }
        ],
        "annotations": [
            {
                "x": issue_time,
                "y": 1,
                "yref": "paper",
                "text": "Issue time",
                "showarrow": True,
                "arrowhead": 2,
            }
        ],
    }
