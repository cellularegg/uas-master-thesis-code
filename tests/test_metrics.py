import numpy as np
import pandas as pd
import pytest

from src.metrics import metric_tables, water_level_regime_tables


def test_metric_tables_calculates_aggregate_and_horizon_metrics():
    actual = pd.DataFrame({"target_01": [1.0, 3.0], "target_02": [2.0, 4.0]})
    predictions = np.array([[2.0, 1.0], [5.0, 1.0]])

    aggregate, per_horizon = metric_tables(
        actual,
        predictions,
        target_columns=["target_01", "target_02"],
        station_id="station-1",
    )

    assert aggregate.loc[0, "station_id"] == "station-1"
    assert aggregate.loc[0, "scored_issue_times"] == 2
    assert aggregate.loc[0, "scored_values"] == 4
    assert aggregate.loc[0, "mae"] == pytest.approx(1.75)
    assert aggregate.loc[0, "rmse"] == pytest.approx(np.sqrt(3.75))
    assert aggregate.loc[0, "me"] == pytest.approx(-0.25)
    assert aggregate.loc[0, "r2"] == pytest.approx(-2.0)
    assert per_horizon["target"].tolist() == ["target_01", "target_02"]
    assert per_horizon["mae"].tolist() == pytest.approx([1.5, 2.0])
    assert per_horizon["me"].tolist() == pytest.approx([1.5, -2.0])
    assert per_horizon["r2"].tolist() == pytest.approx([-1.5, -4.0])


def test_metric_tables_rejects_prediction_shape_mismatch():
    actual = pd.DataFrame({"target_01": [1.0, 3.0]})

    with pytest.raises(ValueError, match="Prediction shape"):
        metric_tables(
            actual,
            np.ones((2, 2)),
            target_columns=["target_01"],
            station_id="station-1",
        )


def test_water_level_regime_tables_assigns_boundaries_and_overlapping_alarm():
    actual = pd.DataFrame(
        {
            "target_01": [10.0, 20.0, 30.0, 25.0, 40.0],
            "target_02": [10.0, 20.0, 30.0, 25.0, 40.0],
        }
    )
    predictions = actual.to_numpy() + 1.0

    aggregate, per_horizon = water_level_regime_tables(
        actual,
        predictions,
        target_columns=["target_01", "target_02"],
        station_id="station-1",
        quartile_cutoffs_cm=(10.0, 20.0, 30.0),
        alarm_threshold_cm=25.0,
    )

    assert aggregate["regime"].tolist() == ["Q1", "Q2", "Q3", "Q4", "Alarm"]
    assert aggregate["scored_values"].tolist() == [2, 2, 4, 2, 6]
    assert per_horizon["scored_values"].tolist() == [
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        3,
        3,
    ]
    assert aggregate["horizon_hours"].isna().all()
    assert per_horizon["horizon_hours"].tolist() == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    assert (
        aggregate.loc[aggregate["regime"].eq("Alarm"), "lower_bound_cm"].iloc[0] == 25.0
    )
    assert aggregate.loc[aggregate["regime"].eq("Q1"), "upper_bound_cm"].iloc[0] == 10.0
    assert aggregate["mae"].tolist() == pytest.approx([1.0] * 5)
    assert aggregate["rmse"].tolist() == pytest.approx([1.0] * 5)
    assert aggregate["me"].tolist() == pytest.approx([1.0] * 5)


def test_water_level_regime_tables_retains_empty_groups_with_null_metrics():
    actual = pd.DataFrame({"target_01": [1.0, 2.0]})

    aggregate, per_horizon = water_level_regime_tables(
        actual,
        np.array([[1.5], [2.5]]),
        target_columns=["target_01"],
        station_id="station-1",
        quartile_cutoffs_cm=(10.0, 20.0, 30.0),
    )

    assert aggregate["scored_values"].tolist() == [2, 0, 0, 0, 0]
    assert per_horizon["scored_values"].tolist() == [2, 0, 0, 0, 0]
    assert (
        aggregate.loc[aggregate["regime"].ne("Q1"), ["mae", "rmse", "me"]]
        .isna()
        .all()
        .all()
    )


def test_water_level_regime_tables_includes_exact_default_alarm_boundary():
    actual = pd.DataFrame({"target_01": [544.999, 545.0]})

    aggregate, _ = water_level_regime_tables(
        actual,
        actual.to_numpy(),
        target_columns=["target_01"],
        station_id="207241-at",
        quartile_cutoffs_cm=(100.0, 200.0, 300.0),
    )

    alarm = aggregate.loc[aggregate["regime"].eq("Alarm")].iloc[0]
    assert alarm["lower_bound_cm"] == 545.0
    assert alarm["scored_values"] == 1
    assert aggregate.loc[aggregate["regime"].eq("Q4"), "scored_values"].iloc[0] == 2


@pytest.mark.parametrize("cutoffs", [(20.0, 10.0, 30.0), (10.0, 30.0, 20.0)])
def test_water_level_regime_tables_rejects_unordered_cutoffs(cutoffs):
    with pytest.raises(ValueError, match="ordered"):
        water_level_regime_tables(
            pd.DataFrame({"target_01": [1.0]}),
            np.array([[1.0]]),
            target_columns=["target_01"],
            station_id="station-1",
            quartile_cutoffs_cm=cutoffs,
        )
