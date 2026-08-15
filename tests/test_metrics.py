import numpy as np
import pandas as pd
import pytest

from src.metrics import metric_tables


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
