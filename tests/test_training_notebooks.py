import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID
from src.feature_engineering import feature_column_names

NOTEBOOK_PATH = Path("04_train_ridge.ipynb")


def _cell_source(notebook: dict, cell_id: str) -> str:
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise AssertionError(f"Notebook cell {cell_id!r} not found")


def _ridge_contract_namespace(tmp_path: Path) -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    metadata_path = tmp_path / "all_stations_feature_metadata.json"
    station_prefix = f"{TARGET_STATION_ID}__"
    target_columns = [
        f"{station_prefix}target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    raw_columns = ("water_level", "imputed", "precipitation", "temperature_2m")
    neighbor_station_ids = [
        "207019-at",
        "207027-at",
        "207340-at",
        "207068-at",
        "Ennshafen1.Rivermeter-at",
        "207084-at",
        "207357-at",
    ]
    predictor_columns = [
        f"{station_prefix}{column}" for column in feature_column_names()
    ] + [
        f"{neighbor_station_id}__{column}"
        for neighbor_station_id in neighbor_station_ids
        for column in raw_columns
    ]
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": [TARGET_STATION_ID],
                "configuration": {"horizon_hours": FORECAST_HORIZON_HOURS},
                "predictor_columns": predictor_columns,
                "target_columns": target_columns,
            }
        ),
        encoding="utf-8",
    )

    setup_source = _cell_source(notebook, "3").replace(
        'PROCESSED_DIR = Path("data/processed/joined")',
        f"PROCESSED_DIR = Path({str(tmp_path)!r})",
    )
    namespace: dict = {}
    exec(setup_source, namespace)  # noqa: S102
    exec(_cell_source(notebook, "6"), namespace)  # noqa: S102
    return namespace


def test_ridge_notebook_uses_all_metadata_predictors_and_direct_targets(
    tmp_path: Path,
) -> None:
    namespace = _ridge_contract_namespace(tmp_path)
    station_prefix = f"{TARGET_STATION_ID}__"
    expected_predictors = [
        f"{station_prefix}{column}" for column in feature_column_names()
    ] + [
        f"{station_id}__{column}"
        for station_id in (
            "207019-at",
            "207027-at",
            "207340-at",
            "207068-at",
            "Ennshafen1.Rivermeter-at",
            "207084-at",
            "207357-at",
        )
        for column in ("water_level", "imputed", "precipitation", "temperature_2m")
    ]

    expected_targets = [
        f"{station_prefix}target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    assert namespace["TARGET_COLUMNS"] == expected_targets
    assert len(namespace["TARGET_COLUMNS"]) == 24
    assert f"{station_prefix}water_level" in namespace["FEATURE_COLUMNS"]
    assert "207357-at__water_level" in namespace["FEATURE_COLUMNS"]
    assert namespace["FEATURE_COLUMNS"] == expected_predictors
    assert namespace["FULL_FEATURE_COLUMNS"] == expected_predictors
    subsets = namespace["FEATURE_SUBSETS"]
    assert list(subsets) == [
        "full",
        "all_station_hydrology_quality_time",
        "raw_all_stations",
        "target_station_full",
        "target_station_hydrology_quality_time",
        "current_water_levels_all_stations",
    ]
    assert [len(columns) for columns in subsets.values()] == [81, 55, 32, 53, 41, 8]
    assert len({tuple(columns) for columns in subsets.values()}) == 6
    for columns in subsets.values():
        assert len(columns) == len(set(columns))
        assert columns == [
            column for column in expected_predictors if column in columns
        ]

    target_full = subsets["target_station_full"]
    assert target_full == expected_predictors[:53]
    assert all(column.startswith(station_prefix) for column in target_full)
    assert all(
        column.rsplit("__", maxsplit=1)[-1] not in {"precipitation", "temperature_2m"}
        for column in subsets["target_station_hydrology_quality_time"]
    )
    assert any(
        column.endswith("water_level_lag_168h")
        for column in subsets["target_station_hydrology_quality_time"]
    )
    assert any(
        column.endswith("utc_hour_sin")
        for column in subsets["target_station_hydrology_quality_time"]
    )
    assert all(
        column.rsplit("__", maxsplit=1)[-1] in {"water_level", "imputed"}
        or column.rsplit("__", maxsplit=1)[-1].startswith("water_level_")
        or column.rsplit("__", maxsplit=1)[-1].startswith("imputed_count_")
        or column.rsplit("__", maxsplit=1)[-1].startswith("utc_")
        for column in subsets["all_station_hydrology_quality_time"]
    )
    assert all(
        column.rsplit("__", maxsplit=1)[-1]
        in {"water_level", "imputed", "precipitation", "temperature_2m"}
        for column in subsets["raw_all_stations"]
    )
    assert all(
        column.rsplit("__", maxsplit=1)[-1] == "water_level"
        for column in subsets["current_water_levels_all_stations"]
    )
    assert not any(
        column.rsplit("__", maxsplit=1)[-1]
        in {"timestamp", "station_id", "target_valid"}
        for column in namespace["FEATURE_COLUMNS"]
    )

    rows = 4
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            namespace["TARGET_VALID_COLUMN"]: [True, True, False, True],
        }
    )
    for column in namespace["FEATURE_COLUMNS"]:
        frame[column] = False if column.endswith("__imputed") else 1.0
    for target_number, column in enumerate(namespace["TARGET_COLUMNS"], start=1):
        frame[column] = np.arange(rows, dtype=float) + target_number
    frame.loc[2, namespace["TARGET_COLUMNS"]] = np.nan
    frame.loc[1, "207357-at__water_level"] = np.nan

    eligible = namespace["eligible_rows"](
        frame, station_id=TARGET_STATION_ID, artifact_name="synthetic"
    )

    assert eligible.tolist() == [True, False, False, True]
    assert frame.loc[eligible, namespace["TARGET_COLUMNS"]].shape == (2, 24)


def test_ridge_eligibility_uses_full_contract_for_every_candidate(
    tmp_path: Path,
) -> None:
    namespace = _ridge_contract_namespace(tmp_path)
    target_columns = namespace["TARGET_COLUMNS"]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            namespace["TARGET_VALID_COLUMN"]: [True, True],
        }
    )
    for column in namespace["FULL_FEATURE_COLUMNS"]:
        frame[column] = False if column.endswith("__imputed") else 1.0
    frame[target_columns] = 1.0

    weather_column = f"{TARGET_STATION_ID}__precipitation"
    assert (
        weather_column
        not in namespace["FEATURE_SUBSETS"]["target_station_hydrology_quality_time"]
    )
    frame.loc[1, weather_column] = np.nan

    eligible = namespace["eligible_rows"](
        frame, station_id=TARGET_STATION_ID, artifact_name="synthetic"
    )

    assert eligible.tolist() == [True, False]


def test_ridge_candidate_selection_applies_all_tie_breakers(tmp_path: Path) -> None:
    namespace = _ridge_contract_namespace(tmp_path)
    select_candidate = namespace["select_candidate"]

    assert select_candidate(
        pd.DataFrame(
            [
                {"subset": "Wide", "alpha": 0.1, "feature_count": 10, "mae_mean": 2.0},
                {"subset": "Lean", "alpha": 0.1, "feature_count": 2, "mae_mean": 1.0},
            ]
        )
    ) == ("Lean", 0.1)
    assert select_candidate(
        pd.DataFrame(
            [
                {"subset": "Wide", "alpha": 0.01, "feature_count": 10, "mae_mean": 1.0},
                {"subset": "Lean", "alpha": 0.1, "feature_count": 2, "mae_mean": 1.0},
            ]
        )
    ) == ("Lean", 0.1)
    assert select_candidate(
        pd.DataFrame(
            [
                {"subset": "Same", "alpha": 1.0, "feature_count": 2, "mae_mean": 1.0},
                {"subset": "Same", "alpha": 0.1, "feature_count": 2, "mae_mean": 1.0},
            ]
        )
    ) == ("Same", 0.1)
    assert select_candidate(
        pd.DataFrame(
            [
                {"subset": "Zulu", "alpha": 0.1, "feature_count": 2, "mae_mean": 1.0},
                {"subset": "Alpha", "alpha": 0.1, "feature_count": 2, "mae_mean": 1.0},
            ]
        )
    ) == ("Alpha", 0.1)


def test_ridge_notebook_builds_horizon_wise_test_error_boxplots(
    tmp_path: Path,
) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = _ridge_contract_namespace(tmp_path)
    exec(_cell_source(notebook, "8"), namespace)  # noqa: S102

    target_columns = namespace["TARGET_COLUMNS"]
    test_issue_times = 5
    actual_values = np.arange(
        test_issue_times * len(target_columns), dtype=float
    ).reshape(test_issue_times, len(target_columns))
    actual = pd.DataFrame(actual_values, columns=target_columns)
    errors = np.arange(1, actual_values.size + 1, dtype=float).reshape(
        actual_values.shape
    )
    predictions = actual_values - errors
    _, per_horizon_metrics = namespace["metric_tables"](
        actual,
        predictions,
        target_columns=target_columns,
        station_id=TARGET_STATION_ID,
    )

    boxplot_values, category_labels, summary_markers = namespace[
        "horizon_error_boxplot_payload"
    ](actual, predictions, per_horizon_metrics, target_columns)

    assert category_labels == [f"H+{horizon:02d}" for horizon in range(1, 25)]
    assert set(boxplot_values) == {"MAE", "RMSE"}
    assert all(len(values) == 24 for values in boxplot_values.values())
    assert all(
        values_for_horizon.shape == (test_issue_times,)
        for values in boxplot_values.values()
        for values_for_horizon in values
    )
    for horizon in range(24):
        np.testing.assert_array_equal(
            boxplot_values["MAE"][horizon], errors[:, horizon]
        )
        np.testing.assert_array_equal(
            boxplot_values["RMSE"][horizon], errors[:, horizon]
        )
    np.testing.assert_allclose(summary_markers["MAE"], per_horizon_metrics["mae"])
    np.testing.assert_allclose(summary_markers["RMSE"], per_horizon_metrics["rmse"])

    figure = namespace["error_boxplots_figure"](
        boxplot_values,
        category_labels,
        title="synthetic test errors",
        summary_markers=summary_markers,
    )
    try:
        assert len(figure.axes) == 1
    finally:
        namespace["plt"].close(figure)

    evaluation_source = _cell_source(notebook, "18")
    assert "horizon_error_boxplot_payload(" in evaluation_source
    assert '"All 24 horizons"' not in evaluation_source
