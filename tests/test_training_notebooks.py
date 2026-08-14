import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID

NOTEBOOK_PATH = Path("04_train_ridge.ipynb")


def _cell_source(notebook: dict, cell_id: str) -> str:
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            source = cell["source"]
            return "".join(source) if isinstance(source, list) else source
    raise AssertionError(f"Notebook cell {cell_id!r} not found")


def test_ridge_notebook_wires_shared_loading_subsets_and_cohorts(
    tmp_path: Path,
) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    processed_dir = tmp_path / "joined"
    processed_dir.mkdir()
    metadata_path = processed_dir / "all_stations_feature_metadata.json"
    train_path = processed_dir / "all_stations_train_features.parquet"
    test_path = processed_dir / "all_stations_test_features.parquet"
    station_id = TARGET_STATION_ID
    target_columns = [
        f"{station_id}__target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    predictor_columns = [
        f"{station_id}__water_level",
        f"{station_id}__imputed",
        f"{station_id}__precipitation",
        f"{station_id}__temperature_2m",
        f"{station_id}__water_level_lag_24h",
        f"{station_id}__imputed_count_24h",
        f"{station_id}__utc_hour_sin",
        "station-b__water_level",
        "station-b__imputed",
        "station-b__precipitation",
        "station-b__temperature_2m",
    ]
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": [station_id],
                "configuration": {"horizon_hours": FORECAST_HORIZON_HOURS},
                "predictor_columns": predictor_columns,
                "target_columns": target_columns,
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"], utc=True
            ),
            f"{station_id}__target_valid": [True, True, False],
        }
    )
    for column in predictor_columns:
        frame[column] = False if column.endswith("__imputed") else 1.0
    for target_number, column in enumerate(target_columns, start=1):
        frame[column] = np.arange(len(frame), dtype=float) + target_number
    frame.loc[2, target_columns] = np.nan
    frame.to_parquet(train_path, index=False)
    frame.iloc[[1]].to_parquet(test_path, index=False)

    setup_source = _cell_source(notebook, "3").replace(
        'PROCESSED_DIR = Path("data/processed/joined")',
        f"PROCESSED_DIR = Path({str(processed_dir)!r})",
    )
    namespace: dict = {}
    exec(setup_source, namespace)  # noqa: S102
    exec(_cell_source(notebook, "10"), namespace)  # noqa: S102
    exec(_cell_source(notebook, "12"), namespace)  # noqa: S102

    assert namespace["contract"].station_id == station_id
    assert namespace["FULL_FEATURE_COLUMNS"] == predictor_columns
    assert list(namespace["FEATURE_SUBSETS"]) == [
        "full",
        "all_station_hydrology_quality_time",
        "raw_all_stations",
        "target_station_full",
        "target_station_hydrology_quality_time",
        "current_water_levels_all_stations",
    ]
    assert namespace["train_rows"]["timestamp"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)
    )
    assert len(namespace["test_rows"]) == 1
    assert namespace["INPUT_PARQUET_SHA256_PARAMS"] == {
        "train_input_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_input_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }
    assert "absolute_error_boxplot_payload(" in _cell_source(notebook, "18")


def test_ridge_notebook_logs_input_hashes_on_every_mlflow_run() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cv_source = _cell_source(notebook, "14")
    test_source = _cell_source(notebook, "18")

    assert cv_source.count("**INPUT_PARQUET_SHA256_PARAMS") == 2
    assert test_source.count("**INPUT_PARQUET_SHA256_PARAMS") == 1


def test_ridge_candidate_selection_applies_all_tie_breakers() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict = {"pd": pd}
    exec(_cell_source(notebook, "6"), namespace)  # noqa: S102
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
