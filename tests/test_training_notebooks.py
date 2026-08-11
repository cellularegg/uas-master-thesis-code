import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID

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
    metadata_path.write_text(
        json.dumps(
            {
                "engineered_station_ids": [TARGET_STATION_ID],
                "configuration": {"horizon_hours": FORECAST_HORIZON_HOURS},
                "predictor_columns": [
                    f"{station_prefix}{column}"
                    for column in (
                        "water_level",
                        "imputed",
                        "precipitation",
                        "temperature_2m",
                    )
                ],
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


def test_ridge_notebook_uses_the_prefixed_direct_multi_horizon_contract(
    tmp_path: Path,
) -> None:
    namespace = _ridge_contract_namespace(tmp_path)
    station_prefix = f"{TARGET_STATION_ID}__"

    expected_targets = [
        f"{station_prefix}target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    assert namespace["TARGET_COLUMNS"] == expected_targets
    assert len(namespace["TARGET_COLUMNS"]) == 24
    assert f"{station_prefix}water_level" in namespace["FEATURE_COLUMNS"]

    rows = 3
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            namespace["TARGET_VALID_COLUMN"]: [True, False, True],
        }
    )
    for column in namespace["FEATURE_COLUMNS"]:
        frame[column] = False if column.endswith("__imputed") else 1.0
    for target_number, column in enumerate(namespace["TARGET_COLUMNS"], start=1):
        frame[column] = [target_number, np.nan, target_number + 10.0]

    eligible = namespace["eligible_rows"](
        frame, station_id=TARGET_STATION_ID, artifact_name="synthetic"
    )

    assert eligible.tolist() == [True, False, True]
    assert frame.loc[eligible, namespace["TARGET_COLUMNS"]].shape == (2, 24)
