from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.split_folds import (
    SplitConfig,
    calculate_target_eligibility,
    split_station_frame,
    write_split_artifacts,
)


def _station_frame(rows: int, *, station_id: str = "station-at") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "water_level": range(rows),
            "imputed": False,
            "station_id": station_id,
        }
    )


def test_split_reconstructs_source_with_exact_chronological_partition() -> None:
    source = _station_frame(503).sample(frac=1, random_state=42)

    result = split_station_frame(source, station_id="station-at")

    assert len(result.train) == 402
    assert len(result.test) == 101
    assert result.train["timestamp"].max() < result.test["timestamp"].min()
    assert not result.train["target_valid"].iloc[-24:].any()
    assert not result.test["target_valid"].iloc[-24:].any()
    reconstructed = pd.concat(
        [
            result.train.drop(columns=["target_valid", *result.role_columns]),
            result.test.drop(columns="target_valid"),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        reconstructed,
        source.sort_values("timestamp").reset_index(drop=True),
    )


def test_split_builds_five_expanding_folds_with_deterministic_rounding() -> None:
    result = split_station_frame(_station_frame(503), station_id="station-at")

    assert result.validation_block_sizes == (41, 40, 40, 40, 40)
    assert result.role_columns == tuple(
        f"fold_{number:02d}_role" for number in range(1, 6)
    )

    previous_train_count = 0
    for fold_number, block_size in enumerate(result.validation_block_sizes, start=1):
        roles = result.train[f"fold_{fold_number:02d}_role"]
        counts = roles.value_counts()
        assert counts["embargo"] == 24
        assert counts["validation"] == block_size - 24
        assert counts["train"] > previous_train_count
        previous_train_count = counts["train"]


def test_split_carries_its_configuration_into_artifact_metadata(tmp_path: Path) -> None:
    station_id = "station-at"
    source_path = tmp_path / f"{station_id}_hourly.parquet"
    source = _station_frame(503)
    source.to_parquet(source_path, index=False)
    config = SplitConfig(
        test_fraction=0.25,
        n_folds=3,
        initial_train_fraction=0.55,
        horizon_hours=12,
    )

    split = split_station_frame(source, station_id=station_id, config=config)
    metadata = write_split_artifacts(
        split,
        station_id=station_id,
        source_path=source_path,
        output_dir=tmp_path / "processed",
    )

    assert split.config is config
    assert metadata["configuration"] == {
        "forecast_horizon_hours": 12,
        "initial_train_fraction_of_development": 0.55,
        "n_folds": 3,
        "test_fraction": 0.25,
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SplitConfig(test_fraction=1.0),
        lambda: SplitConfig(n_folds=0),
        lambda: SplitConfig(initial_train_fraction=0.0),
        lambda: SplitConfig(horizon_hours=0),
    ],
)
def test_split_config_rejects_values_that_cannot_define_folds(
    factory: Callable[[], SplitConfig],
) -> None:
    with pytest.raises(ValueError):
        factory()


def test_validation_target_periods_are_contained_and_disjoint() -> None:
    result = split_station_frame(_station_frame(503), station_id="station-at")
    target_periods: list[set[int]] = []

    for role_column in result.role_columns:
        validation_positions = np.flatnonzero(result.train[role_column] == "validation")
        target_times = {
            int(issue_position) + offset
            for issue_position in validation_positions
            for offset in range(1, 25)
        }
        target_periods.append(target_times)

    for index, target_period in enumerate(target_periods):
        assert all(
            target_period.isdisjoint(other) for other in target_periods[index + 1 :]
        )


def test_target_eligibility_requires_24_observed_non_imputed_future_values() -> None:
    frame = _station_frame(80)
    frame.loc[10, "water_level"] = np.nan
    frame.loc[30, "imputed"] = True

    eligible = calculate_target_eligibility(frame)

    assert not eligible.iloc[:10].any()
    assert not eligible.iloc[6:30].any()
    assert eligible.iloc[30]
    assert eligible.iloc[40:56].all()
    assert not eligible.iloc[-24:].any()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="water_level"), "required columns"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "unique",
        ),
        (lambda frame: frame.drop(index=10), "contiguous hourly"),
        (
            lambda frame: frame.assign(
                timestamp=frame["timestamp"].dt.tz_localize(None)
            ),
            "UTC",
        ),
        (
            lambda frame: frame.assign(
                station_id=[
                    "other" if index == 0 else "station-at" for index in frame.index
                ]
            ),
            "one station",
        ),
    ],
)
def test_split_rejects_invalid_station_timelines(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    invalid = mutate(_station_frame(503))

    with pytest.raises(ValueError, match=message):
        split_station_frame(invalid, station_id="station-at")


def test_split_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="insufficient history"):
        split_station_frame(_station_frame(200), station_id="station-at")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_write_artifacts_records_schemas_metadata_and_all_file_hashes(
    tmp_path: Path,
) -> None:
    station_id = "station-at"
    source_path = tmp_path / f"{station_id}_hourly.parquet"
    output_dir = tmp_path / "processed"
    source = _station_frame(503)
    source.to_parquet(source_path, index=False)
    split = split_station_frame(source, station_id=station_id)

    metadata = write_split_artifacts(
        split,
        station_id=station_id,
        source_path=source_path,
        output_dir=output_dir,
    )

    train_path = output_dir / f"{station_id}_train_splits.parquet"
    test_path = output_dir / f"{station_id}_test.parquet"
    metadata_path = output_dir / f"{station_id}_split_metadata.json"
    persisted_train = pd.read_parquet(train_path)
    persisted_test = pd.read_parquet(test_path)
    persisted_metadata = json.loads(metadata_path.read_text())

    assert persisted_metadata == metadata
    assert metadata["schema_version"] == "1.0"
    assert metadata["generated_at_utc"].endswith("Z")
    assert metadata["source"]["sha256"] == _sha256(source_path)
    assert metadata["artifacts"]["train_splits"]["sha256"] == _sha256(train_path)
    assert metadata["artifacts"]["test"]["sha256"] == _sha256(test_path)
    assert metadata["generator"]["sha256"] == _sha256(Path("src/split_folds.py"))
    assert not Path(metadata["source"]["path"]).is_absolute()
    assert all(
        not Path(artifact["path"]).is_absolute()
        for artifact in metadata["artifacts"].values()
    )
    assert metadata["configuration"] == {
        "forecast_horizon_hours": 24,
        "initial_train_fraction_of_development": 0.5,
        "n_folds": 5,
        "test_fraction": 0.2,
    }
    assert len(metadata["folds"]) == 5
    assert metadata["folds"][0]["nominal_validation_rows"] == 41
    assert metadata["folds"][0]["role_counts"]["embargo"] == 24
    assert metadata["folds"][0]["eligible_counts"]["validation"] == 17
    assert metadata["artifacts"]["train_splits"]["null_counts"] == {
        column: int(count) for column, count in persisted_train.isna().sum().items()
    }
    assert metadata["artifacts"]["test"]["schema"] == {
        column: str(dtype) for column, dtype in persisted_test.dtypes.items()
    }
    assert set(persisted_train.columns[-6:]) == {
        "target_valid",
        "fold_01_role",
        "fold_02_role",
        "fold_03_role",
        "fold_04_role",
        "fold_05_role",
    }
    assert list(persisted_test.columns[-1:]) == ["target_valid"]
    assert all(
        str(persisted_train[column].dtype) == "category"
        for column in split.role_columns
    )
    assert metadata_path.stat().st_mtime_ns >= train_path.stat().st_mtime_ns
    assert metadata_path.stat().st_mtime_ns >= test_path.stat().st_mtime_ns
