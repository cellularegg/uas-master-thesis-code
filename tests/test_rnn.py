import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import dump  # type: ignore[import-untyped]
from joblib import load as load_joblib  # type: ignore[import-untyped]

from src import rnn
from src.config import FORECAST_HORIZON_HOURS, TARGET_STATION_ID, WEATHER_VARIABLES
from src.dataset import JoinedDataset, load_joined_dataset
from src.rnn import (
    RnnForecaster,
    RnnManifest,
    build_sequences,
    load_raw_channel_frame,
    load_rnn_manifest,
    save_rnn_manifest,
    score_saved_model,
    select_candidate,
)


def _write_feature_artifacts(root: Path) -> tuple[JoinedDataset, Path, Path]:
    """Write a synthetic, contiguous-hourly joined artifact triple."""
    processed_dir = root / "data" / "processed" / "joined"
    processed_dir.mkdir(parents=True)
    station_id = TARGET_STATION_ID
    predictor_columns = [
        f"{station_id}__water_level",
        f"{station_id}__imputed",
        f"{station_id}__precipitation",
        f"{station_id}__temperature_2m",
        "station-b__water_level",
        "station-b__imputed",
        "station-b__precipitation",
        "station-b__temperature_2m",
    ]
    target_columns = [
        f"{station_id}__target_t_plus_{horizon:02d}"
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    metadata_path = processed_dir / "all_stations_feature_metadata.json"
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

    def frame(row_count: int, start: str) -> pd.DataFrame:
        built = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start, periods=row_count, freq="h", tz="UTC"
                ),
                f"{station_id}__target_valid": True,
            }
        )
        for predictor_number, column in enumerate(predictor_columns, start=1):
            built[column] = predictor_number + np.arange(row_count, dtype=float)
        for horizon, column in enumerate(target_columns, start=1):
            built[column] = 10.0 * np.arange(row_count, dtype=float) + horizon
        return built

    train_path = processed_dir / "all_stations_train_features.parquet"
    test_path = processed_dir / "all_stations_test_features.parquet"
    frame(40, "2024-01-01").to_parquet(train_path, index=False)
    frame(40, "2024-03-01").to_parquet(test_path, index=False)
    dataset = load_joined_dataset(
        metadata_path,
        train_path,
        test_path,
        station_id=station_id,
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
        weather_variables=WEATHER_VARIABLES,
        initial_train_fraction=0.5,
        n_validation_folds=2,
        embargo_rows=0,
    )
    return dataset, train_path, test_path


def _cv_results(selected_cell_type: str = "gru") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_type": selected_cell_type,
                "sequence_length": 4,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.1,
                "sequence_eligible_train_rows": 37,
                "mae_mean": 1.0,
                "rmse_mean": 1.5,
            },
            {
                "cell_type": "lstm",
                "sequence_length": 6,
                "hidden_size": 16,
                "num_layers": 2,
                "dropout": 0.1,
                "sequence_eligible_train_rows": 35,
                "mae_mean": 2.0,
                "rmse_mean": 2.5,
            },
        ]
    )


def _per_horizon_metrics(horizons: int = FORECAST_HORIZON_HOURS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "station_id": TARGET_STATION_ID,
                "horizon_hours": horizon,
                "target": f"target_{horizon:02d}",
                "mae": 1.0 + horizon,
                "rmse": 2.0 + horizon,
                "me": 0.25,
                "r2": 0.75,
            }
            for horizon in range(1, horizons + 1)
        ]
    )


def _save(
    dataset: JoinedDataset,
    manifest_path: Path,
    *,
    channel_columns: list[str] | None = None,
    selected_cell_type: str = "gru",
    per_horizon_metrics: pd.DataFrame | None = None,
    sealed_test_metrics: dict[str, float] | None = None,
) -> None:
    save_rnn_manifest(
        manifest_path,
        model_path=manifest_path.with_suffix(".joblib"),
        execution_uuid="execution-1",
        contract=dataset.contract,
        channel_columns=(
            dataset.feature_subsets["raw_all_stations"]
            if channel_columns is None
            else channel_columns
        ),
        selected_cell_type=selected_cell_type,
        selected_sequence_length=4,
        selected_hidden_size=8,
        selected_num_layers=1,
        fixed_dropout=0.1,
        selection_metric="rmse",
        cv_results=_cv_results(selected_cell_type),
        sealed_test_metrics=sealed_test_metrics
        or {"test_mae": 1.0, "test_rmse": 2.0, "test_me": 0.25, "test_r2": 0.75},
        per_horizon_metrics=(
            _per_horizon_metrics()
            if per_horizon_metrics is None
            else per_horizon_metrics
        ),
        scored_test_rows=37,
        cohort={"train_eligible_rows": len(dataset.train_rows)},
        training={"source_artifact": "train.parquet"},
    )


def test_select_candidate_ranks_by_metric_before_ties() -> None:
    winner = select_candidate(
        pd.DataFrame(
            [
                {
                    "cell_type": "lstm",
                    "sequence_length": 72,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "mae_mean": 2.0,
                    "rmse_mean": 2.0,
                },
                {
                    "cell_type": "gru",
                    "sequence_length": 24,
                    "hidden_size": 32,
                    "num_layers": 1,
                    "mae_mean": 1.0,
                    "rmse_mean": 1.0,
                },
            ]
        )
    )
    assert winner == ("gru", 24, 32, 1)


def _tied_candidate(
    *, sequence_length: int, hidden_size: int, num_layers: int, cell_type: str
) -> dict[str, object]:
    return {
        "cell_type": cell_type,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "mae_mean": 1.0,
        "rmse_mean": 1.0,
    }


def test_select_candidate_breaks_ties_through_every_axis_in_order() -> None:
    # Equal-metric rows, isolating one tie-break axis at a time: earlier axes
    # (sequence_length, then hidden_size, then num_layers, then cell_type)
    # must decide before a later axis gets a chance to.
    shorter_sequence_length_winner = select_candidate(
        pd.DataFrame(
            [
                _tied_candidate(
                    sequence_length=72, hidden_size=32, num_layers=1, cell_type="gru"
                ),
                _tied_candidate(
                    sequence_length=24, hidden_size=32, num_layers=1, cell_type="gru"
                ),
            ]
        )
    )
    assert shorter_sequence_length_winner == ("gru", 24, 32, 1)

    smaller_hidden_size_winner = select_candidate(
        pd.DataFrame(
            [
                _tied_candidate(
                    sequence_length=24, hidden_size=64, num_layers=1, cell_type="gru"
                ),
                _tied_candidate(
                    sequence_length=24, hidden_size=32, num_layers=1, cell_type="gru"
                ),
            ]
        )
    )
    assert smaller_hidden_size_winner == ("gru", 24, 32, 1)

    fewer_num_layers_winner = select_candidate(
        pd.DataFrame(
            [
                _tied_candidate(
                    sequence_length=24, hidden_size=32, num_layers=2, cell_type="gru"
                ),
                _tied_candidate(
                    sequence_length=24, hidden_size=32, num_layers=1, cell_type="gru"
                ),
            ]
        )
    )
    assert fewer_num_layers_winner == ("gru", 24, 32, 1)

    stable_cell_type_winner = select_candidate(
        pd.DataFrame(
            [
                _tied_candidate(
                    sequence_length=24, hidden_size=32, num_layers=1, cell_type="lstm"
                ),
                _tied_candidate(
                    sequence_length=24, hidden_size=32, num_layers=1, cell_type="gru"
                ),
            ]
        )
    )
    assert stable_cell_type_winner == ("gru", 24, 32, 1)


def test_load_raw_channel_frame_validates_contiguous_hourly_grid(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "water_level": np.arange(10, dtype=float),
            "imputed": [False] * 10,
        }
    )
    good_path = tmp_path / "good.parquet"
    frame.to_parquet(good_path, index=False)

    channel_frame = load_raw_channel_frame(
        good_path, channel_columns=["water_level", "imputed"]
    )

    assert channel_frame.index.equals(timestamps)
    assert list(channel_frame.columns) == ["water_level", "imputed"]

    gapped = frame.drop(index=5).reset_index(drop=True)
    gapped_path = tmp_path / "gapped.parquet"
    gapped.to_parquet(gapped_path, index=False)
    with pytest.raises(ValueError, match="contiguous hourly grid"):
        load_raw_channel_frame(gapped_path, channel_columns=["water_level", "imputed"])

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicated_path = tmp_path / "duplicated.parquet"
    duplicated.to_parquet(duplicated_path, index=False)
    with pytest.raises(ValueError, match="unique and increasing"):
        load_raw_channel_frame(
            duplicated_path, channel_columns=["water_level", "imputed"]
        )

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_channel_frame(
            good_path, channel_columns=["water_level", "temperature_2m"]
        )


def test_build_sequences_builds_complete_windows_only() -> None:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    raw_channel_frame = pd.DataFrame(
        {"water_level": np.arange(20, dtype=float), "imputed": [False] * 20},
        index=timestamps,
    )
    # A single missing hour at position 10 invalidates the four windows ending
    # at positions 10-13 (sequence_length=4).
    raw_channel_frame.loc[timestamps[10], "water_level"] = np.nan
    # An infinite value at position 5 must be treated the same as NaN: it
    # invalidates the four windows ending at positions 5-8.
    raw_channel_frame.loc[timestamps[5], "water_level"] = np.inf
    # Position 15 is imputed but present, so it must stay eligible.
    raw_channel_frame.loc[timestamps[15], "imputed"] = True

    candidate_rows = pd.DataFrame(
        {"timestamp": timestamps, "target_1": np.arange(20, dtype=float)}
    )

    sequences, targets, origins = build_sequences(
        raw_channel_frame,
        candidate_rows,
        sequence_length=4,
        channel_columns=["water_level", "imputed"],
        target_columns=["target_1"],
        artifact_name="train",
    )

    assert sequences.shape == (9, 4, 2)
    assert targets.shape == (9, 1)
    assert len(origins) == 9
    assert np.isfinite(sequences).all()
    origin_timestamps = set(pd.to_datetime(origins["timestamp"], utc=True))
    assert timestamps[15] in origin_timestamps
    for excluded_position in (0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13):
        assert timestamps[excluded_position] not in origin_timestamps


def test_build_sequences_rejects_when_no_window_is_complete() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    raw_channel_frame = pd.DataFrame(
        {"water_level": np.arange(5, dtype=float)}, index=timestamps
    )
    candidate_rows = pd.DataFrame(
        {"timestamp": timestamps, "target_1": np.arange(5, dtype=float)}
    )

    with pytest.raises(ValueError, match="no eligible"):
        build_sequences(
            raw_channel_frame,
            candidate_rows,
            sequence_length=10,
            channel_columns=["water_level"],
            target_columns=["target_1"],
            artifact_name="train",
        )


def _synthetic_sequences(
    n: int = 30, sequence_length: int = 4, n_channels: int = 3, horizon: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    sequences = rng.uniform(-1.0, 1.0, size=(n, sequence_length, n_channels))
    targets = rng.uniform(-1.0, 1.0, size=(n, horizon))
    return sequences, targets


def test_rnn_forecaster_fit_predict_round_trips_shapes_and_finiteness() -> None:
    sequences, targets = _synthetic_sequences()
    forecaster = RnnForecaster(
        cell_type="gru",
        hidden_size=4,
        num_layers=1,
        max_epochs=2,
        batch_size=8,
        device="cpu",
    )

    forecaster.fit(sequences, targets)
    predictions = forecaster.predict(sequences)

    assert predictions.shape == (len(sequences), targets.shape[1])
    assert predictions.dtype == np.float64
    assert np.isfinite(predictions).all()


def test_rnn_forecaster_is_reproducible_with_a_fixed_random_state() -> None:
    sequences, targets = _synthetic_sequences()

    first = RnnForecaster(
        cell_type="lstm",
        hidden_size=4,
        num_layers=1,
        max_epochs=2,
        batch_size=8,
        random_state=7,
        device="cpu",
    ).fit(sequences, targets)
    second = RnnForecaster(
        cell_type="lstm",
        hidden_size=4,
        num_layers=1,
        max_epochs=2,
        batch_size=8,
        random_state=7,
        device="cpu",
    ).fit(sequences, targets)

    np.testing.assert_allclose(first.predict(sequences), second.predict(sequences))


def test_rnn_forecaster_persists_a_cpu_module_via_joblib(tmp_path: Path) -> None:
    sequences, targets = _synthetic_sequences()
    forecaster = RnnForecaster(
        cell_type="gru",
        hidden_size=4,
        num_layers=1,
        max_epochs=2,
        batch_size=8,
        device="cpu",
    ).fit(sequences, targets)
    model_path = tmp_path / "rnn.joblib"

    dump(forecaster, model_path)
    reloaded = load_joblib(model_path)

    np.testing.assert_allclose(
        reloaded.predict(sequences), forecaster.predict(sequences)
    )
    assert next(reloaded.module_.parameters()).device.type == "cpu"


def test_rnn_manifest_round_trips_the_selection_and_sealed_test_record(
    tmp_path: Path,
) -> None:
    dataset, _train_path, _test_path = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "rnn.json"

    _save(dataset, manifest_path)
    manifest = load_rnn_manifest(
        manifest_path,
        contract=dataset.contract,
        channel_columns=dataset.feature_subsets["raw_all_stations"],
    )

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "1.0"
    assert manifest.execution_uuid == "execution-1"
    assert manifest.selected_cell_type == "gru"
    assert manifest.selected_sequence_length == 4
    assert manifest.selected_hidden_size == 8
    assert manifest.selected_num_layers == 1
    assert manifest.fixed_dropout == 0.1
    assert manifest.selection_metric == "rmse"
    assert manifest.channel_columns == tuple(
        dataset.feature_subsets["raw_all_stations"]
    )
    assert manifest.target_columns == dataset.contract.target_columns
    assert manifest.cv_results["cell_type"].tolist() == ["gru", "lstm"]
    assert manifest.sealed_test_metrics == {
        "test_mae": 1.0,
        "test_rmse": 2.0,
        "test_me": 0.25,
        "test_r2": 0.75,
    }
    assert manifest.scored_test_rows == 37
    assert manifest.horizon_metrics["horizon_hours"].tolist() == list(
        range(1, FORECAST_HORIZON_HOURS + 1)
    )


def test_load_rnn_manifest_reports_missing_manifest(tmp_path: Path) -> None:
    dataset, _train_path, _test_path = _write_feature_artifacts(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Run 04_07_train_rnn.ipynb through its sealed-test cell",
    ):
        load_rnn_manifest(
            tmp_path / "absent.json",
            contract=dataset.contract,
            channel_columns=dataset.feature_subsets["raw_all_stations"],
        )


def test_load_rnn_manifest_rejects_an_older_schema_version(tmp_path: Path) -> None:
    dataset, _train_path, _test_path = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "rnn.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["schema_version"] = "0.9"
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        load_rnn_manifest(
            manifest_path,
            contract=dataset.contract,
            channel_columns=dataset.feature_subsets["raw_all_stations"],
        )


def test_load_rnn_manifest_rejects_contract_mismatches(tmp_path: Path) -> None:
    dataset, _train_path, _test_path = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "rnn.json"
    _save(dataset, manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))

    for field, value, message in (
        ("station_id", "other-station", "station_id"),
        ("forecast_horizon_hours", FORECAST_HORIZON_HOURS + 1, "forecast horizon"),
        ("target_columns", ["only-one"], "target contract"),
        ("channel_columns", ["only-one"], "channel columns"),
    ):
        manifest_path.write_text(json.dumps({**stored, field: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_rnn_manifest(
                manifest_path,
                contract=dataset.contract,
                channel_columns=dataset.feature_subsets["raw_all_stations"],
            )


def test_save_rnn_manifest_rejects_an_incomplete_sealed_test_record(
    tmp_path: Path,
) -> None:
    dataset, _train_path, _test_path = _write_feature_artifacts(tmp_path)
    manifest_path = tmp_path / "rnn.json"

    with pytest.raises(ValueError, match="Sealed-test metrics are missing"):
        _save(
            dataset,
            manifest_path,
            sealed_test_metrics={"test_mae": 1.0, "test_rmse": 2.0},
        )
    with pytest.raises(ValueError, match="non-finite"):
        _save(
            dataset,
            manifest_path,
            sealed_test_metrics={
                "test_mae": float("nan"),
                "test_rmse": 2.0,
                "test_me": 0.25,
                "test_r2": 0.75,
            },
        )
    with pytest.raises(ValueError, match="do not match the forecast contract"):
        _save(
            dataset,
            manifest_path,
            per_horizon_metrics=_per_horizon_metrics(FORECAST_HORIZON_HOURS - 1),
        )
    # Nothing is written until every check passes.
    assert not manifest_path.exists()


class _FakeSavedRnn:
    def __init__(self) -> None:
        self.sequence_values: np.ndarray | None = None

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        self.sequence_values = sequences.copy()
        row_numbers = np.arange(len(sequences), dtype=float)[:, None]
        horizon_numbers = np.arange(1, FORECAST_HORIZON_HOURS + 1, dtype=float)[None, :]
        return row_numbers * 10.0 + horizon_numbers + 100.0


def test_score_saved_model_scores_only_the_sequence_eligible_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _train_path, test_path = _write_feature_artifacts(tmp_path)
    channel_columns = dataset.feature_subsets["raw_all_stations"]
    raw_test_channel_frame = load_raw_channel_frame(
        test_path, channel_columns=channel_columns
    )
    sequence_length = 6
    manifest = RnnManifest(
        execution_uuid="execution-1",
        station_id=dataset.contract.station_id,
        selected_cell_type="gru",
        selected_sequence_length=sequence_length,
        selected_hidden_size=8,
        selected_num_layers=1,
        fixed_dropout=0.1,
        selection_metric="rmse",
        channel_columns=tuple(channel_columns),
        target_columns=dataset.contract.target_columns,
        cv_results=_cv_results(),
        sealed_test_metrics={
            "test_mae": 1.0,
            "test_rmse": 2.0,
            "test_me": 0.25,
            "test_r2": 0.75,
        },
        horizon_metrics=_per_horizon_metrics(),
        scored_test_rows=len(dataset.test_rows) - (sequence_length - 1),
    )
    fake_model = _FakeSavedRnn()
    monkeypatch.setattr(rnn, "load_joblib", lambda _path: fake_model)

    predictions, origins = score_saved_model(
        manifest, tmp_path / "rnn.joblib", dataset.test_rows, raw_test_channel_frame
    )

    assert len(origins) == len(dataset.test_rows) - (sequence_length - 1)
    assert len(origins) <= len(dataset.test_rows)
    assert predictions.shape == (len(origins), FORECAST_HORIZON_HOURS)
    assert fake_model.sequence_values is not None
    assert fake_model.sequence_values.shape == (
        len(origins),
        sequence_length,
        len(channel_columns),
    )
