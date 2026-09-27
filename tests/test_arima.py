import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]

from src import arima
from src.arima import (
    ArimaCandidate,
    FittedArima,
    candidate_grid,
    candidate_table,
    evaluate_candidate,
    fit_arima,
    latest_complete_segment,
    load_arima_model,
    observed_hourly_history,
    rolling_forecasts,
    save_arima_model,
    select_candidate,
    training_history,
)
from src.dataset import JoinedFeatureContract


@pytest.fixture
def history() -> pd.Series:
    return pd.Series(
        200 + np.cumsum(np.random.default_rng(519).normal(size=160)),
        index=pd.date_range("2024-01-01", periods=160, freq="h", tz="UTC"),
    )


@pytest.fixture
def contract() -> JoinedFeatureContract:
    return JoinedFeatureContract(
        station_id="station-a",
        target_valid_column="station-a__target_valid",
        predictor_columns=("station-a__water_level", "station-a__imputed"),
        target_columns=tuple(f"station-a__target_t_plus_{h:02d}" for h in range(1, 25)),
    )


@pytest.fixture
def model(history: pd.Series, contract: JoinedFeatureContract) -> FittedArima:
    # Known fixed AR(1) parameters isolate filtering from optimizer behavior.
    results = SARIMAX(history.iloc[:80].to_numpy(), order=(1, 0, 0)).filter([0.9, 1.0])
    return FittedArima(
        results=results,
        station_id=contract.station_id,
        target_columns=contract.target_columns,
        state_time=history.index[79],
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
    )


def test_observed_history_restores_missing_and_preserves_elapsed_time(
    history: pd.Series,
) -> None:
    frame = pd.DataFrame({"water_level": history, "imputed": False})
    frame.iloc[3, 1] = True
    frame = frame.drop(history.index[4]).iloc[::-1]
    frame.index = frame.index.tz_convert("Europe/Vienna")
    observed = observed_hourly_history(frame)
    assert observed.index.equals(history.index)
    assert observed.iloc[3:5].isna().all()
    assert observed.iloc[5] == history.iloc[5]


@pytest.mark.parametrize(
    "invalid", ["duplicates", "naive", "fractional", "flags", "infinite"]
)
def test_observed_history_rejects_invalid_inputs(
    history: pd.Series, invalid: str
) -> None:
    frame = pd.DataFrame({"water_level": history, "imputed": False})
    if invalid == "duplicates":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif invalid == "naive":
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    elif invalid == "fractional":
        frame.index += pd.Timedelta(minutes=1)
    elif invalid == "flags":
        frame["imputed"] = 0
    else:
        frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        observed_hourly_history(frame)


def test_training_history_keeps_every_observed_hour_and_gap_through_cutoff(
    history: pd.Series,
) -> None:
    history.iloc[:5] = np.nan
    history.iloc[30:50] = np.nan
    history.iloc[108:111] = np.nan
    segment = training_history(history, cutoff=history.index[110])
    # Only leading missing hours are dropped; gaps and trailing gaps stay missing.
    assert segment.index.equals(history.index[5:111])
    assert segment.iloc[25:45].isna().all()
    assert segment.iloc[-3:].isna().all()
    changed_future = history.copy()
    changed_future.iloc[111:] = -10000
    pd.testing.assert_series_equal(
        segment, training_history(changed_future, cutoff=history.index[110])
    )


def test_training_history_rejects_invalid_cutoffs(history: pd.Series) -> None:
    with pytest.raises(ValueError, match="outside"):
        training_history(history, cutoff=history.index[-1] + pd.Timedelta(hours=1))
    history.iloc[:100] = np.nan
    with pytest.raises(ValueError, match="No observed"):
        training_history(history, cutoff=history.index[99])


def test_latest_complete_segment_is_the_trailing_gap_free_run(
    history: pd.Series,
) -> None:
    history.iloc[40:42] = np.nan
    history.iloc[150:] = np.nan
    assert latest_complete_segment(history).index.equals(history.index[42:150])
    with pytest.raises(ValueError):
        latest_complete_segment(history * np.nan)


def test_fit_uses_full_history_with_missing_hours(
    history: pd.Series, contract: JoinedFeatureContract
) -> None:
    history.iloc[20:30] = np.nan
    fitted, diagnostics = fit_arima(
        history,
        cutoff=history.index[99],
        contract=contract,
        candidate=ArimaCandidate((1, 0, 0), True),
    )
    assert fitted.state_time == history.index[99]
    assert fitted.results.nobs == 100
    assert diagnostics["fit_hours"] == 100
    assert diagnostics["fit_observed_hours"] == 90
    assert diagnostics["fit_missing_hours"] == 10
    assert diagnostics["fit_cutoff"] == history.index[99].isoformat()


def test_rolling_forecasts_use_only_past_observations_and_leave_model_unchanged(
    history: pd.Series,
    model: FittedArima,
) -> None:
    history.iloc[85:90] = np.nan
    issues = history.index[[80, 84, 87, 90, 100]]
    params_before = model.results.params.copy()
    state_before = model.results.filter_results.filtered_state.copy()
    predictions = rolling_forecasts(model, history, issues)
    # Full filtering at each origin is an independent reference, including NaNs.
    for row, issue in enumerate(issues):
        reference = SARIMAX(history.loc[:issue].to_numpy(), order=(1, 0, 0)).filter(
            params_before
        )
        np.testing.assert_allclose(predictions[row], reference.forecast(24))
    changed = history.copy()
    changed.loc[history.index[91] :] = -10000
    np.testing.assert_allclose(
        predictions[:4], rolling_forecasts(model, changed, issues[:4])
    )
    assert not np.allclose(
        predictions[-1], rolling_forecasts(model, changed, issues)[-1]
    )
    np.testing.assert_array_equal(model.results.params, params_before)
    np.testing.assert_array_equal(
        model.results.filter_results.filtered_state, state_before
    )
    assert model.state_time == history.index[79]
    assert predictions.shape == (5, 24)


@pytest.mark.parametrize(
    "invalid", ["past", "duplicates", "reverse", "missing_hour", "too_short"]
)
def test_rolling_forecasts_reject_invalid_origins_or_grid(
    history: pd.Series,
    model: FittedArima,
    invalid: str,
) -> None:
    issues = history.index[[80, 90]]
    if invalid == "past":
        issues = history.index[[79, 90]]
    elif invalid == "duplicates":
        issues = history.index[[80, 80]]
    elif invalid == "reverse":
        issues = issues[::-1]
    elif invalid == "missing_hour":
        history = history.drop(history.index[85])
    else:
        history = history.iloc[:90]
    with pytest.raises(ValueError):
        rolling_forecasts(model, history, issues)


def test_real_arima_fit_and_saved_model_roundtrip(
    history: pd.Series,
    contract: JoinedFeatureContract,
    tmp_path: Path,
) -> None:
    fitted, diagnostics = fit_arima(
        history,
        cutoff=history.index[99],
        contract=contract,
        candidate=ArimaCandidate((1, 1, 0), False),
    )
    assert diagnostics["fit_hours"] == 100
    assert diagnostics["fit_start"] == history.index[0].isoformat()
    assert diagnostics["fit_missing_hours"] == 0
    model_path, manifest_path = tmp_path / "arima.joblib", tmp_path / "arima.json"
    save_arima_model(fitted, model_path, manifest_path, execution_uuid="test-execution")
    restored = load_arima_model(model_path, manifest_path, contract=contract)
    issues = history.index[[100, 120, 150]]
    np.testing.assert_array_equal(
        rolling_forecasts(fitted, history, issues),
        rolling_forecasts(restored, history, issues),
    )
    manifest = json.loads(manifest_path.read_text())
    assert "fit_hours" not in manifest
    assert "converged" not in manifest
    assert not any(key.startswith(("cv_", "test_")) for key in manifest)
    assert manifest["execution_uuid"] == "test-execution"
    assert manifest["schema_version"] == "5.0"
    assert manifest["trend"] == "n"
    assert manifest["model_file"] == "arima.joblib"
    assert {"training_data_policy", "future_predictor_policy"} <= set(manifest)


@pytest.mark.parametrize(
    "mismatch",
    [
        "station",
        "targets",
        "frequency",
        "artifact",
        "estimator",
        "schema_version",
        "order",
        "intercept",
        "trend",
        "training_data_policy",
        "update_policy",
    ],
)
def test_manifest_rejects_mismatch_before_deserialization(
    model: FittedArima,
    contract: JoinedFeatureContract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    model_path, manifest_path = tmp_path / "arima.joblib", tmp_path / "arima.json"
    save_arima_model(model, model_path, manifest_path, execution_uuid="test")
    if mismatch == "station":
        contract = replace(contract, station_id="other")
    elif mismatch == "targets":
        contract = replace(contract, target_columns=contract.target_columns[::-1])
    elif mismatch == "frequency":
        manifest = json.loads(manifest_path.read_text())
        manifest["input_frequency"] = "D"
        manifest_path.write_text(json.dumps(manifest))
    elif mismatch == "artifact":
        model_path.write_bytes(b"different artifact")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest[mismatch] = {
            "order": [-1, 0, 0],
            "intercept": "false",
            "trend": "c",
        }.get(mismatch, "invalid")
        manifest_path.write_text(json.dumps(manifest))

    def forbidden_load(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Deserialization must not happen for an invalid manifest")

    monkeypatch.setattr(arima, "load", forbidden_load)
    with pytest.raises(ValueError):
        load_arima_model(model_path, manifest_path, contract=contract)


def test_grid_is_complete() -> None:
    grid = candidate_grid()
    assert len(grid) == len(set(grid)) == 52
    assert {c.order for c in grid} == {
        (p, d, q)
        for p in range(7)
        for d in range(2)
        for q in range(4)
        if d == 1 or p > 0
    }
    assert all(c.intercept == (c.order[1] == 0) for c in grid)
    assert sum(c.intercept for c in grid) == 24


@pytest.mark.parametrize(
    "order,intercept",
    [((-1, 0, 0), False), ((0, 0, 0), 1), ((1, 1, 0), True), ((0, 0), False)],
)
def test_invalid_candidate(order: Any, intercept: Any) -> None:
    with pytest.raises(ValueError):
        ArimaCandidate(order, intercept)


def _evaluation(
    candidate: ArimaCandidate, score: float = 1.0
) -> "arima.CandidateEvaluation":
    return arima.CandidateEvaluation(
        candidate,
        [{"status": "ok"}],
        pd.DataFrame(),
        pd.DataFrame(),
        {"cv_rmse_mean": score, "cv_mae_mean": score},
        0,
        0,
        [(0, 0)],
    )


def test_selection_ties_and_failed_folds() -> None:
    ordered = [
        ArimaCandidate((0, 0, 0), False),
        ArimaCandidate((0, 0, 0), True),
        ArimaCandidate((0, 1, 0), False),
        ArimaCandidate((0, 0, 1), False),
        ArimaCandidate((1, 0, 0), False),
        ArimaCandidate((1, 0, 0), True),
        ArimaCandidate((1, 1, 0), False),
        ArimaCandidate((1, 0, 1), False),
    ]
    for i in range(len(ordered)):
        table = candidate_table([_evaluation(c) for c in reversed(ordered[i:])])
        assert select_candidate(table) == (ordered[i].order, ordered[i].intercept)
    table = candidate_table([_evaluation(ordered[-1], 0.5), _evaluation(ordered[0])])
    assert select_candidate(table) == (ordered[-1].order, ordered[-1].intercept)
    failed = _evaluation(ordered[0], 0.0)
    failed.fold_details.append({"status": "failed"})
    table = candidate_table([failed, _evaluation(ordered[1])])
    assert table["intercept"].tolist() == [True]
    assert select_candidate(table) == (ordered[1].order, ordered[1].intercept)
    with pytest.raises(ValueError, match="every validation fold"):
        select_candidate(candidate_table([failed]))


def _rows(history: pd.Series, contract: JoinedFeatureContract) -> pd.DataFrame:
    rows = pd.DataFrame({"timestamp": history.index})
    for h, target in enumerate(contract.target_columns, 1):
        rows[target] = history.shift(-h).to_numpy()
    return rows


def test_synthetic_cv_and_failure_handling(
    history: pd.Series,
    contract: JoinedFeatureContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(history, contract)
    folds = [
        (np.arange(40), np.array([65, 70, 75])),
        (np.arange(60), np.array([90, 95, 100])),
    ]
    candidate = ArimaCandidate((0, 1, 0), False)
    result = evaluate_candidate(candidate, history, rows, folds, contract=contract)
    assert len(result.aggregate) == 2
    assert result.aggregate["fold"].tolist() == [1, 2]
    assert result.summary["cv_rmse_mean"] == result.aggregate["rmse"].mean()
    assert [d["scored_issue_times"] for d in result.fold_details] == [3, 3]
    # Each fold fits through its last training issue time plus the horizon.
    assert [d["fit_cutoff"] for d in result.fold_details] == [
        history.index[63].isoformat(),
        history.index[83].isoformat(),
    ]
    assert [d["fit_hours"] for d in result.fold_details] == [64, 84]
    assert select_candidate(candidate_table([result])) == ((0, 1, 0), False)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("invalid forecast")

    monkeypatch.setattr(arima, "rolling_forecasts", fail)
    failed = evaluate_candidate(candidate, history, rows, folds, contract=contract)
    assert not failed.summary
    assert all("invalid forecast" in d["failure"] for d in failed.fold_details)
    assert all("converged" in d for d in failed.fold_details)
    monkeypatch.setattr(arima, "fit_arima", fail)
    failed = evaluate_candidate(candidate, history, rows, folds, contract=contract)
    assert all(d["status"] == "failed" for d in failed.fold_details)


def test_nonconvergence_retained_and_logged(
    history: pd.Series,
    contract: JoinedFeatureContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import mlflow

    real_sarimax = arima.SARIMAX

    def nonconverged(*args: Any, **kwargs: Any) -> Any:
        estimator = real_sarimax(*args, **kwargs)
        original = estimator.fit

        def fit(**options: Any) -> Any:
            assert options["maxiter"] == 200
            result = original(**options)
            result.mle_retvals["converged"] = False
            return result

        estimator.fit = fit
        return estimator

    monkeypatch.setattr(arima, "SARIMAX", nonconverged)
    result = evaluate_candidate(
        ArimaCandidate((0, 1, 0), False),
        history,
        _rows(history, contract),
        [(np.arange(56), np.array([85, 90]))],
        contract=contract,
    )
    assert result.summary
    assert result.fold_details[0]["converged"] is False
    assert result.fold_details[0]["fit_hours"] == 80
    logged: list[dict[str, Any]] = []
    from contextlib import nullcontext

    runs: dict[str, dict[str, Any]] = {}

    class FakeClient:
        def create_run(self, **kwargs: Any) -> SimpleNamespace:
            runs["child"] = dict(kwargs)
            return SimpleNamespace(info=SimpleNamespace(run_id="child"))

        def set_terminated(self, run_id: str, end_time: int) -> None:
            runs[run_id]["end_time"] = end_time

    monkeypatch.setattr(
        mlflow,
        "active_run",
        lambda: SimpleNamespace(
            info=SimpleNamespace(run_id="parent", experiment_id="0"),
            data=SimpleNamespace(tags={"execution_uuid": "test"}),
        ),
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", FakeClient)
    monkeypatch.setattr(mlflow, "start_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr(mlflow, "log_params", lambda values: logged.append(values))
    monkeypatch.setattr(mlflow, "log_metrics", lambda values: None)
    monkeypatch.setattr(mlflow, "set_tag", lambda *args: None)
    arima.log_candidate(result)
    assert any(values.get("converged") is False for values in logged)
    # Fold runs span the worker-side evaluation, not the later logging call.
    started_at_ms, ended_at_ms = result.fold_times_ms[0]
    assert result.started_at_ms <= started_at_ms <= ended_at_ms <= result.ended_at_ms
    assert runs["child"]["start_time"] == started_at_ms
    assert runs["child"]["end_time"] == ended_at_ms
    assert runs["child"]["tags"]["mlflow.parentRunId"] == "parent"


@pytest.mark.parametrize("n_workers", [0, -1, True, 1.5])
def test_parallel_search_rejects_invalid_worker_count(
    n_workers: Any,
    contract: JoinedFeatureContract,
) -> None:
    with pytest.raises(ValueError, match="n_workers"):
        list(
            arima.evaluate_candidates(
                [],
                pd.Series(dtype=float),
                pd.DataFrame(),
                [],
                contract=contract,
                n_workers=n_workers,
            )
        )


def test_parallel_search_matches_sequential_results(
    history: pd.Series,
    contract: JoinedFeatureContract,
) -> None:
    rows = _rows(history, contract)
    folds = [(np.arange(40), np.array([65, 70])), (np.arange(60), np.array([90, 95]))]
    candidates = [
        ArimaCandidate((0, 1, 0), False),
        ArimaCandidate((0, 0, 0), True),
    ]
    serial = list(
        arima.evaluate_candidates(
            candidates, history, rows, folds, contract=contract, n_workers=1
        )
    )
    parallel = list(
        arima.evaluate_candidates(
            candidates, history, rows, folds, contract=contract, n_workers=2
        )
    )
    assert [result.candidate for result in parallel] == candidates
    for expected, actual in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(expected.aggregate, actual.aggregate)
        pd.testing.assert_frame_equal(expected.horizons, actual.horizons)
        assert expected.fold_details == actual.fold_details
        assert actual.summary == pytest.approx(expected.summary)
    assert select_candidate(candidate_table(serial)) == select_candidate(
        candidate_table(parallel)
    )

    # Worker-side fit failures remain diagnostics and disqualify the candidate.
    short_history = history.copy()
    short_history.iloc[:80] = np.nan
    failed = list(
        arima.evaluate_candidates(
            candidates, short_history, rows, folds, contract=contract, n_workers=2
        )
    )
    assert all(not result.summary for result in failed)
    assert all(result.fold_details[0]["status"] == "failed" for result in failed)
    with pytest.raises(ValueError, match="every validation fold"):
        select_candidate(candidate_table(failed))
