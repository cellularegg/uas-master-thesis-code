import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import arimax
from src.arimax import (
    ArimaxCandidate,
    FittedArimax,
    candidate_grid,
    candidate_table,
    evaluate_candidate,
    fit_arimax,
    load_arimax_model,
    rolling_forecasts,
    save_arimax_model,
    select_candidate,
)
from src.dataset import JoinedFeatureContract
from src.feature_engineering import (
    build_feature_frame,
    feature_column_names,
    target_column_names,
    target_level_features,
    utc_calendar_features,
)
from src.training import numeric_predictors


@pytest.fixture(scope="module")
def data() -> tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(47)
    index = pd.date_range("2023-12-29", periods=190, freq="h", tz="UTC")
    weather = rng.normal(size=len(index))
    levels = np.empty(len(index))
    levels[0] = 100
    for i in range(1, len(index)):
        levels[i] = 20 + 0.8 * levels[i - 1] + weather[i - 1] + rng.normal(scale=0.3)
    x = pd.DataFrame(
        {
            "a__water_level": levels,
            "a__weather": weather,
            "a__constant": 1.0,
            "a__redundant": weather * 2,
            "a__utc_hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "a__utc_hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "a__utc_day_of_week_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "a__utc_day_of_week_cos": np.cos(2 * np.pi * index.dayofweek / 7),
            "a__utc_day_of_year_sin": np.sin(2 * np.pi * (index.dayofyear - 1) / 365),
            "a__utc_day_of_year_cos": np.cos(2 * np.pi * (index.dayofyear - 1) / 365),
        },
        index=index,
    )
    y = pd.DataFrame({"water_level": levels, "imputed": False}, index=index)
    columns = tuple(f"a__target_t_plus_{h:02d}" for h in range(1, 25))
    contract = JoinedFeatureContract(
        "a", "a__target_valid", tuple(map(str, x.columns)), columns
    )
    rows = x.reset_index(names="timestamp")
    for h, column in enumerate(columns, 1):
        rows[column] = y["water_level"].shift(-h).to_numpy()
    return contract, rows.iloc[:-24].copy(), x, y


@pytest.fixture(scope="module")
def fitted(
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> FittedArimax:
    contract, rows, _, _ = data
    model, _ = fit_arimax(
        rows.iloc[:75],
        contract=contract,
        candidate=ArimaxCandidate("full", contract.predictor_columns, (1, 0, 0)),
        maxiter=30,
    )
    assert isinstance(model, FittedArimax)
    return model


def test_single_fit_retains_redundant_columns_and_hourly_gaps(
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, rows, _, _ = data
    training = rows.iloc[:45].drop(index=[10, 11])
    model, detail = fit_arimax(
        training,
        contract=contract,
        candidate=ArimaxCandidate("full", contract.predictor_columns, (0, 0, 0)),
        maxiter=5,
    )
    assert isinstance(model, FittedArimax)
    assert model.predictor_scaler.n_features_in_ == len(contract.predictor_columns)
    assert model.target_scaler.n_features_in_ == 1
    assert model.coefficients.shape == (len(contract.predictor_columns) + 1,)
    assert model.result.model.k_exog == 0
    assert model.result.nobs == 45
    assert np.isnan(model.result.model.endog[10:12]).all()
    assert detail["constant_features"] == 1
    assert detail["predictor_rank"] < len(contract.predictor_columns)
    assert detail["fit_rows"] == 43
    assert (
        detail["last_label_time"]
        == (training["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).isoformat()
    )


def _calendar(timestamp: pd.Timestamp) -> dict[str, float]:
    days = 366 if timestamp.is_leap_year else 365
    angles = {
        "hour": 2 * np.pi * timestamp.hour / 24,
        "day_of_week": 2 * np.pi * timestamp.dayofweek / 7,
        "day_of_year": 2 * np.pi * (timestamp.dayofyear - 1) / days,
    }
    return {
        f"utc_{name}_{function.__name__}": float(function(angle))
        for name, angle in angles.items()
        for function in (np.sin, np.cos)
    }


def reference_forecasts(
    model: FittedArimax, x: pd.DataFrame, y: pd.DataFrame, issues: pd.DatetimeIndex
) -> np.ndarray:
    """Refilter from training separately at each origin and forecast stepwise."""
    hour = pd.Timedelta(hours=1)
    observed = y["water_level"].where(~y["imputed"].astype(bool))
    output = np.empty((len(issues), 24))
    for row, issue in enumerate(issues):
        residuals = []
        for input_time in pd.date_range(
            model.state_time + hour, issue - hour, freq="h"
        ):
            predictors = x.reindex([input_time])[list(model.contract.predictor_columns)]
            label = observed.get(input_time + hour, np.nan)
            if not np.isfinite(
                predictors.to_numpy(dtype=float)
            ).all() or not np.isfinite(label):
                residuals.append(np.nan)
                continue
            scaled_x = model.predictor_scaler.transform(
                predictors[list(model.candidate.feature_columns)].to_numpy(dtype=float)
            )
            design = arimax._design(scaled_x, model.candidate.order)
            residuals.append(
                (label - model.target_scaler.mean_[0]) / model.target_scaler.scale_[0]
                - float(design[0] @ model.coefficients)
            )
        state = (
            model.result.model.clone(
                np.concatenate([model.result.model.endog[:, 0], np.asarray(residuals)])
            ).filter(model.result.params)
            if residuals
            else model.result
        )
        errors = np.asarray(state.forecast(24))
        issue_x = x.reindex([issue])[list(model.candidate.feature_columns)].iloc[0]
        for step in range(24):
            future = issue_x.copy()
            if step:
                for base_name, value in _calendar(issue + step * hour).items():
                    column = f"a__{base_name}"
                    if column in future.index:
                        future[column] = value
                if "a__water_level" in future.index:
                    future["a__water_level"] = output[row, step - 1]
            scaled = model.predictor_scaler.transform(
                pd.DataFrame([future])[list(model.candidate.feature_columns)].to_numpy(
                    dtype=float
                )
            )
            regression = float(
                arimax._design(scaled, model.candidate.order)[0] @ model.coefficients
            )
            output[row, step] = (
                regression + errors[step]
            ) * model.target_scaler.scale_[0] + model.target_scaler.mean_[0]
    return output


@pytest.mark.parametrize(
    "order", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 0, 2)]
)
def test_24_step_state_forecast_matches_stepwise_reference(
    order: tuple[int, int, int],
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, rows, x, y = data
    model, _ = fit_arimax(
        rows.iloc[:65],
        contract=contract,
        candidate=ArimaxCandidate("full", contract.predictor_columns, order),
        maxiter=20,
    )
    assert isinstance(model, FittedArimax)
    x, y = x.copy(), y.copy()
    x.iloc[69, 1] = np.nan
    x = x.drop(x.index[71])
    y.loc[y.index[72], "imputed"] = True
    y.loc[y.index[72], "water_level"] = 1e9
    y = y.drop(y.index[73])
    issues = pd.DatetimeIndex(rows["timestamp"].iloc[np.array([74, 75, 80])])
    actual = rolling_forecasts(model, x, y, issues)
    assert actual.shape == (3, 24)
    np.testing.assert_allclose(
        actual, reference_forecasts(model, x, y, issues), rtol=1e-8, atol=1e-8
    )


def test_causal_updates_and_training_snapshot(
    fitted: FittedArimax,
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, rows, x, y = data
    issues = pd.DatetimeIndex(rows["timestamp"].iloc[80:83])
    state = fitted.result.filter_results.filtered_state.copy()
    nobs = fitted.result.nobs
    baseline = rolling_forecasts(fitted, x, y, issues)
    future_y, future_x = y.copy(), x.copy()
    future_y.loc[future_y.index > issues[0], "water_level"] += 1000
    future_x.loc[future_x.index > issues[0], "a__weather"] += 1000
    np.testing.assert_allclose(
        baseline[:1], rolling_forecasts(fitted, future_x, future_y, issues[:1])
    )
    np.testing.assert_allclose(
        baseline[:1],
        rolling_forecasts(fitted, x.loc[: issues[0]], y.loc[: issues[0]], issues[:1]),
    )
    changed = y.copy()
    changed["water_level"] += (
        pd.Series({issues[0]: 5.0}).reindex(changed.index).fillna(0)
    )
    assert not np.allclose(
        baseline[:1], rolling_forecasts(fitted, x, changed, issues[:1])
    )
    np.testing.assert_array_equal(fitted.result.filter_results.filtered_state, state)
    assert fitted.result.nobs == nobs
    np.testing.assert_allclose(baseline, rolling_forecasts(fitted, x, y, issues))


def test_missing_imputed_and_incomplete_update_rows(
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, rows, x, y = data
    model, _ = fit_arimax(
        rows.iloc[:75],
        contract=contract,
        candidate=ArimaxCandidate("weather", ("a__weather",), (1, 0, 0)),
        maxiter=30,
    )
    issues = pd.DatetimeIndex(rows["timestamp"].iloc[81:83])
    x, y = x.copy(), y.copy()
    x.loc[x.index[77], "a__weather"] = np.nan
    x = x.drop(x.index[78])
    y.loc[y.index[80], "imputed"] = True
    baseline = rolling_forecasts(model, x, y, issues)
    changed = y.copy()
    changed.loc[changed.index[[78, 79, 80]], "water_level"] += 1e6
    np.testing.assert_allclose(baseline, rolling_forecasts(model, x, changed, issues))
    changed["water_level"] += (
        pd.Series({issues[0]: 5.0}).reindex(changed.index).fillna(0)
    )
    assert not np.allclose(baseline, rolling_forecasts(model, x, changed, issues))


def test_calendar_rollover_and_issue_time_values(
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, rows, x, y = data
    model, _ = fit_arimax(
        rows.iloc[:60],
        contract=contract,
        candidate=ArimaxCandidate("full", contract.predictor_columns, (1, 0, 0)),
        maxiter=20,
    )
    assert isinstance(model, FittedArimax)
    issue = pd.DatetimeIndex([pd.Timestamp("2023-12-31 23:00", tz="UTC")])
    predictions = rolling_forecasts(model, x, y, issue)
    np.testing.assert_allclose(predictions, reference_forecasts(model, x, y, issue))
    # The 24 input hours cross a UTC day boundary and the 2023/2024 year boundary.
    assert issue[0].year == 2023
    before = pd.Timestamp("2023-12-31 23:00", tz="UTC")
    after = before + pd.Timedelta(hours=1)
    assert (
        _calendar(before)["utc_day_of_year_cos"]
        != _calendar(after)["utc_day_of_year_cos"]
    )
    changed = x.copy()
    changed.loc[changed.index > issue[0], "a__weather"] += 10000
    np.testing.assert_allclose(predictions, rolling_forecasts(model, changed, y, issue))


def test_prefixed_calendar_predictor_advances_each_forecast_hour(
    fitted: FittedArimax,
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, x, y = data
    issue = fitted.state_time + pd.Timedelta(hours=1)
    zero_coefficients = np.zeros_like(fitted.coefficients)
    calendar_coefficients = zero_coefficients.copy()
    column = "a__utc_hour_sin"
    column_position = fitted.candidate.feature_columns.index(column)
    calendar_coefficients[column_position + 1] = 2.0  # d=0 has an intercept.
    baseline = rolling_forecasts(
        replace(fitted, coefficients=zero_coefficients), x, y, [issue]
    )
    forecast = rolling_forecasts(
        replace(fitted, coefficients=calendar_coefficients), x, y, [issue]
    )
    hours = pd.date_range(issue, periods=24, freq="h")
    expected_calendar = np.sin(2 * np.pi * hours.hour / 24)
    expected_difference = (
        2.0
        * (expected_calendar - fitted.predictor_scaler.mean_[column_position])
        / fitted.predictor_scaler.scale_[column_position]
        * fitted.target_scaler.scale_[0]
    )
    np.testing.assert_allclose(forecast[0] - baseline[0], expected_difference)


@pytest.mark.parametrize(
    "case", ["early", "missing_predictor", "unordered", "naive", "short_history"]
)
def test_invalid_forecast_inputs(
    case: str,
    fitted: FittedArimax,
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, rows, x, y = data
    x = x.copy()
    issues = pd.DatetimeIndex(rows["timestamp"].iloc[80:82])
    if case == "early":
        issues = pd.DatetimeIndex([fitted.state_time])
    elif case == "missing_predictor":
        x.loc[issues[0], "a__weather"] = np.nan
    elif case == "unordered":
        issues = issues[::-1]
    elif case == "naive":
        issues = issues.tz_localize(None)
    elif case == "short_history":
        x = x.loc[issues[0] :]
    with pytest.raises(ValueError):
        rolling_forecasts(fitted, x, y, issues)


def test_manifest_roundtrip_and_schema_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fitted: FittedArimax,
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, rows, x, y = data
    path, meta = tmp_path / "arimax.joblib", tmp_path / "arimax.json"
    save_arimax_model(fitted, path, meta, execution_uuid="synthetic")
    restored = load_arimax_model(path, meta, contract=contract)
    issues = rows["timestamp"].iloc[80:82]
    np.testing.assert_allclose(
        rolling_forecasts(restored, x, y, issues),
        rolling_forecasts(fitted, x, y, issues),
    )
    manifest = json.loads(meta.read_text())
    assert manifest["schema_version"] == "6.0"
    assert {"training_data_policy", "future_predictor_policy", "trend"} <= set(manifest)
    assert manifest["fitted_order"] == list(fitted.candidate.order)
    assert manifest["coefficient_count"] == len(fitted.coefficients)
    assert len(manifest["target_mean"]) == 1
    assert not any(
        any(
            word in key
            for word in ("cv_", "test_", "converged", "rank", "cohort", "regime")
        )
        for key in manifest
    )
    original = manifest.copy()
    for key, value in (
        ("schema_version", "5.0"),
        ("training_data_policy", "invalid"),
        ("formulation", "direct_regression_with_arima_errors"),
        ("fitted_order", [0, 0, 0]),
        ("input_columns", ["bad"]),
        ("target_scale", [0]),
        ("model_sha256", "bad"),
    ):
        manifest = {**original, key: value}
        meta.write_text(json.dumps(manifest))
        monkeypatch.setattr(
            arimax,
            "load",
            lambda *a, **k: pytest.fail("deserialized before manifest validation"),
        )
        with pytest.raises(ValueError):
            load_arimax_model(path, meta, contract=contract)


def test_grid_cv_selection_and_failed_fold(
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, rows, x, y = data
    grid = candidate_grid({f"subset_{i}": ["a__water_level"] for i in range(6)})
    assert len(grid) == len(set(grid)) == 78
    assert {candidate.order for candidate in grid} == {
        (p, d, q)
        for d in range(2)
        for p in range(3 if d == 0 else 2)
        for q in range(3 if d == 0 else 2)
    }
    candidate = ArimaxCandidate("full", contract.predictor_columns, (0, 0, 0))
    folds = [(np.arange(45), np.arange(50, 55)), (np.arange(60), np.arange(65, 70))]
    evaluation = evaluate_candidate(
        candidate, x, y, rows, folds, contract=contract, maxiter=2
    )
    assert [f["status"] for f in evaluation.fold_details] == ["ok", "ok"]
    assert all("fit_diagnostics" in f for f in evaluation.fold_details)
    assert len(evaluation.horizons) == 48
    assert select_candidate(candidate_table([evaluation])) == ("full", 0, 0, 0)
    bad_x = x.copy()
    bad_x.loc[rows["timestamp"].iloc[65], "a__weather"] = np.nan
    failed = evaluate_candidate(
        candidate, bad_x, y, rows, folds, contract=contract, maxiter=2
    )
    assert [f["status"] for f in failed.fold_details] == ["ok", "failed"]
    assert candidate_table([failed]).empty


def test_nonfinite_fit_rejected(
    data: tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, rows, _, _ = data

    def broken(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.full(args[-1].shape[1], np.nan)

    monkeypatch.setattr(arimax, "_gls_coefficients", broken)
    with pytest.raises(ValueError, match="one-step fit has nonfinite"):
        fit_arimax(
            rows.iloc[:45],
            contract=contract,
            candidate=ArimaxCandidate("full", contract.predictor_columns, (0, 0, 0)),
        )


def _stage3_station(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rain = rng.gamma(0.4, 2.0, size=rows)
    levels = np.empty(rows)
    levels[0] = 150.0
    for i in range(1, rows):
        levels[i] = 30 + 0.8 * levels[i - 1] + 2.0 * rain[i - 1] + rng.normal()
    imputed = np.zeros(rows, dtype=bool)
    imputed[[40, 41, 230]] = True
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-26", periods=rows, freq="h", tz="UTC"),
            "water_level": levels,
            "imputed": imputed,
            "station_id": "a",
            "precipitation": rain.astype("float32"),
            "temperature_2m": (280 + rng.normal(size=rows)).astype("float32"),
        }
    )


def test_recursion_rebuilds_target_features_like_stage3() -> None:
    """Feeding forecasts through the full Stage-3 pipeline gives the same path."""
    station = _stage3_station()
    features = build_feature_frame(station, station_id="a")
    predictors = tuple(f"a__{name}" for name in feature_column_names())
    targets = tuple(f"a__{name}" for name in target_column_names())
    contract = JoinedFeatureContract("a", "a__target_valid", predictors, targets)
    joined = features.drop(columns=["station_id"]).rename(
        columns=lambda c: c if c == "timestamp" else f"a__{c}"
    )
    x = joined.set_index("timestamp")
    y = station.set_index("timestamp")[["water_level", "imputed"]]
    eligible = joined["a__target_valid"] & np.isfinite(
        numeric_predictors(joined, predictors).to_numpy()
    ).all(axis=1)
    cohort = joined.loc[eligible].reset_index(drop=True)
    model, _ = fit_arimax(
        cohort.iloc[:150],
        contract=contract,
        candidate=ArimaxCandidate("target_station_full", predictors, (0, 0, 0)),
    )
    issues = pd.DatetimeIndex(cohort["timestamp"].to_numpy()[[180, 200, 260]])
    actual = rolling_forecasts(model, x, y, issues)

    hour = pd.Timedelta(hours=1)
    recomputed = {f"a__{name}" for name in target_level_features(y[[]], y[[]])}
    recomputed |= {f"a__{name}" for name in utc_calendar_features(y.index.to_series())}
    for row, issue in enumerate(issues):
        issue_x = x.reindex([issue])[list(predictors)].iloc[0].astype(float)
        extended = station.loc[station["timestamp"] <= issue].copy()
        for step in range(24):
            if step:
                future_row = extended.iloc[[-1]].assign(
                    timestamp=issue + step * hour,
                    water_level=actual[row, step - 1],
                    imputed=False,
                )
                extended = pd.concat([extended, future_row], ignore_index=True)
            stage3 = build_feature_frame(extended, station_id="a").iloc[-1]
            future = issue_x.copy()
            for column in recomputed:
                future[column] = float(stage3[column.removeprefix("a__")])
            scaled = model.predictor_scaler.transform(
                future.to_numpy(dtype=float)[None, :]
            )
            expected = float(arimax._design(scaled, (0, 0, 0))[0] @ model.coefficients)
            expected = (
                expected * model.target_scaler.scale_[0] + model.target_scaler.mean_[0]
            )
            np.testing.assert_allclose(actual[row, step], expected, rtol=1e-10)
    # Recursion makes later steps depend on the lag features, not the issue row.
    assert not np.allclose(actual[:, 1:], actual[:, :1])
