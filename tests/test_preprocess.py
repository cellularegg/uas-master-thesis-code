import numpy as np
import pandas as pd

from src.preprocess import clean_water_level, merge_weather


def _raw_water(values: list[float | None]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "sourceDate": index,
            "value": values,
            "station_id": "station-at",
        }
    )


def test_clean_water_level_reindexes_to_a_strict_hourly_grid() -> None:
    raw = _raw_water([1.0, 2.0, 3.0])
    raw = raw.drop(index=1)  # drop the middle timestamp entirely

    result = clean_water_level(raw, max_gap_hours=6)

    assert (
        result.index.tolist()
        == pd.date_range("2024-01-01T00:00", periods=3, freq="h", tz="UTC").tolist()
    )


def test_clean_water_level_interpolates_short_gaps_and_flags_them() -> None:
    values: list[float | None] = [1.0, np.nan, np.nan, np.nan, 5.0, 6.0]
    raw = _raw_water(values)

    result = clean_water_level(raw, max_gap_hours=3)

    assert result["water_level"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["imputed"].tolist() == [False, True, True, True, False, False]


def test_clean_water_level_leaves_long_gaps_as_nan_and_unflagged() -> None:
    values: list[float | None] = [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]
    raw = _raw_water(values)

    result = clean_water_level(raw, max_gap_hours=3)

    water_level = result["water_level"].tolist()
    assert water_level[0] == 1.0
    assert water_level[-1] == 6.0
    assert all(np.isnan(value) for value in water_level[1:-1])
    assert result["imputed"].tolist() == [False, False, False, False, False, False]
    assert result["station_id"].unique().tolist() == ["station-at"]


def test_merge_weather_keeps_only_requested_variables_and_left_joins() -> None:
    water_index = pd.date_range("2024-01-01T00:00", periods=3, freq="h", tz="UTC")
    water = pd.DataFrame(
        {"water_level": [1.0, 2.0, 3.0], "imputed": False, "station_id": "station-at"},
        index=water_index,
    )
    water.index = water.index.rename("timestamp")

    weather_index = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T02:00Z"], utc=True)
    weather = pd.DataFrame(
        {
            "time": weather_index,
            "temperature_2m": [10.0, 12.0],
            "precipitation": [0.0, 1.0],
            "weather_model": "era5",
            "requested_latitude": 47.0,
        }
    )

    result = merge_weather(
        water, weather, variables=["temperature_2m", "precipitation"]
    )

    assert list(result.columns) == [
        "water_level",
        "imputed",
        "station_id",
        "temperature_2m",
        "precipitation",
    ]
    assert np.allclose(
        result["temperature_2m"].tolist(), [10.0, np.nan, 12.0], equal_nan=True
    )
    assert np.allclose(
        result["precipitation"].tolist(), [0.0, np.nan, 1.0], equal_nan=True
    )
