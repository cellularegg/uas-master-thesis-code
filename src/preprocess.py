"""Clean raw water-level series and merge them with weather data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def clean_water_level(raw: pd.DataFrame, *, max_gap_hours: int) -> pd.DataFrame:
    """Reindex a station's water-level history to a strict hourly UTC grid.

    Gaps of at most ``max_gap_hours`` consecutive missing hours are linearly
    interpolated and flagged via the ``imputed`` column; longer gaps are left
    as ``NaN`` and not flagged.

    Args:
        raw: Raw PegelAlarm history with ``sourceDate``, ``value``, and
            ``station_id`` columns.
        max_gap_hours: Maximum length, in hours, of a gap that gets
            interpolated.

    Returns:
        DataFrame indexed by hourly UTC ``timestamp`` with ``water_level``,
        ``imputed``, and ``station_id`` columns.
    """
    station_id = raw["station_id"].iloc[0]
    series = raw.set_index("sourceDate")["value"].sort_index()
    grid = pd.date_range(
        series.index.min(), series.index.max(), freq="h", name="timestamp"
    )
    series = series.reindex(grid)

    missing = series.isna()
    gap_id = (missing != missing.shift()).cumsum()
    gap_length = missing.groupby(gap_id).transform("size").where(missing, 0)
    interpolated = series.interpolate(
        method="time", limit_area="inside", limit_direction="forward"
    )
    filled = series.where(gap_length > max_gap_hours, interpolated)
    imputed = missing & filled.notna()

    return pd.DataFrame(
        {
            "water_level": filled,
            "imputed": imputed,
            "station_id": station_id,
        }
    )


def merge_weather(
    water: pd.DataFrame, weather: pd.DataFrame, *, variables: Sequence[str]
) -> pd.DataFrame:
    """Left-join weather variables onto a water-level hourly grid.

    Args:
        water: Output of :func:`clean_water_level`, indexed by ``timestamp``.
        weather: Raw ERA5 weather history with a ``time`` column plus
            ``variables`` and other metadata columns.
        variables: Weather columns to keep.

    Returns:
        ``water`` with the requested weather columns appended.
    """
    trimmed = weather.set_index("time")[list(variables)]
    trimmed.index = trimmed.index.rename("timestamp")
    return water.join(trimmed, how="left")


def preprocess_station(
    station_id: str,
    *,
    raw_dir: Path,
    max_gap_hours: int,
    weather_variables: Sequence[str],
) -> pd.DataFrame:
    """Load a station's raw parquet files and build its analysis-ready frame.

    Args:
        station_id: PegelAlarm station identifier.
        raw_dir: Directory containing the raw parquet files from
            `01_fetch_data.ipynb`.
        max_gap_hours: Maximum length, in hours, of a water-level gap that
            gets interpolated.
        weather_variables: Weather columns to keep.

    Returns:
        Merged, hourly, analysis-ready DataFrame for the station.
    """
    water_raw = pd.read_parquet(
        raw_dir / f"pegelalarm_{station_id}_height_hour.parquet"
    )
    weather_raw = pd.read_parquet(raw_dir / f"openmeteo_{station_id}_era5_hour.parquet")

    water = clean_water_level(water_raw, max_gap_hours=max_gap_hours)
    return merge_weather(water, weather_raw, variables=weather_variables).reset_index()
