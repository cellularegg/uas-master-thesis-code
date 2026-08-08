"""Fetch raw PegelAlarm and ERA5 weather data to local parquet files."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import openmeteo_requests
import pandas as pd
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from .basic_api_access import REQUEST_TIMEOUT_SECONDS, BasicApiAccess

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOUR = timedelta(hours=1)
HISTORY_CHUNK = timedelta(days=8)


class OpenMeteoClient(Protocol):
    """Protocol for an Open-Meteo API client, satisfied by openmeteo_requests.Client."""

    def weather_api(self, url: str, params: Any, **kwargs: Any) -> Sequence[Any]:
        """Fetch weather data for the given URL and parameters."""
        ...


def utc_current_hour() -> datetime:
    """Return the current UTC time truncated to the top of the hour."""
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def flatten_station_catalog(payload: Any) -> pd.DataFrame:
    """Flatten station records and retain only scalar-valued metadata."""
    if isinstance(payload, dict):
        for key in ("stations", "station"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError("PegelAlarm station payload is not a list")

    catalog = pd.json_normalize(payload, sep="_")
    scalar_columns = [
        column
        for column in catalog.columns
        if not catalog[column].map(lambda value: isinstance(value, (dict, list))).any()
    ]
    return catalog.loc[:, scalar_columns]


def hourly_chunks(
    start: datetime, end: datetime
) -> Iterator[tuple[datetime, datetime]]:
    """Yield inclusive, non-overlapping requests containing at most 192 hours."""
    current = start
    while current <= end:
        chunk_end = min(current + HISTORY_CHUNK - HOUR, end)
        yield current, chunk_end
        current = chunk_end + HOUR


def find_archive_start(
    api: BasicApiAccess,
    station_id: str,
    end: datetime,
    *,
    unit: str,
) -> datetime:
    """Find the earliest available timestamp in a station's PegelAlarm archive."""
    broad_start = datetime(1900, 1, 1, tzinfo=UTC)
    yearly = api.query_historic_data(
        station_id, broad_start, end, unit=unit, granularity="year"
    )
    if yearly.empty:
        raise RuntimeError(f"No PegelAlarm archive found for station {station_id}")
    if "sourceDate" not in yearly:
        raise RuntimeError(
            f"Yearly PegelAlarm archive for {station_id} has no sourceDate"
        )
    return yearly["sourceDate"].min().to_pydatetime().astimezone(UTC)


def fetch_hourly_history(
    api: BasicApiAccess,
    station_id: str,
    start: datetime,
    end: datetime,
    *,
    unit: str,
    granularity: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch and concatenate hourly PegelAlarm history in chunks."""
    chunks = list(hourly_chunks(start, end))
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in tqdm(
        chunks, desc=f"PegelAlarm {station_id}", unit="chunk", disable=not show_progress
    ):
        frame = api.query_historic_data(
            station_id,
            chunk_start,
            chunk_end,
            unit=unit,
            granularity=granularity,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(
            f"No hourly PegelAlarm history found for station {station_id}"
        )

    history = pd.concat(frames, ignore_index=True)
    history["sourceDate"] = pd.to_datetime(history["sourceDate"], utc=True)
    history = (
        history.sort_values("sourceDate")
        .drop_duplicates(subset="sourceDate", keep="last")
        .reset_index(drop=True)
    )
    history["station_id"] = station_id
    return history


def yearly_date_chunks(start: datetime, end: datetime) -> Iterator[tuple[str, str]]:
    """Yield inclusive, non-overlapping calendar-year date ranges as ISO strings."""
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp.tz is None or end_timestamp.tz is None:
        raise ValueError("Open-Meteo history boundaries must be timezone-aware")
    start_timestamp = start_timestamp.tz_convert("UTC")
    end_timestamp = end_timestamp.tz_convert("UTC")
    if start_timestamp > end_timestamp:
        raise ValueError("Open-Meteo history start must not be after its end")

    current = start_timestamp.date()
    final = end_timestamp.date()
    while current <= final:
        chunk_end = min(current.replace(month=12, day=31), final)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def fetch_era5(
    station_id: str,
    latitude: float,
    longitude: float,
    start: datetime,
    end: datetime,
    *,
    variables: Sequence[str],
    model: str,
    client: OpenMeteoClient | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch and concatenate ERA5 hourly weather history in yearly chunks."""
    requested_variables = list(variables)
    openmeteo_client = client if client is not None else openmeteo_requests.Client()
    frames: list[pd.DataFrame] = []
    chunks = list(yearly_date_chunks(start, end))
    for start_date, end_date in tqdm(
        chunks, desc=f"ERA5 {station_id}", unit="year", disable=not show_progress
    ):
        parameters = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": requested_variables,
            "models": model,
            "timezone": "GMT",
        }
        responses = openmeteo_client.weather_api(
            OPEN_METEO_ARCHIVE_URL,
            params=parameters,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not responses:
            raise RuntimeError("Open-Meteo response has no weather data")

        response = responses[0]
        hourly = response.Hourly()
        if hourly is None:
            raise RuntimeError("Open-Meteo response has no hourly time series")
        variable_count = hourly.VariablesLength()
        if variable_count < len(requested_variables):
            missing = ", ".join(requested_variables[variable_count:])
            raise RuntimeError(
                f"Open-Meteo response is missing requested variables: {missing}"
            )

        hourly_data: dict[str, Any] = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.to_timedelta(hourly.Interval(), unit="s"),
                inclusive="left",
            )
        }
        for index, variable in enumerate(requested_variables):
            hourly_data[variable] = hourly.Variables(index).ValuesAsNumpy()

        frame = pd.DataFrame(hourly_data)
        if frame.empty:
            continue
        frame = frame.loc[frame[requested_variables].notna().any(axis=1)].copy()
        if frame.empty:
            continue
        frame["station_id"] = station_id
        frame["requested_latitude"] = latitude
        frame["requested_longitude"] = longitude
        frame["grid_latitude"] = response.Latitude()
        frame["grid_longitude"] = response.Longitude()
        frame["grid_elevation"] = response.Elevation()
        frame["weather_model"] = model
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No ERA5 history found for station {station_id}")
    weather = pd.concat(frames, ignore_index=True)
    start_timestamp = pd.Timestamp(start).tz_convert("UTC")
    end_timestamp = pd.Timestamp(end).tz_convert("UTC")
    weather = weather.loc[weather["time"].between(start_timestamp, end_timestamp)]
    if weather.empty:
        raise RuntimeError(
            f"No ERA5 history found in the water-data range for station {station_id}"
        )
    return (
        weather.sort_values("time")
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )


def resolve_station_coordinates(
    catalog: pd.DataFrame, station_id: str
) -> tuple[float, float]:
    """Look up a station's latitude and longitude in the station catalog."""
    id_column = _first_column(catalog, ("commonid", "commonId", "common_id"))
    match = catalog.loc[
        catalog[id_column].astype(str).str.casefold() == station_id.casefold()
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Expected exactly one metadata record for station {station_id}, found {len(match)}"
        )
    latitude_column = _first_column(
        catalog, ("latitude", "lat", "location_latitude", "position_latitude")
    )
    longitude_column = _first_column(
        catalog,
        ("longitude", "lon", "lng", "location_longitude", "position_longitude"),
    )
    return float(match.iloc[0][latitude_column]), float(match.iloc[0][longitude_column])


def write_parquet_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame to a parquet file via a temp file and atomic rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def summarize_failures(failures: Mapping[str, BaseException]) -> str:
    """Format a summary message for a mapping of artifact names to failures."""
    details = "; ".join(f"{artifact}: {error}" for artifact, error in failures.items())
    return f"Raw-data fetch completed with {len(failures)} failure(s): {details}"


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    by_casefold = {column.casefold(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in by_casefold:
            return by_casefold[candidate.casefold()]
    raise RuntimeError(
        f"None of the expected metadata columns exist: {', '.join(candidates)}"
    )
