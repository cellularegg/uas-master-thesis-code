"""Fetch raw PegelAlarm and GeoSphere INCA weather data to local parquet files."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from io import StringIO
from typing import Any, Protocol

import pandas as pd
import requests
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from .basic_api_access import REQUEST_TIMEOUT_SECONDS, BasicApiAccess
from .config import INCA_DATASET_ID

GEOSPHERE_INCA_URL = (
    f"https://dataset.api.hub.geosphere.at/v1/timeseries/historical/{INCA_DATASET_ID}"
)
INCA_NATIVE_COLUMNS = {
    "RR [kg m-2]": "precipitation",
    "T2M [degree_Celsius]": "temperature_2m",
}
HOUR = timedelta(hours=1)
HISTORY_CHUNK = timedelta(days=8)


class GeoSphereClient(Protocol):
    """Protocol for the small subset of the requests client used by INCA."""

    def get(self, url: str, **kwargs: Any) -> Any:
        """Fetch a CSV response for the given URL and query parameters."""
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


def fetch_inca(
    station_id: str,
    latitude: float,
    longitude: float,
    start: datetime,
    end: datetime,
    *,
    client: GeoSphereClient | None = None,
) -> pd.DataFrame:
    """Fetch hourly INCA analysis data at the nearest grid point to a gauge.

    The GeoSphere API uses inclusive request bounds; duplicate timestamps are
    therefore removed after parsing before the requested UTC interval is applied.
    Missing weather values are preserved as ``NaN``; non-empty non-numeric
    weather values are rejected.
    """
    start_timestamp = _utc_timestamp(start, name="start")
    end_timestamp = _utc_timestamp(end, name="end")
    if start_timestamp > end_timestamp:
        raise ValueError("INCA history start must not be after its end")

    parameters: list[tuple[str, str]] = [
        ("parameters", "RR"),
        ("parameters", "T2M"),
        ("start", start_timestamp.strftime("%Y-%m-%dT%H:%M")),
        ("end", end_timestamp.strftime("%Y-%m-%dT%H:%M")),
        ("lat_lon", f"{latitude},{longitude}"),
        ("output_format", "csv"),
    ]
    http_client = client if client is not None else requests
    try:
        response = http_client.get(
            GEOSPHERE_INCA_URL, params=parameters, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as error:
        raise RuntimeError("GeoSphere INCA request failed") from error
    if response.status_code >= 400:
        raise RuntimeError(
            f"GeoSphere INCA request failed with HTTP {response.status_code}"
        )

    try:
        weather = pd.read_csv(StringIO(response.text))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RuntimeError("GeoSphere INCA response is not valid CSV") from error
    if weather.empty:
        raise RuntimeError(f"No INCA history found for station {station_id}")
    required_columns = {"time", "lat", "lon", *INCA_NATIVE_COLUMNS}
    missing = sorted(required_columns.difference(weather.columns))
    if missing:
        raise RuntimeError(f"GeoSphere INCA response is missing columns: {missing}")

    weather["time"] = pd.to_datetime(weather["time"], utc=True, errors="coerce")
    if weather["time"].isna().any():
        raise RuntimeError("GeoSphere INCA response contains invalid timestamps")
    invalid_weather = pd.Series(False, index=weather.index)
    for native_column in INCA_NATIVE_COLUMNS:
        raw_values = weather[native_column]
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        invalid_weather |= raw_values.notna() & numeric_values.isna()
        weather[native_column] = numeric_values
    if invalid_weather.any():
        raise RuntimeError("GeoSphere INCA response contains invalid weather values")
    for coordinate_column in ("lat", "lon"):
        weather[coordinate_column] = pd.to_numeric(
            weather[coordinate_column], errors="coerce"
        )
    if weather[["lat", "lon"]].isna().any().any():
        raise RuntimeError("GeoSphere INCA response contains invalid grid coordinates")
    grid_coordinates = weather[["lat", "lon"]].drop_duplicates()
    if len(grid_coordinates) != 1:
        raise RuntimeError("GeoSphere INCA response contains multiple grid coordinates")

    weather = weather.rename(columns=INCA_NATIVE_COLUMNS)
    weather = weather.loc[
        weather["time"].between(start_timestamp, end_timestamp),
        ["time", *INCA_NATIVE_COLUMNS.values()],
    ]
    if weather.empty:
        raise RuntimeError(
            f"No INCA history found in the water-data range for station {station_id}"
        )
    weather = (
        weather.sort_values("time")
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )
    weather["station_id"] = station_id
    weather["requested_latitude"] = latitude
    weather["requested_longitude"] = longitude
    weather["grid_latitude"] = float(grid_coordinates.iloc[0]["lat"])
    weather["grid_longitude"] = float(grid_coordinates.iloc[0]["lon"])
    weather["weather_model"] = INCA_DATASET_ID
    return weather


def _utc_timestamp(value: datetime, *, name: str) -> pd.Timestamp:
    """Convert a timezone-aware boundary to UTC, rejecting naive datetimes."""
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError(f"INCA history {name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


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
