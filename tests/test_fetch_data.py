from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
import requests

from src.fetch_data import (
    GEOSPHERE_INCA_URL,
    fetch_hourly_history,
    fetch_inca,
    flatten_station_catalog,
    hourly_chunks,
    resolve_station_coordinates,
)


class StubPegelApi:
    def __init__(self, frames: list[pd.DataFrame]):
        self.frames = iter(frames)
        self.calls: list[tuple[datetime, datetime]] = []

    def query_historic_data(
        self,
        commonid: str,
        start: datetime,
        end: datetime,
        *,
        unit: str,
        granularity: str,
    ) -> pd.DataFrame:
        self.calls.append((start, end))
        return next(self.frames)


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


class FakeGeoSphereClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = iter(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        return next(self.responses)


INCA_CSV = """time,RR [kg m-2],T2M [degree_Celsius],lat,lon
2024-01-01T00:00+00:00,0.0,1.4,47.10029,14.20232
2024-01-01T01:00+00:00,1.8,1.85,47.10029,14.20232
2024-01-01T02:00+00:00,2.2,1.9,47.10029,14.20232
"""


def test_hourly_chunks_are_non_overlapping_and_at_most_eight_days() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 10, tzinfo=UTC)

    assert list(hourly_chunks(start, end)) == [
        (start, datetime(2024, 1, 8, 23, tzinfo=UTC)),
        (datetime(2024, 1, 9, tzinfo=UTC), end),
    ]


def test_fetch_hourly_history_ignores_empty_chunks_sorts_and_deduplicates() -> None:
    api = StubPegelApi(
        [
            pd.DataFrame(
                {
                    "sourceDate": pd.to_datetime(
                        ["2024-01-01T02:00Z", "2024-01-01T01:00Z"], utc=True
                    ),
                    "value": [2.0, 1.0],
                }
            ),
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "sourceDate": pd.to_datetime(
                        ["2024-01-01T02:00Z", "2024-01-17T00:00Z"], utc=True
                    ),
                    "value": [20.0, 17.0],
                }
            ),
        ]
    )

    result = fetch_hourly_history(
        api,  # type: ignore[arg-type]
        "station-at",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 17, tzinfo=UTC),
        unit="height",
        granularity="hour",
        show_progress=False,
    )

    assert result["sourceDate"].tolist() == list(
        pd.to_datetime(
            ["2024-01-01T01:00Z", "2024-01-01T02:00Z", "2024-01-17T00:00Z"], utc=True
        )
    )
    assert result["value"].tolist() == [1.0, 20.0, 17.0]
    assert result["station_id"].unique().tolist() == ["station-at"]


def test_fetch_inca_requests_native_parameters_and_maps_csv_columns() -> None:
    client = FakeGeoSphereClient([FakeResponse(INCA_CSV)])

    result = fetch_inca(
        "station-at",
        47.0,
        14.0,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
        client=client,
    )

    url, kwargs = client.calls[0]
    assert url == GEOSPHERE_INCA_URL
    assert kwargs["params"] == [
        ("parameters", "RR"),
        ("parameters", "T2M"),
        ("start", "2024-01-01T00:00"),
        ("end", "2024-01-01T02:00"),
        ("lat_lon", "47.0,14.0"),
        ("output_format", "csv"),
    ]
    assert kwargs["timeout"] == 30
    assert result.columns.tolist() == [
        "time",
        "precipitation",
        "temperature_2m",
        "station_id",
        "requested_latitude",
        "requested_longitude",
        "grid_latitude",
        "grid_longitude",
        "weather_model",
    ]
    assert result["time"].tolist() == list(
        pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    )
    assert result["grid_latitude"].unique().tolist() == [47.10029]
    assert result["grid_longitude"].unique().tolist() == [14.20232]
    assert result["weather_model"].unique().tolist() == ["inca-v1-1h-1km"]
    assert "grid_elevation" not in result


def test_fetch_inca_deduplicates_inclusive_response_timestamps() -> None:
    csv = INCA_CSV + "2024-01-01T02:00+00:00,9.0,9.0,47.10029,14.20232\n"

    result = fetch_inca(
        "station-at",
        47.0,
        14.0,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
        client=FakeGeoSphereClient([FakeResponse(csv)]),
    )

    assert result["precipitation"].tolist() == [0.0, 1.8, 9.0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            FakeResponse("time,RR [kg m-2],lat,lon\n2024-01-01T00:00Z,1,47,14\n"),
            "missing columns",
        ),
        (
            FakeResponse(
                "time,RR [kg m-2],T2M [degree_Celsius],lat,lon\nnot-a-time,1,2,47,14\n"
            ),
            "invalid timestamps",
        ),
        (
            FakeResponse(
                "time,RR [kg m-2],T2M [degree_Celsius],lat,lon\n2024-01-01T00:00Z,invalid,2,47,14\n"
            ),
            "invalid weather values",
        ),
        (
            FakeResponse("time,RR [kg m-2],T2M [degree_Celsius],lat,lon\n"),
            "No INCA history",
        ),
        (FakeResponse("error", status_code=503), "HTTP 503"),
    ],
)
def test_fetch_inca_rejects_invalid_responses(
    response: FakeResponse, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        fetch_inca(
            "station-at",
            47.0,
            14.0,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            client=FakeGeoSphereClient([response]),
        )


def test_fetch_inca_rejects_no_overlap_and_multiple_grid_points() -> None:
    no_overlap = (
        "time,RR [kg m-2],T2M [degree_Celsius],lat,lon\n2023-12-31T23:00Z,1,2,47,14\n"
    )
    multiple_grid_points = INCA_CSV.replace("47.10029,14.20232", "47.20029,14.20232", 1)

    with pytest.raises(RuntimeError, match="water-data range"):
        fetch_inca(
            "station-at",
            47.0,
            14.0,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            client=FakeGeoSphereClient([FakeResponse(no_overlap)]),
        )
    with pytest.raises(RuntimeError, match="multiple grid coordinates"):
        fetch_inca(
            "station-at",
            47.0,
            14.0,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            client=FakeGeoSphereClient([FakeResponse(multiple_grid_points)]),
        )


def test_fetch_inca_wraps_request_exceptions() -> None:
    class FailingGeoSphereClient:
        def get(self, url: str, **kwargs: Any) -> None:
            raise requests.ConnectionError("unavailable")

    with pytest.raises(RuntimeError, match="GeoSphere INCA request failed"):
        fetch_inca(
            "station-at",
            47.0,
            14.0,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            client=FailingGeoSphereClient(),  # type: ignore[arg-type]
        )


def test_catalog_flattens_scalars_and_resolves_coordinates() -> None:
    catalog = flatten_station_catalog(
        [
            {
                "commonid": "207241-at",
                "location": {"latitude": 47.1, "longitude": 14.2},
                "measurements": [{"value": 1.0}],
            }
        ]
    )

    assert "measurements" not in catalog
    assert resolve_station_coordinates(catalog, "207241-at") == (47.1, 14.2)
