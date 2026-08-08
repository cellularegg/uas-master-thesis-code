from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from src.fetch_data import (
    fetch_era5,
    fetch_hourly_history,
    flatten_station_catalog,
    hourly_chunks,
    resolve_station_coordinates,
    yearly_date_chunks,
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


class FakeHourlyVariable:
    def __init__(self, values: list[float | None]):
        self.values = values

    def ValuesAsNumpy(self) -> list[float | None]:
        return self.values


class FakeHourly:
    def __init__(self, start: int, end: int, variables: list[list[float | None]]):
        self.start = start
        self.end = end
        self.variables = [FakeHourlyVariable(values) for values in variables]

    def Time(self) -> int:
        return self.start

    def TimeEnd(self) -> int:
        return self.end

    def Interval(self) -> int:
        return 3600

    def VariablesLength(self) -> int:
        return len(self.variables)

    def Variables(self, index: int) -> FakeHourlyVariable:
        return self.variables[index]


class FakeWeatherResponse:
    def __init__(self, hourly: FakeHourly | None):
        self.hourly = hourly

    def Hourly(self) -> FakeHourly | None:
        return self.hourly

    def Latitude(self) -> float:
        return 47.1

    def Longitude(self) -> float:
        return 14.2

    def Elevation(self) -> float:
        return 500.0


class FakeOpenMeteoClient:
    def __init__(self, responses: list[FakeWeatherResponse]):
        self.responses = iter(responses)
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def weather_api(
        self, url: str, params: dict[str, Any], **kwargs: Any
    ) -> list[FakeWeatherResponse]:
        self.calls.append((url, params, kwargs))
        return [next(self.responses)]


def test_hourly_chunks_are_non_overlapping_and_at_most_eight_days() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 10, tzinfo=UTC)

    chunks = list(hourly_chunks(start, end))

    assert chunks == [
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


def test_yearly_chunks_do_not_overlap() -> None:
    assert list(
        yearly_date_chunks(
            datetime(2023, 12, 30, 2, tzinfo=UTC),
            datetime(2025, 1, 2, 3, tzinfo=UTC),
        )
    ) == [
        ("2023-12-30", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-01-02"),
    ]


def test_fetch_era5_uses_utc_variables_chunks_and_coordinate_metadata() -> None:
    client = FakeOpenMeteoClient(
        [
            FakeWeatherResponse(
                FakeHourly(
                    1704060000,
                    1704067200,
                    [[0.0, 1.0], [None, 2.0]],
                )
            ),
            FakeWeatherResponse(
                FakeHourly(
                    1704067200,
                    1704078000,
                    [[2.0, 3.0, None], [3.0, 4.0, None]],
                )
            ),
        ]
    )
    result = fetch_era5(
        "station-at",
        47.0,
        14.0,
        datetime(2023, 12, 31, 23, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
        variables=["precipitation", "temperature_2m"],
        model="era5",
        client=client,
        show_progress=False,
    )

    assert [(call[1]["start_date"], call[1]["end_date"]) for call in client.calls] == [
        ("2023-12-31", "2023-12-31"),
        ("2024-01-01", "2024-01-01"),
    ]
    assert all(call[1]["timezone"] == "GMT" for call in client.calls)
    assert all(call[1]["models"] == "era5" for call in client.calls)
    assert all(
        call[1]["hourly"] == ["precipitation", "temperature_2m"]
        for call in client.calls
    )
    assert all(call[2]["timeout"] == 30 for call in client.calls)
    assert all(
        call[0] == "https://archive-api.open-meteo.com/v1/archive"
        for call in client.calls
    )
    assert result["time"].tolist() == list(
        pd.to_datetime(
            ["2023-12-31T23:00Z", "2024-01-01T00:00Z", "2024-01-01T01:00Z"], utc=True
        )
    )
    assert result["requested_latitude"].unique().tolist() == [47.0]
    assert result["requested_longitude"].unique().tolist() == [14.0]
    assert result["grid_latitude"].unique().tolist() == [47.1]


def test_fetch_era5_uses_default_openmeteo_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeOpenMeteoClient(
        [FakeWeatherResponse(FakeHourly(1704067200, 1704070800, [[1.0]]))]
    )
    monkeypatch.setattr("src.fetch_data.openmeteo_requests.Client", lambda: client)

    result = fetch_era5(
        "station-at",
        47.0,
        14.0,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        variables=["precipitation"],
        model="era5",
        show_progress=False,
    )

    assert len(result) == 1
    assert len(client.calls) == 1


def test_fetch_era5_rejects_missing_hourly_variables() -> None:
    client = FakeOpenMeteoClient(
        [FakeWeatherResponse(FakeHourly(1704067200, 1704070800, [[1.0]]))]
    )

    with pytest.raises(RuntimeError, match="temperature_2m"):
        fetch_era5(
            "station-at",
            47.0,
            14.0,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            variables=["precipitation", "temperature_2m"],
            model="era5",
            client=client,
            show_progress=False,
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
