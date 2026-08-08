from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import requests

from src.basic_api_access import REQUEST_TIMEOUT_SECONDS, BasicApiAccess


class FakeResponse:
    def __init__(
        self, body: Any, *, url: str = "https://example.test", status: int = 200
    ):
        self._body = body
        self.url = url
        self.status_code = status

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def api() -> BasicApiAccess:
    instance = object.__new__(BasicApiAccess)
    instance.xAuthToken = "test-token"
    return instance


def test_query_current_data_filters_by_country(
    monkeypatch: pytest.MonkeyPatch, api: BasicApiAccess
) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        captured.update(url=url, **kwargs)
        return FakeResponse(
            {"status": {"code": 200}, "payload": [{"commonid": "x-at"}]}
        )

    monkeypatch.setattr(requests, "get", fake_get)

    assert api.query_current_data(country_code="AT") == [{"commonid": "x-at"}]
    assert captured["params"]["countryCode"] == "AT"
    assert captured["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_history_boundaries_are_converted_to_utc_in_summer(
    monkeypatch: pytest.MonkeyPatch, api: BasicApiAccess
) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        captured.update(url=url, **kwargs)
        return FakeResponse(
            {"status": {"code": 200}, "payload": {"history": []}}, url=url
        )

    monkeypatch.setattr(requests, "get", fake_get)
    vienna = ZoneInfo("Europe/Vienna")

    result = api.query_historic_data(
        "207241-at",
        datetime(2024, 7, 1, 2, tzinfo=vienna),
        datetime(2024, 7, 1, 10, tzinfo=vienna),
    )

    assert result.empty
    assert captured["params"]["loadStartDate"] == "01.07.2024T00:00:00+0000"
    assert captured["params"]["loadEndDate"] == "01.07.2024T08:00:00+0000"


def test_history_normalizes_mixed_offsets_to_utc(
    monkeypatch: pytest.MonkeyPatch, api: BasicApiAccess
) -> None:
    body = {
        "status": {"code": 200},
        "payload": {
            "history": [
                {"sourceDate": "31.03.2024T01:00:00+0100", "value": 1.0},
                {"sourceDate": "31.03.2024T03:00:00+0200", "value": 2.0},
            ]
        },
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse(body))

    result = api.query_historic_data(
        "207241-at",
        datetime(2024, 3, 31, tzinfo=UTC),
        datetime(2024, 4, 1, tzinfo=UTC),
    )

    assert isinstance(result["sourceDate"].dtype, pd.DatetimeTZDtype)
    assert str(result["sourceDate"].dt.tz) == "UTC"
    assert result["sourceDate"].tolist() == [
        pd.Timestamp("2024-03-31T00:00:00Z"),
        pd.Timestamp("2024-03-31T01:00:00Z"),
    ]


def test_history_requires_aware_boundaries(api: BasicApiAccess) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        api.query_historic_data(
            "207241-at",
            datetime(2024, 1, 1),  # noqa: DTZ001 - naive datetime is the point of this test
            datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_api_error_is_not_treated_as_empty_history(
    monkeypatch: pytest.MonkeyPatch, api: BasicApiAccess
) -> None:
    body = {"status": {"code": 401, "message": "invalid token"}, "payload": {}}
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse(body))

    with pytest.raises(RuntimeError, match="invalid token"):
        api.query_historic_data(
            "207241-at",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )
