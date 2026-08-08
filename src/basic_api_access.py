"""Client for the PegelAlarm REST API."""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from . import auth_token_generator as alg

REQUEST_TIMEOUT_SECONDS = 30


class BasicApiAccess:
    """Authenticate against and query the PegelAlarm API."""

    xAuthToken: str | None = None

    def __init__(self, usr: str, pwd: str):
        """Log in with the given credentials and store the X-AUTH-TOKEN."""
        credentials = {"username": usr, "password": pwd}
        self.xAuthToken = self.get_xauth_token(credentials)

    def get_xauth_token(self, credentials: dict[str, str]) -> str:
        """Fetch an API key and exchange it for an X-AUTH-TOKEN."""
        api_key = self.get_api_key(credentials)
        x_auth_token = alg.AuthTokenGenerator().calc_xauth_token(
            credentials["username"], api_key
        )
        return x_auth_token

    @staticmethod
    def get_api_key(credentials_int: dict[str, str]) -> str:
        """Log in to PegelAlarm and return the API key."""
        url = "https://api.pegelalarm.at/api/login"
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(credentials_int),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = BasicApiAccess._json_body(response)
        BasicApiAccess._raise_for_api_error(body, url)
        try:
            api_key = body["payload"]["apiKey"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"PegelAlarm login response from {url} has no API key"
            ) from exc
        if not isinstance(api_key, str) or not api_key:
            raise RuntimeError(
                f"PegelAlarm login response from {url} has no valid API key"
            )
        return api_key

    def query_current_data(
        self,
        station_name: str = "",
        water_name: str = "",
        common_id: str = "",
        country_code: str = "",
    ) -> Any:
        """Query the current-data station list, optionally filtered."""
        assert self.xAuthToken is not None
        parameters = {
            "qStationName": station_name,
            "qWater": water_name,
            "commonid": common_id,
            "countryCode": country_code,
        }
        headers = {"Content-Type": "application/json", "X-AUTH-TOKEN": self.xAuthToken}
        response = requests.get(
            "https://api.pegelalarm.at/api/station/1.1/list",
            params=parameters,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = self._json_body(response)
        self._raise_for_api_error(body, response.url)
        if "payload" not in body:
            raise RuntimeError(
                f"PegelAlarm response from {response.url} has no payload"
            )
        return body["payload"]

    # Possible values for 'unit': "height" and "flow"
    # Possible values for 'granularity': "raw" (for recent 3 months of data), "hour", "day", "month", "year" and "era"
    def query_historic_data(
        self,
        commonid: str,
        load_start_date_utc: datetime,
        load_end_date_utc: datetime,
        unit: str = "height",
        granularity: str = "hour",
    ) -> pd.DataFrame:
        """Query historic station data for a date range as a DataFrame."""
        assert self.xAuthToken is not None
        if (
            load_start_date_utc.utcoffset() is None
            or load_end_date_utc.utcoffset() is None
        ):
            raise ValueError("PegelAlarm history boundaries must be timezone-aware")
        start_utc = load_start_date_utc.astimezone(UTC)
        end_utc = load_end_date_utc.astimezone(UTC)
        if start_utc > end_utc:
            raise ValueError("PegelAlarm history start must not be after its end")

        url = f"https://api.pegelalarm.at/api/station/1.1/{unit}/{commonid}/history"
        parameters = {
            "loadStartDate": start_utc.strftime("%d.%m.%YT%H:%M:%S%z"),
            "loadEndDate": end_utc.strftime("%d.%m.%YT%H:%M:%S%z"),
            "granularity": granularity,
        }
        headers = {"Content-Type": "application/json", "X-AUTH-TOKEN": self.xAuthToken}
        response = requests.get(
            url,
            params=parameters,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = self._json_body(response)
        self._raise_for_api_error(body, response.url)
        try:
            history = body["payload"]["history"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"PegelAlarm response from {response.url} has no history"
            ) from exc
        if history is None:
            history = []
        if not isinstance(history, list):
            raise TypeError(f"PegelAlarm history from {response.url} is not a list")

        history_df = pd.DataFrame(history)
        if history_df.empty:
            return history_df
        if "sourceDate" not in history_df:
            raise RuntimeError(
                f"PegelAlarm history from {response.url} has no sourceDate"
            )
        history_df["sourceDate"] = pd.to_datetime(
            history_df["sourceDate"], format="%d.%m.%YT%H:%M:%S%z", utc=True
        )
        return history_df

    @staticmethod
    def _json_body(response: requests.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError(
                f"PegelAlarm returned invalid JSON from {response.url}"
            ) from exc
        if not isinstance(body, dict):
            raise TypeError(
                f"PegelAlarm returned a non-object response from {response.url}"
            )
        return body

    @staticmethod
    def _raise_for_api_error(body: dict[str, Any], url: str) -> None:
        status = body.get("status")
        if not isinstance(status, dict):
            return
        code = status.get("code")
        if code is not None and int(code) != 200:
            message = status.get("message", status)
            raise RuntimeError(f"PegelAlarm API error from {url}: {message}")
