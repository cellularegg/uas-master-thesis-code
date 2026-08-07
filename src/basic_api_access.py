from datetime import datetime
from typing import Any

import requests
import json
import pandas as pd
from . import auth_token_generator as alg


class BasicApiAccess(object):
    xAuthToken: str | None = None

    def __init__(self, usr: str, pwd: str):
        credentials = {"username": usr, "password": pwd}
        self.xAuthToken = self.get_xauth_token(credentials)

    def get_xauth_token(self, credentials: dict) -> str:
        api_key = self.get_api_key(credentials)
        if api_key is None:
            raise RuntimeError("Failed to obtain pegelalarm.at API key; check credentials.")
        x_auth_token = alg.AuthTokenGenerator().calc_xauth_token(credentials["username"], api_key)
        return x_auth_token

    @staticmethod
    def get_api_key(credentials_int: dict) -> str | None:
        url = "https://api.pegelalarm.at/api/login"
        payload = json.dumps(credentials_int)
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url,
            headers=headers,
            data=payload,
            # , verify=False
        )

        if response.status_code == 200:
            if response.json()["status"]["code"] == 200:
                return response.json()["payload"]["apiKey"]
            else:
                print("Invalid login: " + str(response.json()))
        else:
            print("Failed to call API: " + response.url)
        return None

    def query_current_data(self, station_name: str = "", water_name: str = "", common_id: str = "") -> Any:
        assert self.xAuthToken is not None
        parameters = {"qStationName": station_name, "qWater": water_name, "commonid": common_id}
        headers = {"Content-Type": "application/json", "X-AUTH-TOKEN": self.xAuthToken}
        response = requests.get(
            "https://api.pegelalarm.at/api/station/1.1/list",
            params=parameters,
            headers=headers,
            # , verify=False
        )
        if response.status_code == 200:
            # jsonPrint(response.json()['payload'])
            return response.json()["payload"]
        else:
            print("Failed to call API: " + response.url)

    # Possible values for 'unit': "height" and "flow"
    # Possible values for 'granularity': "raw" (for recent 3 months of data), "hour", "day", "month", "year" and "era"
    def query_historic_data(
        self,
        commonid: str,
        load_start_date_utc: datetime,
        load_end_date_utc: datetime,
        unit: str = "height",
        granularity: str = "hour",
    ) -> pd.DataFrame | None:
        assert self.xAuthToken is not None
        load_start_date_str = load_start_date_utc.strftime("%d.%m.%YT%H:%M:%S") + "%2B0200"
        load_end_date_str = load_end_date_utc.strftime("%d.%m.%YT%H:%M:%S") + "%2B0200"
        url = (
            "https://api.pegelalarm.at/api/station/1.1/"
            + unit
            + "/"
            + commonid
            + "/history?"
            + "loadStartDate="
            + load_start_date_str
            + "&loadEndDate="
            + load_end_date_str
            + "&granularity="
            + granularity
        )
        headers = {"Content-Type": "application/json", "X-AUTH-TOKEN": self.xAuthToken}
        response = requests.get(
            url,
            headers=headers,
            # , verify=False
        )
        if response.status_code == 200:
            history_df = pd.DataFrame(response.json()["payload"]["history"])
            if history_df.empty:
                print("DataFrame ist leer.")
            else:
                history_df["sourceDate"] = pd.to_datetime(history_df["sourceDate"], format="%d.%m.%YT%H:%M:%S%z")
            return history_df
        else:
            print("Failed to call API: " + response.url)
        return None
