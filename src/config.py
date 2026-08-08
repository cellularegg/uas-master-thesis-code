"""Project configuration loaded from environment variables and constants."""

import os

from dotenv import load_dotenv

load_dotenv()

PEGELALARM_USERNAME = os.getenv("PEGELALARM_USERNAME")
PEGELALARM_PASSWORD = os.getenv("PEGELALARM_PASSWORD")

STATION_IDS = ["207241-at"]
COUNTRY_CODE = "AT"
UNIT = "height"
GRANULARITY = "hour"
SKIP_IF_EXISTS = True

ERA5_MODEL = "era5"
WEATHER_VARIABLES = [
    "precipitation",
    "rain",
    "snowfall",
    "temperature_2m",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
]

FORECAST_HORIZON_HOURS = 24
