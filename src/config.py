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

INCA_DATASET_ID = "inca-v1-1h-1km"
WEATHER_VARIABLES = [
    "precipitation",
    "temperature_2m",
]

FORECAST_HORIZON_HOURS = 24
MAX_INTERPOLATION_GAP_HOURS = 6
TEST_FRACTION = 0.20
