import os

from dotenv import load_dotenv

load_dotenv()

PEGELALARM_USERNAME = os.getenv("PEGELALARM_USERNAME")
PEGELALARM_PASSWORD = os.getenv("PEGELALARM_PASSWORD")

STATION_IDS = ["207241-at"]
UNIT = "height"
GRANULARITY = "hour"
FORECAST_HORIZON_HOURS = 24
