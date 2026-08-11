"""Project configuration loaded from environment variables and constants."""

import os

from dotenv import load_dotenv

load_dotenv()

# Username used to authenticate requests to the PegelAlarm data service.
PEGELALARM_USERNAME = os.getenv("PEGELALARM_USERNAME")
# Password used to authenticate requests to the PegelAlarm data service.
PEGELALARM_PASSWORD = os.getenv("PEGELALARM_PASSWORD")

# Station identifiers included in the downloaded and processed dataset.
STATION_IDS = [
    "207241-at",
    "207019-at",
    "207027-at",
    "207340-at",
    "207068-at",
    "Ennshafen1.Rivermeter-at",
    "207084-at",
    "207100-at",
    "207357-at",
    "DUER-at",
]
# Station whose water level is the primary modeling target.
TARGET_STATION_ID = "207241-at"
# Country code used for the station/data source configuration.
COUNTRY_CODE = "AT"
# Measurement unit requested for water-level observations.
UNIT = "height"
# Time resolution requested when fetching station observations.
GRANULARITY = "hour"
# Whether existing downloaded artifacts may be reused instead of fetched again.
SKIP_IF_EXISTS = True
# Local MLflow backend used to store experiment runs and metrics.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# Number of chronological validation folds used to compare Ridge alphas.
N_VALIDATION_FOLDS = 5
# Fraction of eligible rows targeted for the first expanding training window.
INITIAL_TRAIN_FRACTION = 0.50
# Number of hourly rows left between each fold's training and validation data.
EMBARGO_HOURS = 24
# Aggregate CV metric used to select the best Ridge alpha.
CV_SELECTION_METRIC = "mae"

# INCA weather dataset identifier used for weather feature retrieval.
INCA_DATASET_ID = "inca-v1-1h-1km"
# Weather variables fetched and retained as model inputs.
WEATHER_VARIABLES = [
    "precipitation",
    "temperature_2m",
]

# Number of future hourly water levels generated as forecast targets.
FORECAST_HORIZON_HOURS = 24
# Maximum missing water-level gap length eligible for interpolation.
MAX_INTERPOLATION_GAP_HOURS = 6
# Fraction of chronological rows reserved for the sealed test partition.
TEST_FRACTION = 0.20
# Minimum target-range overlap required to retain a non-target station.
MIN_TARGET_RANGE_OVERLAP = 0.90
