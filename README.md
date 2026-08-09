# uas-master-thesis-code-notebooks

Code for a master's thesis on short-term forecasting of river water-level
data, using the [pegelalarm.at](https://pegelalarm.at) API (Austrian
water-level service). Target station `207241-at`, hourly height readings,
forecasting 24 hours ahead.

## Setup

1. `uv sync` (or `make requirements`)
2. Copy `.env.example` to `.env` and fill in your pegelalarm.at credentials:
   ```
   cp .env.example .env
   ```
3. `make hooks` — installs the pre-commit hooks (ruff on commit; mypy + pytest
   on push) and the nbstripout git filter that keeps notebook outputs out of
   commits. This registers local git config, so re-run it after each fresh
   clone.

## Notebook run order

Run the notebooks directly, or use the equivalent `make` target. The current
Make dependencies are intentionally asymmetric: `make features` depends on
`make data`, and either training target therefore re-runs fetching,
preprocessing, and feature engineering. `make evaluate` does not depend on
training, while `make model_selection` re-runs evaluation first.

| # | Notebook | `make` target | Output |
|---|----------|----------------|--------|
| 1 | `01_fetch_data.ipynb` | `make data` (runs 01 + 02) | `data/raw/` |
| 2 | `02_preprocessing.ipynb` | ↑ | chronological train/test artifacts in `data/processed/` |
| 3 | `03_feature_engineering.ipynb` | `make features` (runs 01 + 02 + 03) | `data/processed/` |
| 4 | `04_train_persistence.ipynb` | `make train-persistence` | MLflow run, `models/` |
| 4 | `04_train_ridge.ipynb` | `make train-ridge` | MLflow run, `models/` |
| — | (both of the above) | `make train` | — |
| 5 | `05_evaluate.ipynb` | `make evaluate` | comparison plots/tables |
| 6 | `06_model_selection.ipynb` | `make model_selection` (runs 05 + 06) | selected model/run |

`04_train_persistence.ipynb` / `04_train_ridge.ipynb` share stage number 4:
they're parallel model candidates, not sequential steps. `make evaluate` and
`make model_selection` assume `make train` has already been run — run the
full pipeline top to bottom with `make data features train evaluate
model_selection` if you're starting from scratch.

Training runs log params/metrics to a local MLflow file store (`./mlruns/`,
inspect with `uv run mlflow ui`); trained model files are saved to `./models/`.

## Weather source

Historical weather comes exclusively from GeoSphere Austria's
[INCA hourly analysis dataset](https://data.hub.geosphere.at/en/dataset/inca-v1-1h-1km).
It provides hourly UTC analyses on a 1 km grid under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). For every gauge the
pipeline queries the historical timeseries API with the gauge's WGS 84
coordinates; GeoSphere returns the nearest grid point, whose returned
coordinates are preserved in the raw artifact. The pipeline uses only the
native `RR` (one-hour precipitation sum) and `T2M` (2 m air temperature)
parameters, normalized to `precipitation` and `temperature_2m`; see the
[timeseries API behavior](https://dataset.api.hub.geosphere.at/v1/docs/user-guide/type.html).
