# uas-master-thesis-code-notebooks

Code for a master's thesis on short-term forecasting of river water-level
data, using the [pegelalarm.at](https://pegelalarm.at/en/) API (Austrian
water-level service). Target station `207241-at`, hourly height readings,
forecasting 24 hours ahead.

For an implementation-oriented walkthrough of fetching, preprocessing, feature
engineering, and model training, see [Stages 1–4 pipeline guide](docs/pipeline.md).

## Setup

1. `uv sync` (or `make requirements`)
2. Copy `.env.example` to `.env` and fill in your pegelalarm.at credentials:

   ```bash
   cp .env.example .env
   ```

3. `make hooks` — installs the pre-commit hooks (ruff on commit; mypy + pytest
   on push) and the nbstripout git filter that keeps notebook outputs out of
   commits. This registers local git config, so re-run it after each fresh
   clone.

## Notebook run order

Run the notebooks directly, or use the equivalent `make` target. The current
Make dependencies are intentionally asymmetric: `make features` depends on
`make data`, and any stage-4 training target therefore re-runs fetching,
preprocessing, and feature engineering. `make evaluate` does not depend on
training.

| # | Notebook | `make` target | Output |
| --- | ---------- | ---------------- | -------- |
| 1 | `01_fetch_data.ipynb` | `make data` (runs 01 + 02) | `data/raw/` |
| 2 | `02_preprocessing.ipynb` | ↑ | chronological train/test artifacts in `data/processed/` |
| 3 | `03_feature_engineering.ipynb` | `make features` (runs 01 + 02 + 03) | `data/processed/` |
| 4 | `04_01_train_persistence.ipynb` | `make train-persistence` | in-notebook metrics and MLflow run hierarchy |
| 4 | `04_02_train_ridge.ipynb` | `make train-ridge` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| 4 | `04_03_train_mlp.ipynb` | `make train-mlp` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| 4 | `04_04_train_xgboost.ipynb` | `make train-xgboost` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| 4 | `04_06_train_extra_trees.ipynb` | `make train-extra-trees` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| 4 | `04_07_train_rnn.ipynb` | `make train-rnn` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| — | persistence, Ridge, MLP, XGBoost, Extra Trees, and RNN | `make train` | runs the current six-notebook training set |
| 5 | `05_evaluate.ipynb` | `make evaluate` | comparison plots/tables |

All six `04_*_train_*.ipynb` notebooks share stage number 4: they are parallel
model candidates, not sequential steps. The flat-feature candidates use the
same joined cohort and validation folds; the RNN narrows that cohort further for
each sequence length. Every fitted model notebook writes a selected model and
durable manifest to `models/`; persistence has no fitted artifact. `make
evaluate` assumes the desired training notebooks have already been run. If
you're starting from scratch, run the default pipeline top to bottom with
`make data features train evaluate`.

The active stage-4 notebooks display prediction previews and aggregate/per-horizon
metrics. They log candidate and sealed-test runs to MLflow; the fitted-model
training notebooks persist a manifest as the durable execution record.

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
