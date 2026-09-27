# uas-master-thesis-code

The written thesis itself lives in a separate repository:
[cellularegg/uas-master-thesis](https://github.com/cellularegg/uas-master-thesis).
The PDF can be found here: [cellularegg/uas-master-thesis/releases](https://github.com/cellularegg/uas-master-thesis/releases/).

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
| 4 | `04_05_train_arima.ipynb` | `make train-arima` | training ACF/PACF, explicit-grid ARIMA CV/test metrics, MLflow runs, and saved model/manifest |
| 4 | `04_06_train_extra_trees.ipynb` | `make train-extra-trees` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| 4 | `04_07_train_rnn.ipynb` | `make train-rnn` | metrics, MLflow run hierarchy, and saved model/manifest in `models/` |
| 4 | `04_08_train_arimax.ipynb` | `make train-arimax` | recursive ARIMAX subset/order CV, MLflow runs, and saved model/manifest |
| — | persistence, Ridge, MLP, XGBoost, ARIMA, Extra Trees, RNN, and ARIMAX | `make train` | runs the current eight-notebook training set |
| 5 | `05_evaluate.ipynb` | `make evaluate` | comparison plots/tables |

All eight `04_*_train_*.ipynb` notebooks share stage number 4: they are parallel
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

ARIMA evaluates recursive multi-step forecasting for each of 48 fixed nonseasonal SARIMAX configurations (p,q ∈ 0..3, d ∈ {0,1}, intercept only when d=0) on the common CV scoring rows. Each fit uses the full observed hourly target history, with imputed hours treated as missing, through the last training issue time plus the 24-hour horizon. The CV winner is selected with the shared tie-breaking selector, refitted under the same rule, and evaluated on the sealed test. Saved models use a schema-5 manifest; older artifacts must be regenerated.

ARIMAX fits a recursive `X(t) → level(t+1)` regression with ARIMA errors. It
crosses the six existing feature subsets with thirteen nonseasonal orders, using
the shared eligible training rows, folds, embargo, and sealed test.
Run `uv run jupyter execute --inplace 04_08_train_arimax.ipynb` to reuse existing
Stage-3 artifacts; `make train-arimax` also runs the upstream pipeline.

At issue time `t`, observed levels through `t` update the fixed-parameter state
when the target label and predictor row are available. Forecasts recompute UTC
calendar predictors for each future input hour and rebuild the target station's
level, lag, change, rolling, and imputation-count predictors from the forecasts
with the Stage-3 formulas (`target_level_features`). Upstream levels and weather
have no forecasts and are held at their issue-time values. No future observations or predictor rows are read. Missing hours
remain on the time axis, and imputed levels do not update the state. Training-only
standardization retains every column, including redundant predictors; finite
nonconverged fits remain eligible and their diagnostics are logged in MLflow.
The separate `arimax` experiment participates in overall and feature-subset
evaluation. `models/arimax_<station>.joblib` and its JSON manifest preserve the
training snapshot and inference contract; test evaluation does not mutate it.
Saved ARIMAX models use a schema-6 manifest; older artifacts must be regenerated.
