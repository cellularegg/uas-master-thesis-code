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

Run the notebooks directly, or use the equivalent `make` target (each target
runs the notebooks for its own stage only — it does **not** re-run earlier
stages, since re-fetching data or retraining models can take a while and
shouldn't happen just because you want to re-run evaluation):

| # | Notebook | `make` target | Output |
|---|----------|----------------|--------|
| 1 | `01_fetch_data.ipynb` | `make data` (runs 01 + 02) | `data/raw/` |
| 2 | `02_preprocessing.ipynb` | ↑ | `data/interim/` |
| 3 | `03_split_folds.ipynb` | `make features` (runs 03 + 04) | `data/processed/` |
| 4 | `04_feature_engineering.ipynb` | ↑ | `data/processed/` |
| 5 | `05_train_persistence.ipynb` | `make train-persistence` | MLflow run, `models/` |
| 5 | `05_train_ridge.ipynb` | `make train-ridge` | MLflow run, `models/` |
| — | (both of the above) | `make train` | — |
| 6 | `06_evaluate.ipynb` | `make evaluate` | comparison plots/tables |
| 7 | `07_model_selection.ipynb` | `make model_selection` (runs 06 + 07) | selected model/run |

`05_train_persistence.ipynb` / `05_train_ridge.ipynb` share stage number 5:
they're parallel model candidates, not sequential steps. `make evaluate` and
`make model_selection` assume `make train` has already been run — run the
full pipeline top to bottom with `make data features train evaluate
model_selection` if you're starting from scratch.

Training runs log params/metrics to a local MLflow file store (`./mlruns/`,
inspect with `uv run mlflow ui`); trained model files are saved to `./models/`.
