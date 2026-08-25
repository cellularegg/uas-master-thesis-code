# Pipeline implementation guide: Stages 1–4

This guide describes how the notebook pipeline is implemented from raw-data
fetching through model training. It intentionally stops before Stage 5
evaluation and Stage 6 model selection.

Concrete station lists, horizons, split fractions, folds, embargoes, feature
windows, search spaces, and training budgets are not duplicated here. For a
run, `src/config.py` and the executable constants in the relevant notebook are
the configuration sources of truth. Stage-3 feature metadata is the realized
predictor/target contract, and each fitted model's saved manifest is the durable
record of a completed training execution.

## Stage 1: fetching

`01_fetch_data.ipynb` orchestrates the clients and helpers in
`src/basic_api_access.py`, `src/auth_token_generator.py`, and
`src/fetch_data.py`.

PegelAlarm credentials come from `.env` through the configuration module. The
notebook refuses to start when either credential is missing. Login exchanges
the username and password for an API key and then creates the HMAC-signed
`X-AUTH-TOKEN` used for station-catalog and history requests. HTTP failures,
API-level errors, malformed response bodies, and missing payloads are surfaced
as errors rather than accepted as empty data.

The Austrian station catalog is fetched first. The configured target/upstream
station IDs are then processed in their configured order. For each gauge, a
broad yearly PegelAlarm request discovers the earliest archived observation;
hourly height history is downloaded in bounded, non-overlapping chunks,
concatenated, sorted, and deduplicated by timestamp. Empty histories fail the
station. PegelAlarm request boundaries must be timezone-aware and are converted
to UTC; response timestamps are parsed as UTC as well.

GeoSphere Austria's INCA historical timeseries API is queried for each gauge at
its catalog coordinates over that gauge's available water-level interval. The
response represents the nearest grid point. Native weather fields are renamed
to the project vocabulary, numeric values and timestamps are validated, missing
weather values remain missing, duplicate inclusive-boundary timestamps are
removed, and requested/grid coordinates plus the weather-model identifier are
preserved. (The raw artifact uses `grid_*` columns for the returned point.)

When `SKIP_IF_EXISTS` is enabled, each existing catalog, water-history, or
weather Parquet is reused independently. Reuse is therefore artifact-based,
not an all-or-nothing cache. Failures are caught per artifact so the remaining
stations in that phase can finish. The notebook resets its failure collection
between the water and weather phases; the final summarized exception therefore
reports weather-phase failures rather than acting as a cumulative audit. A
missing water artifact normally reappears as a weather dependency failure, but
an already-reused weather artifact bypasses that check. Downstream stages still
fail if a required raw artifact is absent.

Stage 1 writes independent files under `data/raw/`:

- `pegelalarm_stations_<country>.parquet` for the flattened station catalog;
- `pegelalarm_<station>_height_hour.parquet` for each gauge's water history;
- `geosphere_inca_<station>_hour.parquet` for matching INCA weather.

## Stage 2: preprocessing

`02_preprocessing.ipynb` calls `src/preprocess.py` for per-station cleaning,
chronological partitioning, station selection, wide joining, and provenance
writing.

For each configured station, raw water values at or below the configured
validity threshold are treated as missing. The series is sorted and reindexed
onto a strict, unique hourly UTC grid. Interior gaps no longer than the
configured limit are forward-filled from the last valid observation and
marked by `imputed`; longer, leading, and trailing gaps stay null and are not
presented as observed
measurements. INCA precipitation and temperature are left-joined to this water
timeline. Weather is neither interpolated nor backfilled, so missing source
weather remains visible to later eligibility checks.

Each station frame is split chronologically. The leading
`floor((1 - TEST_FRACTION) * N)` rows form training data and the remainder form
the physically sealed test artifact. The partitions are written separately;
they are never recombined for feature lookbacks or model preprocessing.

Before the wide join, each station's complete hourly span is compared with the
target station's span. Stations meeting `MIN_TARGET_RANGE_OVERLAP` are retained,
and the target is retained unconditionally. The notebook orders retained
upstream gauges by river-kilometre distance from the target. Train and test are
joined independently, using the target station's timestamps as the left-hand
timeline. Every non-timestamp column is prefixed `<station_id>__`; an upstream
value remains null when that station has no row at a target timestamp.

Per-station train/test Parquets and JSON manifests are written under
`data/processed/separate/`. The joined outputs are
`data/processed/joined/all_stations_train.parquet`,
`all_stations_test.parquet`, and `all_stations_preprocess_metadata.json`. The
manifests capture configuration, UTC ranges, row counts, artifact schemas,
paths and hashes, plus generator provenance. The joined manifest additionally
records the retained station order and the per-station overlap/retention report.

## Stage 3: feature engineering

`03_feature_engineering.ipynb` uses `src/feature_engineering.py`. It reads the
retained station order from the preprocessing manifest, checks that manifest's
target against the configured target, and reads the joined train and sealed-test
Parquets independently. The artifact writer verifies that the engineered
outputs preserve the current source rows, columns, timestamps, and values.

Only the target station receives derived features and future targets. Upstream
stations remain issue-time raw predictors. Within each partition, positive
shifts create water-level lags and changes; trailing windows create water-level
summary statistics, imputation counts, precipitation sums, and temperature
summaries; cyclic UTC calendar terms encode hour, weekday, and day of year.
Rolling windows include issue time `t`, require a complete window, and preserve
source missingness. No predictor reads future weather or a future water level.
Because train and test are engineered separately, test lookbacks cannot reach
back into training.

Direct targets contain the target station's water level from `t+1` through the
configured horizon. A row is target-eligible only when the entire future window
exists, every future value was observed, and none was imputed. When it is not
eligible, `target_valid` is false and the complete target vector is null. The
feature stage drops no rows, fills no missing predictors, and performs no
scaling. This deliberately leaves warm-up rows and source gaps visible for the
training-stage cohort policy.

The output Parquets are
`all_stations_train_features.parquet` and
`all_stations_test_features.parquet`. The accompanying
`all_stations_feature_metadata.json` is the authoritative downstream contract:
it records the target and engineered station IDs, feature configuration,
ordered predictor columns, ordered target columns, artifact profiles, and
input/output provenance. Stage 4 validates this contract before fitting.

### Downstream feature subsets

Stage 3 writes one ordered predictor contract rather than separate datasets for
each model input. During Stage 4, `src.dataset.load_joined_dataset` derives six
ordered ablation subsets from that contract. In the definitions below,
`<target>` is the configured target-station ID, `<station>` is any station
retained by preprocessing, and every name is fully qualified as
`<station>__<feature>`.

| Subset | Columns included |
| --- | --- |
| `full` | Every predictor declared by the Stage-3 metadata: all raw and derived target-station columns, plus the four raw columns of every retained upstream station. |
| `all_station_hydrology_quality_time` | For every retained station, `water_level` and `imputed`; for the target station, all `water_level_*`, `imputed_count_*`, and `utc_*` columns as well. Weather and weather-derived columns are excluded. |
| `raw_all_stations` | `<station>__water_level`, `<station>__imputed`, `<station>__precipitation`, and `<station>__temperature_2m` for every retained station. |
| `target_station_full` | Every predictor whose prefix is `<target>__`: the target station's four raw columns and every derived column listed below. |
| `target_station_hydrology_quality_time` | `<target>__water_level`, `<target>__imputed`, and all target-station `water_level_*`, `imputed_count_*`, and `utc_*` columns. Weather and weather-derived columns are excluded. |
| `current_water_levels_all_stations` | `<station>__water_level` for every retained station; no lagged, rolling, quality, weather, or calendar columns. |

The target station's complete feature families are:

- raw measurements and quality: `water_level`, `imputed`, `precipitation`, and
  `temperature_2m`;
- water-level history: `water_level_lag_<hours>h` for every configured lag and
  `water_level_change_<hours>h` for every configured change interval;
- rolling water-level statistics:
  `water_level_rolling_{mean,std,min,max}_<hours>h` for every configured rolling
  window;
- rolling quality and weather summaries: `imputed_count_<hours>h`,
  `precipitation_rolling_sum_<hours>h`, and
  `temperature_2m_rolling_mean_<hours>h` for every configured rolling window,
  plus `temperature_2m_rolling_min_24h` and
  `temperature_2m_rolling_max_24h`;
- UTC calendar encodings: `utc_hour_sin`, `utc_hour_cos`,
  `utc_day_of_week_sin`, `utc_day_of_week_cos`, `utc_day_of_year_sin`, and
  `utc_day_of_year_cos`.

The realized station prefixes, configured hours, exact column order, and subset
sizes can change with the preprocessing and feature configuration. The
Stage-3 metadata and the subsets returned by `load_joined_dataset` are therefore
authoritative for a particular run.

## Stage 4: model training

All seven `04_*` notebooks call reusable logic in `src/`; notebooks retain the
run-specific constants, orchestration, displays, and plots. The shared entry
point `src.dataset.load_joined_dataset` validates the feature metadata against
the configured target/horizon, checks both Parquets contain the declared
columns, and creates the common flat-model cohort. A row qualifies only when
`target_valid` is true and every declared predictor and target is non-null. Train
and sealed test are filtered independently and sorted by issue time.

The full predictor contract determines eligibility even for a smaller ablation
subset or the persistence baseline. This prevents easier subsets from gaining
extra timestamps and keeps like-for-like comparisons. `src.dataset` derives
the ordered named subsets from the metadata contract: full; hydrology, quality,
and time across stations; raw channels across stations; target-station full;
target-station hydrology, quality, and time; and current water levels across
stations.

Chronological validation uses expanding `TimeSeriesSplit` folds. The configured
initial fraction reserves the early training history, validation windows cover
the later eligible rows, and an embargo gap separates each training window from
its validation window. No fold fits a scaler or estimator on validation data.
Candidate ranking uses only aggregate cross-validation metrics and deterministic
estimator-specific tie-breaking. The sealed test is consulted exactly once,
after selection and retraining on all eligible training rows; it never selects
a candidate.

The notebooks are:

- `04_01_train_persistence.ipynb`: repeats the issue-time target water level
  across all declared horizons. It fits nothing, but scores the same cohort and
  fold windows as the fitted flat models.
- `04_02_train_ridge.ipynb`: searches feature subset, Ridge strength, and an
  optional log transform. Predictors are standardized. When selected, eligible
  non-negative water-level predictors and the target are transformed with
  `log1p`; signed changes and standard deviations are not log-transformed.
- `04_03_train_mlp.ipynb`: searches feature subset, hidden-layer architecture,
  and regularization. Both predictors and targets are standardized; the target
  scaler is wrapped with the estimator so predictions return on the original
  scale.
- `04_04_train_xgboost.ipynb`: samples a seeded hyperparameter set and reuses it
  for every subset. The native multi-output tree estimator receives raw numeric
  predictors without scaling.
- `04_05_train_random_forest.ipynb`: samples a seeded hyperparameter set reused
  across subsets and fits a native multi-output `RandomForestRegressor` on raw
  numeric predictors without scaling. Run it with `make train-random-forest`;
  the current aggregate `make train` target intentionally excludes it.
- `04_06_train_extra_trees.ipynb`: follows the same subset/sampled-search shape
  with a native multi-output `ExtraTreesRegressor`. It uses raw numeric
  predictors; randomized split thresholds provide the estimator's defining
  extra randomness.
- `04_07_train_rnn.ipynb`: searches GRU/LSTM cell and sequence architecture over
  the `raw_all_stations` channel contract. It constructs complete contiguous
  hourly lookback sequences separately within train and test. Sequence length
  can further narrow the common cohort and changes its folds. Channels and
  targets are standardized inside the fitted `RnnForecaster`.

For each fitted search, every candidate has an MLflow parent run and nested fold
runs. Persistence uses its own parent with nested fold runs. The selected
sealed-test run logs parameters, aggregate and per-horizon metrics, figures,
input hashes, cohort counts/ranges, and an `execution_uuid` tying the hierarchy
together. MLflow's local SQLite backend is a run log and inspection UI, not a
pipeline input.

Every fitted-model notebook writes
`models/<model>_{TARGET_STATION_ID}.joblib` plus a sibling JSON manifest only
after sealed-test scoring succeeds. The manifest is the durable
model description: it stores the estimator/preprocessor identity, selected
hyperparameters, ordered feature/channel and target contracts, model path, and
`execution_uuid`. CV, sealed-test, regime, cohort, and training diagnostics stay
in MLflow rather than the manifest. Manifest loaders validate the model contract
against the current feature metadata and subset definitions before a saved model
can be rescored. Persistence has no fitted model or manifest.
