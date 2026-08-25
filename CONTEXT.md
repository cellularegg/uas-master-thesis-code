# CONTEXT

The vocabulary this pipeline is built around. One term, one meaning, one place
in the code that owns it. If a name here starts meaning two things, that is the
bug — not the naming.

## issue time

One timestamp `t` at which a forecast is made. A row of a feature artifact is
one issue time: its predictors are all known at `t`, and its targets are the
water levels at `t+1 … t+FORECAST_HORIZON_HOURS`. Called `timestamp` in the
feature artifacts and `issue_time` once predictions are attached to it.

## cohort

The set of issue times a model is fit or scored on. An issue time joins the
cohort only when Stage 3 marked its future window valid *and* every predictor
in the **full** contract is present — never a candidate's own subset, so all
candidates and the persistence baseline are compared on identical rows.
Train and test cohorts are filtered independently and never pooled.

Owned by `src/dataset.py`.

## sealed test

The chronologically last `TEST_FRACTION` of the data, scored exactly once, at
the end of a notebook. No statistic derived from it — not even a scaler mean —
ever reaches a model. "Sealed" is the invariant, not a label: a second scoring
pass, or a refit after seeing the result, breaks it.

## joined dataset

Everything a stage-4 notebook needs from the Stage-3 joined artifacts, loaded
and validated in one call: the feature contract, both eligible cohorts, the
ablation subsets, the validation folds, the input hashes, and the target
context series. A stage-4 notebook does not assemble these itself.

`src/dataset.py`: `load_joined_dataset()` → `JoinedDataset`.

## ablation subset

One named, contract-ordered list of predictor columns — `full`,
`target_station_full`, `current_water_levels_all_stations`, and so on. Subsets
answer "which inputs actually carry the skill", so they are compared on the
same cohort: a smaller subset never gains eligible rows from the predictors it
drops.

## candidate

One `(ablation subset, hyperparameter)` pair, evaluated across every validation
fold. Ridge's search is 6 subsets × 5 alphas = 30 candidates; MLP, XGBoost,
Random Forest, and Extra Trees use their notebook-defined candidate spaces and
reuse each sampled set across all six subsets. A candidate is *selected* by
the configured `CV_SELECTION_METRIC` with explicit estimator-specific
tie-breaking — never by anything the sealed test reported.

`src/ridge.py`, `src/mlp.py`, `src/xgboost_model.py`, `src/random_forest.py`,
and `src/extra_trees.py` own the candidate-selection policies.

## execution

One complete run of a training notebook: the candidate search, the retrain of
the selected candidate, and the single sealed-test pass. Identified by an
`execution_uuid` tagged on every MLflow run it produces and recorded in the
manifest it writes. An execution that crashes part-way is not an execution —
it leaves no manifest.

## manifest

The durable record of one execution, written once, after the sealed test has
been scored. Ridge, MLP, XGBoost, Random Forest, and Extra Trees write a
model-specific JSON manifest carrying the saved model's identity and exact
input/output contract: estimator and preprocessor identity, the selected
candidate and hyperparameters, ordered feature and target columns, model path,
and `execution_uuid`. It contains no CV table, sealed-test metrics, or other
regime, cohort, and training diagnostics. Those diagnostics remain in MLflow,
which is the durable scientific record of the training and evaluation results;
the manifest is the durable record of the serialized model contract.

`src/ridge.py`, `src/mlp.py`, `src/xgboost_model.py`, `src/random_forest.py`,
and `src/extra_trees.py` expose model-specific save/load helpers and manifest
types.

## target context series

The target station's observed `water_level` and `imputed` flag across both
artifacts, indexed by unique UTC timestamp. Ground truth for plotting — not a
model input, and deliberately built from the *raw* frames, so it covers issue
times that no cohort qualifies.

`JoinedDataset.target_context_series`.

## forecast window

One issue time's 24-hour forecast drawn against the target context series ±48
hours around it. A window is eligible only when that whole span is observed
hour by hour with nothing imputed, so a chart never implies ground truth that
was interpolated. The best and worst eligible windows are
chosen by the RMSE across all horizons of that single issue.

`src/plots.py`: `forecast_window_figures()`.
