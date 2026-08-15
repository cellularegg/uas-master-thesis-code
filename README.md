
  # Ridge final-model comparison plots

  ## Summary

  Extend the existing `# Ridge MLflow candidate comparison` section in `04_train_ridge.ipynb`.

  - Load the saved Ridge pipeline and manifest.
  - Score every eligible sealed-test row: all 15,196 × 24 predictions.
  - Add an interactive Plotly time series with an H+1–H+24 slider.
  - Add separate absolute-error and signed-error boxplot figures.

  For H=2, the x-axis will be `issue_time + 2 hours`; actual values use `target_t_plus_02`, and predictions use the second prediction column. Issue time will remain available in hover data.

  ## Key changes

  - Reload the joined test artifact independently so the comparison remains standalone and read-only.
  - Validate that the model manifest matches:
    - the current feature/target contract;
    - the selected MLflow execution UUID;
    - selected subset and alpha;
    - prediction shape and finite values.
  - Fail fast on any provenance or contract mismatch.
  - Expose a wide prediction table containing issue time, actual targets, and all loaded predictions.
  - Build the time-series slider with Plotly `Scattergl` traces and frames, using all test rows without sampling.
  - Add:
    - absolute-error boxplots for H+1–H+24 with MAE/RMSE markers;
    - signed-error boxplots using `prediction - actual`, with mean-error markers and a zero reference line.
  - Keep the existing MLflow metric summary and sealed-test horizon chart intact.

  ## Tests and verification

  - Update notebook-structure tests for the added comparison cells.
  - Extend standalone/read-only checks to cover model loading and prediction generation.
  - Add mocked tests for:
    - all-horizon prediction shape and ordering;
    - valid-time alignment, including H=2;
    - signed-error convention;
    - 24 boxplot traces and marker values;
    - provenance mismatch failure.
  - Run the targeted notebook and training tests, then execute the comparison section against the saved model to verify the plots render.

  ## Assumptions

  - Model files remain `models/ridge_{TARGET_STATION_ID}.joblib` and `.json`.
  - Plotly is used for all new figures.
  - Boxplots contain every test-set error value but do not render a point cloud.
  - Existing unrelated working-tree changes remain untouched.
