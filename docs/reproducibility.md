# Reproducibility

> Produced for Issue #5 — Verify Reproducibility.

## What was verified

- **Train/test split**: `src/data.py:split_train_test()` uses a fixed
  `random_state` (from `src/repro.py`), so the same rows are always
  assigned to train/test across runs.
- **Cross-validation**: `src/cv.py:run_cross_validation()` uses
  `KFold(shuffle=True, random_state=...)` with the same centralized seed,
  so fold assignment is identical across runs.
- **Model reproducibility**: `RandomForestRegressor` is seeded via
  `random_state` (Issue #4); `LinearRegression` and the `DummyRegressor`
  baseline are deterministic by construction and require no seed.
- **Metric calculation**: `src/evaluation.py:calculate_mae()` and the
  `neg_mean_absolute_error` scorer used in `src/cv.py` are deterministic
  given the same predictions.
- **End-to-end check**: `scripts/check_repro.sh` runs
  `python -m scripts.model_comparison` twice and diffs
  `outputs/model_comparison_cv.json` and `outputs/final_test_evaluation.json`
  between runs. Both runs produced byte-identical results:

  ```text
  Baseline CV MAE: 0.914 (+/- 0.010)
  Linear Regression CV MAE: 0.529 (+/- 0.009)
  Random Forest CV MAE: 0.335 (+/- 0.005)
  Best model by CV: Random Forest | final test MAE: 0.328
  ```

## How to reproduce

```bash
pip install -r requirements-locked.txt
./scripts/check_repro.sh
```

Versions verified: Python 3.14.7, scikit-learn 1.9.0, pandas 3.0.5,
numpy 2.5.2 (full list in `requirements-locked.txt`).

## Centralized configuration

All `random_state` values used across the project are read from a single
constant, `RANDOM_STATE` in `src/repro.py`, imported by `src/data.py`,
`src/cv.py`, and `src/models.py`. This avoids seed drift between modules.

## Expected sources of small numerical variation

- **`RandomForestRegressor` parallelism**: when `n_jobs` is set to use
  multiple threads, floating-point summation order across trees/threads can
  vary slightly, which may introduce tiny (typically negligible, far below
  displayed precision) differences in predictions on different machines or
  CPU architectures, even with a fixed `random_state`.
- **Package version drift**: upgrading `scikit-learn`, `numpy`, or their
  underlying BLAS/LAPACK libraries can change internal numerical routines
  and produce small differences in model coefficients or splits.
  `requirements-locked.txt` pins the exact versions used to produce the
  results above.
- **Notebook execution order**: `notebooks/2. Modelling.ipynb` still
  duplicates the train/test split and model training inline (not yet
  migrated to `src/pipeline.py` / `src/models.py`); re-running notebook
  cells out of order, or with a different `random_state` typed manually,
  would not reproduce the same numbers. This is a known limitation to
  address when the notebooks are updated in Issue #10.
