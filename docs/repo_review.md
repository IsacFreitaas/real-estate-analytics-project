# Repository Review — V2 Implementation Plan

> Produced for Issue 1 — Review Current State. This document is a factual inventory of the
> repository as of 2026-09-19 and a concrete, ordered implementation plan for Issues 2–10.

## 1. Repository Structure

```text
real-estate-analytics-project
├── README.md
├── requirements.txt
├── instructions-agent.md
├── images/
│   ├── Isac-Freitas-LinkedIn-Banner.jpg
│   ├── feature-importance.png
│   ├── geographic-distribution.png
│   ├── house-value-distribution.png
│   ├── median-income-vs-house-value.png
│   ├── model-comparison.png
│   └── real-estate-thumbnail.jpg
├── notebooks/
│   ├── 1. EDA.ipynb
│   └── 2. Modelling.ipynb
└── src/
    ├── __init__.py   (empty)
    ├── data.py
    ├── evaluation.py
    └── visualization.py
```

No `tests/` directory and no `.github/workflows/` exist in the repository.

## 2. Existing Notebooks

- `1. EDA.ipynb` — Exploratory data analysis: distribution of features/target, correlation between
  `MedInc` and `MedHouseVal`, geographic distribution of house values, and identification of the
  target's upper-value cap (US$500,000).
- `2. Modelling.ipynb` — Data preparation and modelling workflow:
  - Imports `train_test_split`, `LinearRegression`, `RandomForestRegressor` from scikit-learn.
  - Performs a single `train_test_split(..., random_state=42)` (no cross-validation).
  - Establishes a mean-prediction **baseline**.
  - Trains a `LinearRegression` model directly on the training split.
  - Trains a `RandomForestRegressor(random_state=42)` directly on the training split.
  - Evaluates all three approaches once on the held-out test split using MAE.
  - Produces residual analysis and feature-importance plots via `src/visualization.py`.

## 3. Existing `src/` Modules

- `src/__init__.py` — empty package marker.
- `src/data.py` — `load_california_housing()`: loads the Scikit-learn California Housing dataset
  as a pandas `DataFrame` via `fetch_california_housing(as_frame=True)`. No train/test split logic
  lives here; the split is currently done inline in the modelling notebook.
- `src/evaluation.py`:
  - `calculate_mae(y_true, predictions)` — wraps `sklearn.metrics.mean_absolute_error`.
  - `calculate_error_reduction(previous_mae, new_mae)` — percentage MAE reduction between two models.
- `src/visualization.py`:
  - `plot_model_comparison(results)` — bar chart of MAE per model.
  - `plot_actual_vs_predicted(y_true, predictions)` — scatter plot with identity line.
  - `plot_feature_importance(feature_importance)` — horizontal bar chart.
  - `plot_residuals(predictions, residuals)` — residuals vs. predicted scatter.
  - `plot_residual_distribution(residuals)` — histogram of residuals.

## 4. Existing `requirements.txt`

```text
pandas
scikit-learn
matplotlib
seaborn
numpy
```

No version pins, no `xgboost`, no `pytest`, no linter (e.g., `ruff`).

## 5. Existing Model-Training Workflow

Preprocessing is minimal (no explicit `Pipeline`/`ColumnTransformer`); the dataset returned by
`load_california_housing()` is split once via `train_test_split` inside the notebook and each
model is fit directly on the raw training features. There is currently no reusable pipeline
object, so preprocessing logic (if any is added later) would need to be duplicated per model —
this is the main gap Issue 2 addresses.

## 6. Existing Evaluation Functions

Evaluation is limited to a single train/test split and MAE (`calculate_mae`,
`calculate_error_reduction` in `src/evaluation.py`). There is no cross-validation utility and no
aggregated (mean/std) scoring across folds — this is the gap Issue 3 addresses.

## 7. Existing Visualization Functions

`src/visualization.py` already covers model comparison, actual-vs-predicted, feature importance,
and residual plots (both scatter and distribution). These are reusable for Issues 6–8 and do not
require changes to be adapted to new models, as long as the models expose predictions and
residuals in the same shape used today.

## 8. Existing Requirements / Dependencies

See section 4. `xgboost`, `pytest`, and a linter/formatter are not yet declared and will be added
in Issues 6 and 9 respectively.

## 9. Existing Git Branches

Prior to this Issue, the repository had a single branch (`main`) with no other local or remote
branches. As part of this Issue, `develop` was created from `main`, and
`chore/review-current-state` was created from `develop` to hold this review.

## 10. Existing Tests

None. No `tests/` directory, no `pytest` configuration, and no CI workflow exist in the
repository. This is the gap Issue 9 addresses.

## 11. Existing README Documentation

`README.md` documents the project description, dataset, objective, analytical pipeline, tools,
EDA insights, modelling approach (baseline, Linear Regression, Random Forest), results table
(MAE: 0.906 / 0.533 / 0.328), limitations, future improvements (which already list cross-validation,
hyperparameter tuning, and additional models — aligned with this V2 plan), and run instructions.
It does not yet document a `Pipeline` architecture, cross-validation, XGBoost, error analysis, or
CI/CD, since none of these exist yet.

## 12. Risk Note

The main risk for the upcoming refactor is **data leakage during cross-validation**: because
preprocessing is currently applied ad hoc in the notebook rather than inside a `Pipeline`, any
transformer fit on the full dataset (instead of per-fold on training data only) would leak
information into validation folds and produce optimistic MAE estimates. A secondary risk is
**reproducibility drift** across the two notebooks and future `src/` scripts if `random_state`
values and package versions are not centralized and pinned before introducing cross-validation,
hyperparameter search, and XGBoost.

## 13. Implementation Plan (Issues 2–10)

| Issue | Files to change / add | Effort |
|---|---|---|
| 2. Refactor with `Pipeline` | `src/pipeline.py` (new), `src/preprocessing.py` (new) | Medium |
| 3. Introduce Cross-Validation | `src/cv.py` (new), `outputs/` (new) | Small |
| 4. Adapt Existing Models | `src/models.py` (new), `outputs/model_comparison_cv.json` | Medium |
| 5. Verify Reproducibility | `src/repro.py` (new), `requirements-locked.txt`, `docs/reproducibility.md` | Small |
| 6. Implement XGBoost | `requirements.txt`, `src/models.py`, `outputs/xgb_cv.json` | Small |
| 7. Optimize XGBoost | `src/tune.py` (new), `outputs/xgb_random_search.json` | Medium |
| 8. Error Analysis | `outputs/error_analysis/`, `reports/error_analysis.md` | Medium |
| 9. Unit Tests and CI/CD | `tests/`, `.github/workflows/ci.yml`, `requirements-dev.txt` | Medium |
| 10. Docs and Release | `README.md`, `reports/summary.md`, `RELEASE.md` | Small |

No code was modified in the existing `src/` modules or notebooks as part of this Issue; only this
review document was added.
