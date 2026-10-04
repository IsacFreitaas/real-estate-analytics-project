"""
Descriptive error analysis of the optimized XGBoost pipeline (Issue #8).

The pipeline is rebuilt from the best parameters recorded by the tuning step
(outputs/xgb_random_search.json) and fitted on the training split only. The
untouched test set is used only to describe the errors; nothing is selected
or tuned from it.

Run with:
    python -m scripts.error_analysis
"""

import json

from src.analysis import build_error_frame, summarize_residuals
from src.data import load_california_housing, split_train_test
from src.models import build_xgboost_pipeline
from src.tuning import save_json
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_residuals,
)

SEARCH_SUMMARY_PATH = "outputs/xgb_random_search.json"
OUTPUT_DIR = "outputs/error_analysis"
IMAGE_DIR = "images/error-analysis"


def fit_optimized_pipeline(x_train, y_train):
    """Fit the XGBoost pipeline with the tuned parameters on the train split."""

    with open(SEARCH_SUMMARY_PATH) as file_handle:
        best_params = json.load(file_handle)["best_params"]

    pipeline = build_xgboost_pipeline().set_params(**best_params)

    return pipeline.fit(x_train, y_train)


def main():
    df = load_california_housing()
    x_train, x_test, y_train, y_test = split_train_test(df)

    pipeline = fit_optimized_pipeline(x_train, y_train)
    error_frame = build_error_frame(y_test, pipeline.predict(x_test))

    residual_summary = summarize_residuals(error_frame)
    save_json(residual_summary, f"{OUTPUT_DIR}/residual_summary.json")

    plot_residuals(
        error_frame["predicted"],
        error_frame["residual"],
        output_path=f"{IMAGE_DIR}/residuals-vs-predicted.png")
    plot_residual_distribution(
        error_frame["residual"],
        output_path=f"{IMAGE_DIR}/residual-distribution.png")
    plot_actual_vs_predicted(
        error_frame["actual"],
        error_frame["predicted"],
        output_path=f"{IMAGE_DIR}/actual-vs-predicted.png")

    print(f"Test MAE: {residual_summary['mae']:.3f}")
    print(f"Residual mean: {residual_summary['residual_mean']:.3f}")
    print(f"Residual skew: {residual_summary['residual_skew']:.3f}")


if __name__ == "__main__":
    main()
