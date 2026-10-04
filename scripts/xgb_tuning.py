"""
Run a reproducible RandomizedSearchCV for XGBoost on the training set only.

Run with:
    python -m scripts.xgb_tuning
"""

import pandas as pd

from src.cv import run_cross_validation
from src.data import load_california_housing, split_train_test
from src.evaluation import calculate_mae
from src.models import get_model_pipelines
from src.tuning import (
    build_xgb_randomized_search,
    save_json,
    summarize_randomized_search,
)
from src.visualization import plot_model_comparison


def main():
    df = load_california_housing()
    x_train, x_test, y_train, y_test = split_train_test(df)

    search = build_xgb_randomized_search(x_train, y_train)
    summary = summarize_randomized_search(search)

    save_json(summary, "outputs/xgb_random_search.json")

    optimized_predictions = search.best_estimator_.predict(x_test)
    optimized_test_mae = calculate_mae(y_test, optimized_predictions)

    optimized_test_results = {
        "test_mae": optimized_test_mae,
        "best_cv_mae": summary["best_cv_mae"],
        "best_params": summary["best_params"],
    }
    save_json(optimized_test_results, "outputs/xgb_optimized_test.json")

    comparison = {}
    for name, pipeline in get_model_pipelines().items():
        if name == "XGBoost":
            continue

        cv_results = run_cross_validation(pipeline, x_train, y_train)
        comparison[name] = {
            "mean_mae": cv_results["mean_mae"],
            "std_mae": cv_results["std_mae"],
        }

    comparison["XGBoost (optimized)"] = {
        "mean_mae": summary["best_cv_mae"],
        "std_mae": summary["top_candidates"][0]["std_cv_mae"],
    }

    save_json(comparison, "outputs/model_comparison_optimized.json")

    plot_model_comparison(
        pd.DataFrame([
            {"Model": name, "MAE": metrics["mean_mae"]}
            for name, metrics in comparison.items()
        ]),
        output_path="images/model-comparison-optimized-xgb.png")

    print(f"XGBoost best CV MAE: {summary['best_cv_mae']:.3f}")
    print(f"XGBoost best params: {summary['best_params']}")
    print(f"XGBoost optimized test MAE: {optimized_test_mae:.3f}")


if __name__ == "__main__":
    main()
