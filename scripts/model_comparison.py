"""
Runs the same 5-fold cross-validation methodology for all models on the
training set only, then evaluates the best CV model and XGBoost on the
untouched test set.

Run with:
    python -m scripts.model_comparison
"""

import pandas as pd

from src.cv import run_cross_validation, save_cv_results
from src.data import load_california_housing, split_train_test
from src.evaluation import calculate_mae
from src.models import get_model_pipelines
from src.visualization import plot_model_comparison


def main():
    df = load_california_housing()
    x_train, x_test, y_train, y_test = split_train_test(df)

    pipelines = get_model_pipelines()
    comparison = {}

    for name, pipeline in pipelines.items():
        cv_results = run_cross_validation(pipeline, x_train, y_train)
        if name == "XGBoost":
            save_cv_results(cv_results, "outputs/xgb_cv.json")
        comparison[name] = {
            "mean_mae": cv_results["mean_mae"],
            "std_mae": cv_results["std_mae"],
        }

        print(
            f"{name} CV MAE: {cv_results['mean_mae']:.3f} "
            f"(+/- {cv_results['std_mae']:.3f})")

    save_cv_results(comparison, "outputs/model_comparison_cv.json")

    plot_model_comparison(
        pd.DataFrame([
            {"Model": name, "MAE": metrics["mean_mae"]}
            for name, metrics in comparison.items()
        ]),
        output_path="images/model-comparison-cv.png")

    # Select the model with the lowest CV mean MAE and evaluate it once,
    # on the untouched test set.
    best_name = min(comparison, key=lambda name: comparison[name]["mean_mae"])
    best_pipeline = pipelines[best_name]

    best_pipeline.fit(x_train, y_train)
    predictions = best_pipeline.predict(x_test)
    test_mae = calculate_mae(y_test, predictions)

    print(f"Best model by CV: {best_name} | final test MAE: {test_mae:.3f}")

    save_cv_results(
        {"best_model": best_name, "test_mae": test_mae},
        "outputs/final_test_evaluation.json")

    if best_name == "XGBoost":
        xgboost_test_mae = test_mae
    else:
        xgboost_pipeline = pipelines["XGBoost"]
        xgboost_pipeline.fit(x_train, y_train)
        xgboost_predictions = xgboost_pipeline.predict(x_test)
        xgboost_test_mae = calculate_mae(y_test, xgboost_predictions)

    print(f"XGBoost final test MAE: {xgboost_test_mae:.3f}")

    save_cv_results(
        {"test_mae": xgboost_test_mae},
        "outputs/xgb_test.json")


if __name__ == "__main__":
    main()
