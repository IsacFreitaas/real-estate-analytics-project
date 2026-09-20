"""
Runs the same 5-fold cross-validation methodology (Issue #3) for all models
adapted to the new architecture (Issue #4) on the training set only, then
evaluates the best CV model once on the untouched test set.

Run with:
    python -m scripts.model_comparison
"""

from src.cv import run_cross_validation, save_cv_results
from src.data import load_california_housing, split_train_test
from src.evaluation import calculate_mae
from src.models import get_model_pipelines


def main():
    df = load_california_housing()
    x_train, x_test, y_train, y_test = split_train_test(df)

    pipelines = get_model_pipelines()
    comparison = {}

    for name, pipeline in pipelines.items():
        cv_results = run_cross_validation(pipeline, x_train, y_train)
        comparison[name] = {
            "mean_mae": cv_results["mean_mae"],
            "std_mae": cv_results["std_mae"],
        }

        print(
            f"{name} CV MAE: {cv_results['mean_mae']:.3f} "
            f"(+/- {cv_results['std_mae']:.3f})")

    save_cv_results(comparison, "outputs/model_comparison_cv.json")

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


if __name__ == "__main__":
    main()
