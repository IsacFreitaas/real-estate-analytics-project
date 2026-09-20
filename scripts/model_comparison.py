"""
Runs the same 5-fold cross-validation methodology (Issue #3) for all models
adapted to the new architecture (Issue #4), on the training set only.

Run with:
    python -m scripts.model_comparison
"""

from src.cv import run_cross_validation, save_cv_results
from src.data import load_california_housing, split_train_test
from src.models import get_model_pipelines


def main():
    df = load_california_housing()
    x_train, x_test, y_train, y_test = split_train_test(df)

    comparison = {}

    for name, pipeline in get_model_pipelines().items():
        cv_results = run_cross_validation(pipeline, x_train, y_train)
        comparison[name] = {
            "mean_mae": cv_results["mean_mae"],
            "std_mae": cv_results["std_mae"],
        }

        print(
            f"{name} CV MAE: {cv_results['mean_mae']:.3f} "
            f"(+/- {cv_results['std_mae']:.3f})")

    save_cv_results(comparison, "outputs/model_comparison_cv.json")


if __name__ == "__main__":
    main()
