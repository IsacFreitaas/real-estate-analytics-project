"""
Demonstrates reproducible 5-fold cross-validation (Issue #3) on the training
split only, followed by a single final evaluation on the untouched test set.

Run with:
    python -m scripts.cv_demo
"""

from sklearn.linear_model import LinearRegression

from src.cv import run_cross_validation, save_cv_results
from src.data import load_california_housing, split_train_test
from src.evaluation import calculate_mae
from src.pipeline import build_model_pipeline


def main():
    df = load_california_housing()

    x_train, x_test, y_train, y_test = split_train_test(df)

    pipeline = build_model_pipeline(LinearRegression())

    # Cross-validation uses the training set only.
    cv_results = run_cross_validation(pipeline, x_train, y_train)
    save_cv_results(cv_results, "outputs/linear_regression_cv.json")

    print(
        f"Linear Regression CV MAE: {cv_results['mean_mae']:.3f} "
        f"(+/- {cv_results['std_mae']:.3f})")

    # Final, single evaluation on the untouched test set.
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    test_mae = calculate_mae(y_test, predictions)

    print(f"Linear Regression final test MAE: {test_mae:.3f}")


if __name__ == "__main__":
    main()
