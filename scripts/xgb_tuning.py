"""
Run a reproducible RandomizedSearchCV for XGBoost on the training set only.

Run with:
    python -m scripts.xgb_tuning
"""

from src.data import load_california_housing, split_train_test
from src.tuning import build_xgb_randomized_search, save_json, summarize_randomized_search


def main():
    df = load_california_housing()
    x_train, _, y_train, _ = split_train_test(df)

    search = build_xgb_randomized_search(x_train, y_train)
    summary = summarize_randomized_search(search)

    save_json(summary, "outputs/xgb_random_search.json")

    print(f"XGBoost best CV MAE: {summary['best_cv_mae']:.3f}")
    print(f"XGBoost best params: {summary['best_params']}")


if __name__ == "__main__":
    main()
