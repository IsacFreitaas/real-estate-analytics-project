"""
Feature importance of the optimized XGBoost pipeline.

The pipeline is rebuilt from the best parameters recorded by the tuning step
(outputs/xgb_random_search.json) and fitted on the training split only. The
test set is not used. Importance describes how much the model used each
feature; it does not measure a causal effect on house prices.

Run with:
    python -m scripts.xgb_feature_importance
"""

from scripts.error_analysis import fit_optimized_pipeline
from src.analysis import build_feature_importance_frame
from src.data import load_california_housing, split_train_test
from src.visualization import plot_feature_importance

OUTPUT_PATH = "outputs/xgb_feature_importance.csv"
IMAGE_PATH = "images/xgb-feature-importance.png"


def main():
    df = load_california_housing()
    x_train, _, y_train, _ = split_train_test(df)

    pipeline = fit_optimized_pipeline(x_train, y_train)
    feature_importance = build_feature_importance_frame(pipeline)

    feature_importance.to_csv(OUTPUT_PATH, index=False)
    plot_feature_importance(
        feature_importance,
        title="Optimized XGBoost Feature Importance (gain)",
        output_path=IMAGE_PATH)

    print(feature_importance.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
