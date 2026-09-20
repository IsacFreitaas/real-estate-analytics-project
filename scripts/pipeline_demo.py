"""
Demonstrates fitting the reusable Pipeline (Issue #2) on the training split
only, confirming preprocessing never sees the test data before evaluation.

Run with:
    python -m scripts.pipeline_demo
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data import load_california_housing
from src.evaluation import calculate_mae
from src.pipeline import build_model_pipeline


def main():
    df = load_california_housing()

    x = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    pipeline = build_model_pipeline(LinearRegression())

    # Preprocessing is fit only on x_train, inside the Pipeline.
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    mae = calculate_mae(y_test, predictions)

    print(f"Linear Regression pipeline test MAE: {mae:.3f}")


if __name__ == "__main__":
    main()
