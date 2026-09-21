from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.pipeline import build_model_pipeline
from src.repro import RANDOM_STATE

def build_baseline_pipeline():
    """
    Build the baseline pipeline: predicts the mean training target value
    for every observation, regardless of the input features.

    The baseline is conceptually different from the other models (it does
    not learn from the features), but is wrapped in the same Pipeline so it
    can be evaluated with the same CV/test methodology for comparison.

    Returns:
            sklearn.pipeline.Pipeline: Baseline pipeline.
    """

    return build_model_pipeline(DummyRegressor(strategy="mean"))

def build_linear_regression_pipeline():
    """
    Build the Linear Regression pipeline.

    Returns:
            sklearn.pipeline.Pipeline: Linear Regression pipeline.
    """

    return build_model_pipeline(LinearRegression())

def build_random_forest_pipeline(random_state=RANDOM_STATE):
    """
    Build the Random Forest Regressor pipeline.

    Args:
            random_state (int): Seed for reproducible training.

    Returns:
            sklearn.pipeline.Pipeline: Random Forest pipeline.
    """

    return build_model_pipeline(
        RandomForestRegressor(random_state=random_state))

def get_model_pipelines():
    """
    Build all model pipelines adapted to the new architecture.

    Returns:
            dict: Mapping of model name to its Pipeline instance.
    """

    return {
        "Baseline": build_baseline_pipeline(),
        "Linear Regression": build_linear_regression_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
    }
