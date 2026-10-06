import numpy as np

from src.analysis import build_feature_importance_frame
from src.models import build_xgboost_pipeline
from src.preprocessing import NUMERIC_FEATURES
from tests.test_pipeline import make_synthetic_data


def test_feature_importance_frame_uses_feature_names():
    x_train, y_train = make_synthetic_data(80, seed=0)
    pipeline = build_xgboost_pipeline().fit(x_train, y_train)

    importance = build_feature_importance_frame(pipeline)

    assert set(importance["Feature"]) == set(NUMERIC_FEATURES)
    assert np.isclose(importance["Importance"].sum(), 1.0)
    assert importance["Importance"].is_monotonic_decreasing


def test_feature_importance_identifies_the_informative_feature():
    x_train, y_train = make_synthetic_data(200, seed=0)
    pipeline = build_xgboost_pipeline().fit(x_train, y_train)

    importance = build_feature_importance_frame(pipeline)

    assert importance.loc[0, "Feature"] == "MedInc"
