import numpy as np
import pandas as pd
import pytest

from src.models import get_model_pipelines
from src.preprocessing import NUMERIC_FEATURES


def make_synthetic_data(n_samples, seed):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(size=(n_samples, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES)
    y = pd.Series(x["MedInc"] * 2 + rng.normal(scale=0.1, size=n_samples))

    return x, y


@pytest.mark.parametrize("model_name", list(get_model_pipelines()))
def test_pipeline_fit_predict(model_name):
    x_train, y_train = make_synthetic_data(60, seed=0)
    x_test, _ = make_synthetic_data(15, seed=1)
    x_test.iloc[0, 0] = np.nan  # the preprocessing step must handle it

    pipeline = get_model_pipelines()[model_name]
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    assert len(predictions) == len(x_test)
    assert np.isfinite(predictions).all()
