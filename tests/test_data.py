import pandas as pd
import pytest

from src.data import (
    TARGET_COLUMN,
    load_california_housing,
    split_features_target,
    split_train_test,
)
from src.preprocessing import NUMERIC_FEATURES

EXPECTED_COLUMNS = [*NUMERIC_FEATURES, TARGET_COLUMN]


@pytest.fixture(scope="module")
def housing_df():
    return load_california_housing()


def test_load_returns_dataframe(housing_df):
    assert isinstance(housing_df, pd.DataFrame)
    assert not housing_df.empty


def test_expected_columns_exist(housing_df):
    assert set(EXPECTED_COLUMNS) <= set(housing_df.columns)


def test_target_column_exists(housing_df):
    assert TARGET_COLUMN == "MedHouseVal"
    assert TARGET_COLUMN in housing_df.columns


def test_no_missing_values(housing_df):
    assert not housing_df.isnull().any().any()


def test_features_are_numeric(housing_df):
    numeric = housing_df[EXPECTED_COLUMNS].select_dtypes("number")

    assert list(numeric.columns) == EXPECTED_COLUMNS


def test_dataset_can_enter_modelling_workflow(housing_df):
    x, y = split_features_target(housing_df)
    x_train, x_test, y_train, y_test = split_train_test(housing_df)

    assert TARGET_COLUMN not in x.columns
    assert len(x) == len(y) == len(housing_df)
    assert len(x_train) + len(x_test) == len(housing_df)
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)
