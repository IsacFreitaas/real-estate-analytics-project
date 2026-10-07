from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.repro import RANDOM_STATE

TARGET_COLUMN = "MedHouseVal"

def load_california_housing():
    """
    Load the California Housing dataset as a pandas DataFrame.

    Returns:
            pd.DataFrame: California Housing dataset.
    """
    
    housing = fetch_california_housing(
        as_frame=True)

    return housing.frame

def split_features_target(df, target=TARGET_COLUMN):
    """
    Split a DataFrame into features and target.

    Args:
            df (pd.DataFrame): Full dataset, including the target column.
            target (str): Name of the target column.

    Returns:
            tuple: (X, y) where X is the features DataFrame and y is the
            target Series.
    """

    x = df.drop(target, axis=1)
    y = df[target]

    return x, y

def split_train_test(df, target=TARGET_COLUMN, test_size=0.2, random_state=RANDOM_STATE):
    """
    Split a DataFrame into reproducible train/test feature and target sets.

    This is the single source of truth for the train/test split so it can
    be reused consistently by cross-validation and final test evaluation,
    keeping the test set untouched until final evaluation.

    Args:
            df (pd.DataFrame): Full dataset, including the target column.
            target (str): Name of the target column.
            test_size (float): Proportion of the dataset used for testing.
            random_state (int): Seed for reproducible splitting.

    Returns:
            tuple: (x_train, x_test, y_train, y_test).
    """

    x, y = split_features_target(df, target=target)

    return train_test_split(
        x, y, test_size=test_size, random_state=random_state)