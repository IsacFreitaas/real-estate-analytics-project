from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

def build_numeric_transformer():
    """
    Build the transformer applied to the numeric features of the
    California Housing dataset.

    Returns:
            sklearn.pipeline.Pipeline: Numeric preprocessing steps.
    """

    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

def build_preprocessor(numeric_features=NUMERIC_FEATURES):
    """
    Build the full preprocessing ColumnTransformer for the dataset.

    Args:
            numeric_features (list): Names of the numeric feature columns.

    Returns:
            sklearn.compose.ColumnTransformer: Reusable preprocessing step,
            meant to be fitted only on training folds/data inside a Pipeline.
    """

    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_transformer(), numeric_features),
        ])
