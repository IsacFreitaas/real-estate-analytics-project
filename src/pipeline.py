from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

def build_model_pipeline(model, preprocessor=None):
    """
    Build a reusable Scikit-learn Pipeline combining preprocessing and a model.

    Keeping preprocessing and model training in a single Pipeline object
    ensures transformations are fitted only on training data/folds, which
    prevents data leakage during cross-validation.

    Args:
            model: Any Scikit-learn compatible estimator.
            preprocessor: Optional pre-built preprocessing step. Defaults to
                    `build_preprocessor()` when not provided.

    Returns:
            sklearn.pipeline.Pipeline: Pipeline with "preprocessing" and
            "model" steps, ready to be fit on training data.
    """

    if preprocessor is None:
        preprocessor = build_preprocessor()

    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ])
