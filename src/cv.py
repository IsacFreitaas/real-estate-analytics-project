import json
import os

from sklearn.model_selection import KFold, cross_val_score

from src.repro import RANDOM_STATE

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = RANDOM_STATE
MAE_SCORING = "neg_mean_absolute_error"

def run_cross_validation(
    pipeline,
    x_train,
    y_train,
    n_splits=DEFAULT_N_SPLITS,
    random_state=DEFAULT_RANDOM_STATE,
    scoring=MAE_SCORING):
    """
    Run reproducible k-fold cross-validation on the training set only.

    The same methodology (splitter, scoring, and random_state) can be
    reused for any pipeline/model, keeping model comparison consistent.
    The test set must never be passed to this function.

    Args:
            pipeline: A Scikit-learn compatible Pipeline/estimator.
            x_train: Training features.
            y_train: Training target.
            n_splits (int): Number of cross-validation folds.
            random_state (int): Seed for reproducible fold shuffling.
            scoring (str): Scikit-learn scoring string. Defaults to MAE.

    Returns:
            dict: Per-fold MAE scores plus mean/std MAE and run configuration.
    """

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state)

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring)

    mae_scores = -scores

    return {
        "fold_mae": mae_scores.tolist(),
        "mean_mae": float(mae_scores.mean()),
        "std_mae": float(mae_scores.std()),
        "n_splits": n_splits,
        "random_state": random_state,
    }

def save_cv_results(results, path):
    """
    Persist cross-validation results as a JSON file.

    Args:
            results (dict): Output of `run_cross_validation`.
            path (str): Destination file path.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2)
