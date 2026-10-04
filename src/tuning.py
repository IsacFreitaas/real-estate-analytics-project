import json
import os

from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV

from src.models import build_xgboost_pipeline
from src.repro import RANDOM_STATE

DEFAULT_N_SPLITS = 5
DEFAULT_N_ITER = 20
DEFAULT_RANDOM_STATE = RANDOM_STATE
DEFAULT_SCORING = "neg_mean_absolute_error"
XGB_SEARCH_SPACE = {
    "model__n_estimators": randint(100, 401),
    "model__max_depth": randint(3, 9),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__subsample": uniform(0.7, 0.3),
    "model__colsample_bytree": uniform(0.7, 0.3),
    "model__min_child_weight": randint(1, 8),
    "model__gamma": loguniform(1e-8, 1.0),
    "model__reg_alpha": loguniform(1e-8, 1.0),
    "model__reg_lambda": loguniform(1.0, 10.0),
}

def build_xgb_randomized_search(
    x_train,
    y_train,
    n_iter=DEFAULT_N_ITER,
    random_state=DEFAULT_RANDOM_STATE,
    n_splits=DEFAULT_N_SPLITS,
    scoring=DEFAULT_SCORING):
    """
    Fit a reproducible RandomizedSearchCV for the XGBoost regression pipeline.

    The search is restricted to the training split and uses cross-validation
    internally, so the test set stays untouched until the final evaluation.

    Args:
            x_train: Training features.
            y_train: Training target.
            n_iter (int): Number of sampled hyperparameter configurations.
            random_state (int): Seed for the search and fold shuffling.
            n_splits (int): Number of CV folds.
            scoring (str): Scikit-learn scoring string. Defaults to MAE.

    Returns:
            sklearn.model_selection.RandomizedSearchCV: Fitted search object.
    """

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state)

    search = RandomizedSearchCV(
        estimator=build_xgboost_pipeline(random_state=random_state),
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        random_state=random_state,
        refit=True,
        return_train_score=False,
        verbose=0)

    return search.fit(x_train, y_train)

def summarize_randomized_search(search, top_n=5):
    """
    Summarize the fitted RandomizedSearchCV results in a JSON-friendly form.

    Args:
            search (RandomizedSearchCV): Fitted search object.
            top_n (int): Number of top candidates to keep in the summary.

    Returns:
            dict: Best params, best CV MAE, and top candidate summaries.
    """

    cv_results = search.cv_results_
    ranked_candidates = sorted(
        range(len(cv_results["rank_test_score"])),
        key=lambda index: cv_results["rank_test_score"][index])[:top_n]

    top_candidates = []
    for index in ranked_candidates:
        top_candidates.append({
            "rank": int(cv_results["rank_test_score"][index]),
            "mean_cv_mae": float(-cv_results["mean_test_score"][index]),
            "std_cv_mae": float(cv_results["std_test_score"][index]),
            "params": _to_native(cv_results["params"][index]),
        })

    return {
        "n_iter": int(search.n_iter),
        "n_splits": int(search.cv.n_splits),
        "random_state": int(search.random_state),
        "best_cv_mae": float(-search.best_score_),
        "best_params": _to_native(search.best_params_),
        "top_candidates": top_candidates,
    }

def save_json(data, path):
    """
    Persist JSON data to disk, creating parent directories if required.

    Args:
            data (dict): JSON-serializable payload.
            path (str): Output path.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as file_handle:
        json.dump(_to_native(data), file_handle, indent=2)

def _to_native(value):
    """
    Recursively convert numpy/scipy scalar types to native Python types.
    """

    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_native(item) for item in value]
    if isinstance(value, tuple):
        return [_to_native(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value
