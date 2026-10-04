import pandas as pd

def build_error_frame(y_true, predictions):
    """
    Build a DataFrame with actual values, predictions, and residuals.

    Residual is defined as actual - predicted, so positive values mean the
    model under-predicted the house value.

    Args:
            y_true: Actual target values.
            predictions: Model predictions.

    Returns:
            pd.DataFrame: Columns `actual`, `predicted`, `residual`,
            `abs_error`.
    """

    frame = pd.DataFrame({
        "actual": pd.Series(y_true).to_numpy(),
        "predicted": pd.Series(predictions).to_numpy(),
    })

    frame["residual"] = frame["actual"] - frame["predicted"]
    frame["abs_error"] = frame["residual"].abs()

    return frame

def summarize_residuals(error_frame):
    """
    Summarize the residual distribution with descriptive statistics.

    Args:
            error_frame (pd.DataFrame): Output of `build_error_frame`.

    Returns:
            dict: MAE, residual mean/std/skew, and error percentiles.
    """

    residual = error_frame["residual"]
    abs_error = error_frame["abs_error"]

    return {
        "n_samples": int(len(error_frame)),
        "mae": float(abs_error.mean()),
        "median_abs_error": float(abs_error.median()),
        "p90_abs_error": float(abs_error.quantile(0.90)),
        "p99_abs_error": float(abs_error.quantile(0.99)),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_skew": float(residual.skew()),
    }
