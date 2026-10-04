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

VALUE_RANGE_EDGES = [0, 1, 2, 3, 4, float("inf")]
VALUE_RANGE_LABELS = ["0-1", "1-2", "2-3", "3-4", "4+"]

def analyze_error_by_value_range(
        error_frame,
        edges=VALUE_RANGE_EDGES,
        labels=VALUE_RANGE_LABELS):
    """
    Describe prediction errors by range of the actual target value.

    Ranges are left-open and right-closed, e.g. (1, 2]. The target is
    expressed in units of $100,000.

    Args:
            error_frame (pd.DataFrame): Output of `build_error_frame`.
            edges (list): Range boundaries.
            labels (list): Range names, one per interval.

    Returns:
            pd.DataFrame: Per range sample count and share, MAE, mean
            residual (bias), 90th percentile of absolute error, and share of
            the total absolute error.
    """

    ranges = pd.cut(
        error_frame["actual"],
        bins=edges,
        labels=labels,
        include_lowest=True)

    grouped = error_frame.groupby(ranges, observed=True)

    summary = pd.DataFrame({
        "n_samples": grouped.size(),
        "mae": grouped["abs_error"].mean(),
        "mean_residual": grouped["residual"].mean(),
        "p90_abs_error": grouped["abs_error"].quantile(0.90),
        "total_abs_error": grouped["abs_error"].sum(),
    })

    summary["share_of_samples"] = summary["n_samples"] / len(error_frame)
    summary["share_of_total_error"] = (
        summary["total_abs_error"] / error_frame["abs_error"].sum())

    summary = summary.drop(columns="total_abs_error")
    summary.index.name = "value_range"

    return summary.reset_index()
