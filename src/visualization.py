import matplotlib.pyplot as plt


def _show_or_save(output_path=None):
    """Save the current figure to `output_path`, or display it if omitted."""

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_model_comparison(results, output_path=None):
    """
    Plot MAE comparison between models.

    Args:
            results (pd.DataFrame): DataFrame containing model results.
            output_path (str, optional): Save path for the plot. If omitted,
                display the plot interactively.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        results["Model"],
        results["MAE"])

    plt.title("MAE Comparison Between Models")
    plt.xlabel("Model")
    plt.ylabel("MAE")

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_actual_vs_predicted(y_true, predictions, output_path=None):
    """
    Compare actual values with model predictions.

    Args:
            y_true: Actual values.
            predictions: Predicted values.
            output_path (str, optional): Save path for the plot.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        predictions,
        alpha=0.3)

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--")

    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title("Actual vs. Predicted House Values")

    _show_or_save(output_path)

def plot_feature_importance(
        feature_importance,
        title="Random Forest Feature Importance",
        output_path=None):
    """
    Plot feature importance values.

    Args:
        feature_importance (pd.DataFrame):
            DataFrame containing feature importance values.
        title (str): Plot title, identifying the model that produced it.
        output_path (str, optional): Save path for the plot.
    """

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"])

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)

    plt.gca().invert_yaxis()

    _show_or_save(output_path)

def plot_residuals(predictions, residuals, output_path=None):
    """
    Plot residuals against predicted values.

    Args:
        predictions: Model predictions.
        residuals: Prediction residuals.
        output_path (str, optional): Save path for the plot.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.3
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted House Value")
    plt.ylabel("Residual")
    plt.title("Residuals vs. Predicted Values")

    _show_or_save(output_path)

def plot_residual_distribution(residuals, output_path=None):
    """
    Plot the distribution of model residuals.

    Args:
        residuals: Prediction residuals.
        output_path (str, optional): Save path for the plot.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        residuals,
        bins=30)

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")

    _show_or_save(output_path)
def plot_error_by_value_range(range_summary, output_path=None):
    """
    Plot MAE and mean residual (bias) by range of the actual target value.

    Args:
        range_summary (pd.DataFrame): Output of `analyze_error_by_value_range`.
        output_path (str, optional): Save path for the plot.
    """

    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(12, 5))

    ax_mae.bar(range_summary["value_range"], range_summary["mae"])
    ax_mae.set_xlabel("Actual House Value Range ($100k)")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("MAE by Actual Value Range")

    ax_bias.bar(range_summary["value_range"], range_summary["mean_residual"])
    ax_bias.axhline(y=0, linestyle="--", color="black")
    ax_bias.set_xlabel("Actual House Value Range ($100k)")
    ax_bias.set_ylabel("Mean Residual (actual - predicted)")
    ax_bias.set_title("Mean Residual by Actual Value Range")

    _show_or_save(output_path)
