import matplotlib.pyplot as plt

def plot_model_comparison(results):
    """
    Plot MAE comparison between models.

    Args:
            results (pd.DataFrame): DataFrame containing model results.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        results["Model"],
        results["MAE"])

    plt.title("MAE Comparison Between Models")
    plt.xlabel("Model")
    plt.ylabel("MAE")

    plt.show()

def plot_actual_vs_predicted(y_true, predictions):
    """
    Compare actual values with model predictions.

    Args:
            y_true: Actual values.
            predictions: Predicted values.
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

    plt.show()

def plot_feature_importance(feature_importance):
    """
    Plot feature importance values.

    Args:
        feature_importance (pd.DataFrame):
            DataFrame containing feature importance values.
    """

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"])

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.show()

def plot_residuals(predictions, residuals):
    """
    Plot residuals against predicted values.

    Args:
        predictions: Model predictions.
        residuals: Prediction residuals.
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

    plt.show()

def plot_residual_distribution(residuals):
    """
    Plot the distribution of model residuals.

    Args:
        residuals: Prediction residuals.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        residuals,
        bins=30)

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")

    plt.show()