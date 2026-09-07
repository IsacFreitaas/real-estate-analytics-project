from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, predictions):
    """
    Calculate Mean Absolute Error.

    Args:
            y_true: Actual values.
            predictions: Predicted values.

    Returns:
            float: Mean Absolute Error.
    """

    return mean_absolute_error(
        y_true,
        predictions)

def calculate_error_reduction(
    previous_mae,
    new_mae):
    """
    Calculates the percentage reduction in prediction error.

    Args:
            previous_mae: Previous Mean Absolute Error.
            new_mae: New Mean Absolute Error.

    Returns:
            float: Percentage reduction in error.
    """

    return ((previous_mae - new_mae) / previous_mae) * 100