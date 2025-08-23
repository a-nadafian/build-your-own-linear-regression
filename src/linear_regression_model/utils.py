import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) cost.

    MSE measures the average of the squares of the errors—that is, the average
    squared difference between the estimated values and the actual value.
    It's the cost function that our gradient descent algorithm minimizes.

    Args:
        y_true (np.ndarray): The actual target values.
        y_pred (np.ndarray): The predicted values from the model.

    Returns:
        float: The Mean Squared Error value.
    """
    # The formula for MSE is (1/n) * Σ(y_true - y_pred)^2
    return np.mean((y_true - y_pred)**2)
