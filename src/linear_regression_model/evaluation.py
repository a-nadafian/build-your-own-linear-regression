import numpy as np

def rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    RMSE is the square root of the average of the squared differences between
    prediction and actual observation. It's a measure of the average magnitude
    of the error and is in the same units as the target variable.

    Args:
        y_true (np.ndarray): The actual target values.
        y_pred (np.ndarray): The predicted values from the model.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    """
    Calculates the R-squared (coefficient of determination) regression score.

    R-squared represents the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s). A score of 1.0 indicates
    a perfect fit, while a score of 0 indicates the model does no better than
    predicting the mean of the target.

    Args:
        y_true (np.ndarray): The actual target values.
        y_pred (np.ndarray): The predicted values from the model.

    Returns:
        float: The R-squared score.
    """
    # Calculate the total sum of squares (TSS), which measures the total variance in the data.
    # It's the sum of the squared differences between the actual values and their mean.
    ss_total = np.sum((y_true - np.mean(y_true))**2)

    # Calculate the residual sum of squares (RSS), which measures the error of the model.
    # It's the sum of the squared differences between the actual and predicted values.
    ss_residual = np.sum((y_true - y_pred)**2)

    # The R-squared score is 1 minus the ratio of the residual sum of squares to the total sum of squares.
    return 1 - (ss_residual / ss_total)
