import numpy as np
# We can import our new cost function, though we calculate it implicitly
# during gradient descent. It's good practice to have it available.
from .utils import mean_squared_error

class LinearRegression:
    """
    A custom implementation of a simple linear regression model from scratch.

    This class uses Gradient Descent to learn the optimal parameters (weights and bias)
    for a linear relationship between features and a target variable.

    Attributes:
        learning_rate (float): The step size for each iteration of Gradient Descent.
        n_iterations (int): The number of times the Gradient Descent algorithm will iterate.
        weights (np.ndarray): The learned coefficients (slopes) of the regression line.
        bias (float): The learned intercept of the regression line.
        cost_history (list): A list that stores the cost (MSE) at each iteration.
    """
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        """
        Initializes the LinearRegression model with hyperparameters.

        Args:
            learning_rate (float, optional): Controls the step size of the optimization.
                                             Defaults to 0.01.
            n_iterations (int, optional): The number of training iterations.
                                          Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Trains the linear regression model using Gradient Descent.

        This method iteratively adjusts the model's weights and bias to minimize
        the Mean Squared Error (MSE) cost function.

        Args:
            X (np.ndarray): The training input samples (features). Expected to be a 2D array.
            y (np.ndarray): The target values (labels). Expected to be a 1D array.
        """
        # Get the number of samples (rows) and features (columns) from the input data
        n_samples, n_features = X.shape

        # Initialize the weights and bias to zero.
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []

        # The main Gradient Descent loop
        for i in range(self.n_iterations):
            # Calculate the model's predictions using the current weights and bias.
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate and store the cost (MSE) for the current iteration
            cost = mean_squared_error(y, y_predicted)
            self.cost_history.append(cost)

            # Calculate the gradients of the cost function
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Makes predictions using the trained linear regression model.

        Args:
            X (np.ndarray): The input samples for which to make predictions.

        Returns:
            np.ndarray: The predicted values.
        """
        return np.dot(X, self.weights) + self.bias
