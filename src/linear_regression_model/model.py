import numpy as np

from .utils import (
    ensure_2d_array,
    add_intercept_column,
    gradient_descent,
    mean_squared_error,
    r2_score,
)


class LinearRegressionGD:
    """
    Linear Regression trained with batch Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    epochs : int, default=1000
        Maximum number of training epochs.
    fit_intercept : bool, default=True
        If True, includes an intercept term.
    tol : float | None, default=None
        If set, training stops early when improvement < tol.
    random_state : int | None, default=None
        Seed for reproducibility of weight initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        fit_intercept: bool = True,
        tol: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.fit_intercept = bool(fit_intercept)
        self.tol = tol
        self.random_state = random_state

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.weights_: np.ndarray | None = None
        self.training_loss_: list[float] | None = None

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        X = ensure_2d_array(np.asarray(X, dtype=float))
        if self.fit_intercept:
            X = add_intercept_column(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_prepared = self._prepare_features(X)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        weights, history = gradient_descent(
            X_prepared,
            y_array,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            tol=self.tol,
        )

        self.weights_ = weights
        self.training_loss_ = history.get("mse", [])

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.copy()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        X_prepared = self._prepare_features(X)
        return X_prepared @ self.weights_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return r2_score(np.asarray(y, dtype=float).reshape(-1), y_pred)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return mean_squared_error(np.asarray(y, dtype=float).reshape(-1), y_pred)


