import numpy as np


def ensure_2d_array(X: np.ndarray) -> np.ndarray:
    """
    Ensure input X is a 2D numpy array (n_samples, n_features).
    If X is 1D, reshape to (n_samples, 1).
    """
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X for the intercept term."""
    n_samples = X.shape[0]
    intercept = np.ones((n_samples, 1), dtype=X.dtype)
    return np.hstack((intercept, X))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Handle edge case where variance is zero
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def compute_gradients(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute gradient of MSE with respect to weights for linear regression.

    Gradient: (2/n) * X^T (Xw - y)
    """
    n_samples = X.shape[0]
    predictions = X @ weights
    errors = predictions - y
    gradient = (2.0 / n_samples) * (X.T @ errors)
    return gradient


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    weights_init: np.ndarray | None = None,
    tol: float | None = None,
    verbose: bool = False,
):
    """
    Perform batch gradient descent to minimize MSE for linear regression.

    Returns
    -------
    weights : np.ndarray
        Learned weights vector of shape (n_features, ). Includes intercept if X includes it.
    history : dict
        Contains per-epoch metrics like 'mse'.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_features = X.shape[1]
    if weights_init is None:
        rng = np.random.default_rng()
        weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
    else:
        weights = weights_init.astype(float).copy()

    history: dict[str, list[float]] = {"mse": []}
    previous_loss = None

    for epoch in range(epochs):
        predictions = X @ weights
        loss = mean_squared_error(y, predictions)
        history["mse"].append(loss)

        gradient = compute_gradients(X, y, weights)
        weights -= learning_rate * gradient

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - MSE: {loss:.6f}")

        if tol is not None:
            if previous_loss is not None and abs(previous_loss - loss) < tol:
                break
            previous_loss = loss

    return weights, history


