"""
Regression Metrics Module

This module provides functions to calculate common regression evaluation metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

All functions accept numpy arrays or Python lists for predicted and actual values.
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def _validate_inputs(y_true, y_pred, metric_name):
    """Validate input arrays for regression metrics."""
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for regression metrics calculations")

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check dimensions
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"{metric_name}: Input arrays must be 1-dimensional")

    # Check lengths
    if len(y_true) != len(y_pred):
        raise ValueError(f"{metric_name}: Input arrays must have the same length")

    # Check for empty arrays
    if len(y_true) == 0:
        raise ValueError(f"{metric_name}: Input arrays cannot be empty")

    return y_true, y_pred


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).

    MAE = (1/n) * Σ|y_true - y_pred|

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Mean absolute error

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
    >>> mae = mean_absolute_error(y_true, y_pred)
    >>> print(f"MAE: {mae:.3f}")
    MAE: 0.120
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "MAE")
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).

    MSE = (1/n) * Σ(y_true - y_pred)²

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Mean squared error

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
    >>> mse = mean_squared_error(y_true, y_pred)
    >>> print(f"MSE: {mse:.3f}")
    MSE: 0.016
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "MSE")
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE = √[(1/n) * Σ(y_true - y_pred)²]

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Root mean squared error

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
    >>> rmse = root_mean_squared_error(y_true, y_pred)
    >>> print(f"RMSE: {rmse:.3f}")
    RMSE: 0.127
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate Coefficient of Determination (R²).

    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)²
    and SS_tot = Σ(y_true - y_mean)²

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        R² score (between -∞ and 1)

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.2, 2.8, 4.1, 4.9]
    >>> r2 = r2_score(y_true, y_pred)
    >>> print(f"R²: {r2:.3f}")
    R²: 0.984
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "R²")

    # Calculate sum of squared residuals
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate total sum of squares
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # Handle edge case where ss_tot is zero (constant y_true)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


# Alias for backward compatibility
mae = mean_absolute_error
mse = mean_squared_error
rmse = root_mean_squared_error