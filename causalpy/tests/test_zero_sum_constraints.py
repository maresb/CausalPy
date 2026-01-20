import numpy as np


def _fit_zero_sum_seasonality_no_intercept(
    y: np.ndarray, *, category_idx: np.ndarray, n_categories: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit y ~= seasonal[category_idx] with constraint sum(seasonal)=0 and NO intercept.

    Parameterization: seasonal[K-1] = -sum(seasonal[:K-1]).
    """
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if category_idx.shape != y.shape:
        raise ValueError("category_idx must have same shape as y")
    if n_categories < 2:
        raise ValueError("n_categories must be >= 2")

    n_obs = y.shape[0]
    k = n_categories

    # Reduced design matrix B (n_obs x (k-1)):
    # - if category == i (i<k-1): B[t, i] = 1
    # - if category == k-1:       B[t, :] = -1
    b = np.zeros((n_obs, k - 1), dtype=float)
    for t, c in enumerate(category_idx.tolist()):
        if not (0 <= c < k):
            raise ValueError("category_idx entries must be in [0, n_categories)")
        if c < k - 1:
            b[t, c] = 1.0
        else:
            b[t, :] = -1.0

    beta_hat, *_ = np.linalg.lstsq(b, y, rcond=None)
    y_hat = b @ beta_hat

    seasonal_hat = np.empty(k, dtype=float)
    seasonal_hat[: k - 1] = beta_hat
    seasonal_hat[k - 1] = -float(beta_hat.sum())
    return y_hat, seasonal_hat


def test_zero_sum_constraint_orthogonal_complement_is_constant_component() -> None:
    """
    If you constrain seasonality to be sum-to-zero and omit the intercept, the
    "constant/DC" component (vector of ones) cannot be fit.

    This constructs a signal y = intercept + seasonal[category_idx] where seasonal
    already has sum 0. The optimal constrained fit recovers the seasonal part, leaving
    a constant residual equal to the intercept.
    """
    k = 12
    category_idx = np.arange(k, dtype=int)  # perfectly balanced: each category once

    seasonal_true = np.arange(k, dtype=float)
    seasonal_true -= seasonal_true.mean()  # sum(seasonal_true)=0
    assert np.isclose(seasonal_true.sum(), 0.0)

    intercept_true = 2.5
    y = intercept_true + seasonal_true[category_idx]

    y_hat, seasonal_hat = _fit_zero_sum_seasonality_no_intercept(
        y, category_idx=category_idx, n_categories=k
    )

    # The model cannot represent the constant (all-ones) component.
    residual = y - y_hat
    assert np.allclose(residual, intercept_true)

    # In coefficient space, the fitted seasonal lives in the zero-sum subspace.
    assert np.isclose(seasonal_hat.sum(), 0.0)

    # In observation space (balanced design), the all-ones vector is orthogonal
    # to the seasonal column space, so the residual must be constant.
    ones = np.ones_like(y)
    assert np.isclose(float(np.dot(ones, y_hat)), 0.0)

    # Adding an intercept makes the model able to fit the constant component exactly.
    # Use the same reduced design b as in the fitter to keep the parameterization.
    b = np.zeros((k, k - 1), dtype=float)
    for t, c in enumerate(category_idx.tolist()):
        if c < k - 1:
            b[t, c] = 1.0
        else:
            b[t, :] = -1.0

    x = np.column_stack([np.ones(k, dtype=float), b])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat_with_intercept = x @ coef
    assert np.allclose(y_hat_with_intercept, y)

